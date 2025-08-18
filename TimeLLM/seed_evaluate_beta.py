import os
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
print(f'CUDA_VISIBLE_DEVICES: {cuda_visible_devices}')
import git
import gc
import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import pmdarima as pm
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string

from models import Autoformer, DLinear, TimeMamba, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np

import pandas as pd
from utils.metrics import metric
import matplotlib.pyplot as plt
import wandb 
from torchsummary import summary

import sys
sys.path.insert(0, "/home/nesl/oliver/timeSeriesMamba/Mamba4Cast/src_torch")
from training.models import SSMModel, SSMModelMulti

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content


# ---------- Spectral metrics (no SciPy needed) ----------
import numpy as np
def to_np_f32(t):
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()
    return np.asarray(t, dtype=np.float32)

def _detrend_linear(y):
    t = np.arange(len(y))
    a, b = np.polyfit(t, y, 1)
    return y - (a * t + b)
def spectral_predictability(y, fs=1.0, f_low=None, f_high=None):
    """
    Ω = 1 - (H / log K), with DC removed and proper PSD normalization.
    Returns Ω in [0,1]. High Ω => concentrated spectrum (more forecastable).
    """
    y = np.asarray(y, float)
    if y.size < 8:
        return np.nan

    # detrend + mean-center
    y = _detrend_linear(y)
    y = y - np.nanmean(y)

    # window, rFFT, power
    w = np.hanning(len(y))
    Y = np.fft.rfft(y * w)
    psd = np.abs(Y) ** 2

    # frequency axis + band selection (drop DC)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / fs)
    lo = 0 if f_low is None else int(np.searchsorted(freqs, f_low, side='left'))
    hi = len(freqs) if f_high is None else int(np.searchsorted(freqs, f_high, side='right'))
    lo = max(lo, 1)  # exclude DC bin
    lo = min(lo, len(freqs))
    hi = min(max(hi, lo + 1), len(freqs))

    band = psd[lo:hi]
    K = band.size
    if K == 0:
        return np.nan

    s = float(band.sum())
    if s <= 0.0:
        # degenerate case: treat as fully concentrated (Ω=1)
        return 1.0

    # probability normalization over band
    p = band / s

    # entropy with zero-mask to avoid log(0) bias
    m = p > 0
    H = float(-np.sum(p[m] * np.log(p[m])))
    H_max = np.log(K)
    if H_max <= 0:
        return np.nan

    H_norm = np.clip(H / H_max, 0.0, 1.0)
    Omega = float(np.clip(1.0 - H_norm, 0.0, 1.0))
    return Omega

def largest_lyapunov_rosenstein(y, m=6, tau=None, max_t=50, theiler=10, slope_window=None, nonneg=False):
    """
    Rosenstein LLE estimate from a univariate series y.
    Returns the slope of mean log-distance growth (per-sample rate).
      >0: divergence (chaotic), <0: convergence (stable/periodic).
    Args:
      m: embedding dimension
      tau: delay (if None, pick by first local minimum of ACF; fallback=1)
      max_t: maximum forward steps for divergence curve
      theiler: temporal exclusion window when picking nearest neighbors
      slope_window: optional (t_min, t_max) fit range; default = middle 50%
      nonneg: if True, return max(0, slope)
    """
    y = np.asarray(y, dtype=float)
    if y.size < 50:
        return np.nan

    # normalize
    mu = np.nanmean(y)
    sig = np.nanstd(y)
    if not np.isfinite(sig) or sig <= 0:
        return np.nan
    y = (y - mu) / sig

    # pick tau if needed via first local minimum of ACF
    if tau is None:
        acf = np.correlate(y, y, mode='full')[len(y)-1:]
        if acf[0] <= 0:
            tau = 1
        else:
            acf = acf / (acf[0] + 1e-12)
            tau = 1
            # search up to N/3 but not beyond 200
            upper = min(200, len(y)//3)
            for lag in range(2, max(3, upper)):
                if acf[lag-1] > acf[lag] < acf[lag+1]:
                    tau = lag
                    break

    # build delay embedding
    N = len(y) - (m - 1) * tau
    if N < max(50, 2 * m * tau):
        return np.nan
    X = np.empty((N, m), dtype=float)
    for k in range(m):
        X[:, k] = y[k * tau : k * tau + N]

    # nearest neighbors with Theiler window
    d0, nn_idx = [], []
    idx_all = np.arange(N)
    for i in range(N):
        lo = max(0, i - theiler)
        hi = min(N, i + theiler + 1)
        mask = np.ones(N, dtype=bool)
        mask[lo:hi] = False
        if not mask.any():
            continue
        D = np.linalg.norm(X[mask] - X[i], axis=1)
        j_rel = np.argmin(D)
        j = idx_all[mask][j_rel]
        dij0 = D[j_rel]
        if not np.isfinite(dij0) or dij0 <= 0:
            continue
        d0.append(dij0)
        nn_idx.append(int(j))

    if len(d0) < 10:
        return np.nan

    # usable horizon (both trajectories must remain inside)
    max_t = min(int(max_t), int(N - 1 - np.max(nn_idx)))
    if max_t < 5:
        return np.nan

    # average log divergence curve
    ts = np.arange(1, max_t + 1, dtype=int)
    lns = np.full_like(ts, fill_value=np.nan, dtype=float)
    d0 = np.asarray(d0, dtype=float)
    for ti, t in enumerate(ts):
        vals = []
        for i, j in enumerate(nn_idx):
            ii = i
            jj = j
            if ii + t >= N or jj + t >= N:
                continue
            di = np.linalg.norm(X[ii + t] - X[jj + t])
            if np.isfinite(di) and di > 0:
                vals.append(np.log(di) - np.log(d0[i]))
        if vals:
            lns[ti] = float(np.mean(vals))

    msk = np.isfinite(lns)
    if msk.sum() < 5:
        return np.nan

    # choose fit window
    if slope_window is not None:
        tmin, tmax = slope_window
        use = (ts >= int(tmin)) & (ts <= int(tmax)) & msk
    else:
        q1 = int(0.25 * max_t)
        q3 = int(0.75 * max_t)
        use = (ts >= max(1, q1)) & (ts <= max(q1 + 4, q3)) & msk

    if use.sum() < 5:
        return np.nan

    A = np.vstack([ts[use], np.ones(use.sum())]).T
    slope, _ = np.linalg.lstsq(A, lns[use], rcond=None)[0]
    slope = float(slope)
    return max(0.0, slope) if nonneg else slope

def batch_forecastability(ctx, fut):
    """
    ctx: [B, L, D], fut: [B, H, D]
    Returns per-batch means of Ω and LLE for context and future.
    """
    ctx = _to_np(ctx); fut = _to_np(fut)
    B, L, D = ctx.shape
    om_ctx = []; om_fut = []; lle_ctx = []; lle_fut = []
    for b in range(B):
        for d in range(D):
            x = ctx[b, :, d]; y = fut[b, :, d]
            om_ctx.append(spectral_predictability(x))
            om_fut.append(spectral_predictability(y))
            lle_ctx.append(largest_lyapunov_rosenstein(x, m=6, tau=None, max_t=50, theiler=10))
            lle_fut.append(largest_lyapunov_rosenstein(y, m=6, tau=None, max_t=50, theiler=10))
    # robust aggregates
    return {
        "Omega_ctx_mean": float(np.nanmean(om_ctx)),
        "Omega_future_mean": float(np.nanmean(om_fut)),
        "LLE_ctx_mean": float(np.nanmean(lle_ctx)),
        "LLE_future_mean": float(np.nanmean(lle_fut)),
        "Omega_ctx_med": float(np.nanmedian(om_ctx)),
        "Omega_future_med": float(np.nanmedian(om_fut)),
        "LLE_ctx_med": float(np.nanmedian(lle_ctx)),
        "LLE_future_med": float(np.nanmedian(lle_fut)),
    }
# ---- Cross-dataset safe metrics ----
def _to_np2(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def MASE_batched(pred, true, insample, seasonality=1, eps=1e-12, reduce="mean"):
    """
    pred,true: [B, H, D]
    insample:  [B, L, D]  (context; used for denominator)
    seasonality: lag m (e.g., 1 non-seasonal, 24 for hourly with daily seasonality)
    returns scalar unless reduce=None (then [B,D])
    """
    pred  = _to_np2(pred)
    true  = _to_np2(true)
    ins   = _to_np2(insample)
    assert pred.shape == true.shape, f"{pred.shape} != {true.shape}"
    B, H, D = true.shape
    m = int(seasonality)
    if m < 1:
        raise ValueError("seasonality must be >= 1")

    # denominator from in-sample context
    denom = np.full((B, D), np.nan, dtype=np.float64)
    for b in range(B):
        for d in range(D):
            y = ins[b, :, d].astype(np.float64)
            y = y[~np.isnan(y)]
            if y.size < 2:
                continue
            if m >= y.size:
                diffs = np.abs(np.diff(y))
            else:
                diffs = np.abs(y[m:] - y[:-m])
            if diffs.size:
                denom[b, d] = diffs.mean()

    # numerator: MAE over horizon
    mae = np.nanmean(np.abs(pred - true), axis=1)  # [B,D]

    denom = np.where(~np.isfinite(denom) | (denom < eps), eps, denom)
    mase = mae / denom  # [B,D]
    return float(np.nanmean(mase)) if reduce == "mean" else mase

def sMAPE_batched(pred, true, eps=1e-12):
    """
    sMAPE in [0,2]; commonly reported as % by *100.
    pred,true: [B, H, D] or broadcastable thereto
    """
    pred = _to_np2(pred); true = _to_np2(true)
    num = np.abs(pred - true)
    den = (np.abs(true) + np.abs(pred)) + eps
    smape = 2.0 * num / den  # [B,H,D]
    return float(np.nanmean(smape))

def _to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _band_lims(fs, N, f_low, f_high):
    # bin freqs for rfft
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    lo = 0 if f_low is None else np.searchsorted(freqs, f_low, side='left')
    hi = len(freqs) if f_high is None else np.searchsorted(freqs, f_high, side='right')
    lo = max(lo, 1)  # drop DC by default
    hi = max(hi, lo+1)
    return lo, hi, freqs

def WAPE(pred, true, eps=1e-8):
    # Convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    pred = np.squeeze(pred)
    true = np.squeeze(true)

    if true.ndim == 1:
        true = true[np.newaxis, :]
        pred = pred[np.newaxis, :]

    numerator = np.sum(np.abs(true - pred))
    denominator = np.sum(np.abs(true))
    return numerator / max(denominator, eps)


def spectral_entropy_1d(x, fs=1.0, f_low=None, f_high=None, eps=1e-12, detrend='mean'):
    """
    Normalized spectral entropy in [0,1] for one series x.
    - fs: sampling rate; for hourly, fs=1.0
    - f_low/f_high: optional bandlimits; if None, uses all (except DC)
    - detrend: 'mean' (remove mean), 'lin' (remove best-fit line), or None
    """
    x = np.asarray(x).astype(np.float64)
    N = x.shape[0]
    if N < 8:  # too short; return NaN to avoid nonsense
        return np.nan

    if detrend == 'mean':
        x = x - np.nanmean(x)
    elif detrend == 'lin':
        t = np.arange(N)
        A = np.vstack([t, np.ones(N)]).T
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        x = x - (m * t + b)

    # Hann window to reduce leakage (keeps things lightweight)
    w = np.hanning(N)
    xw = x * w

    X = np.fft.rfft(xw, n=N)
    Pxx = (np.abs(X) ** 2)  # power spectrum (unnormalized)

    lo, hi, _ = _band_lims(fs, N, f_low, f_high)
    band = Pxx[lo:hi].astype(np.float64)
    s = band.sum()
    if s <= eps:
        return 0.0  # no power in band ⇒ "fully regular" by this measure

    p = band / (s + eps)         # normalize to probability mass
    H = -np.sum(p * np.log(p + eps))
    H_max = np.log(len(p))
    return float(H / (H_max + eps))

def seasonality_strength_1d(x, fs=1.0, f_peak_window=None, detrend='mean'):
    """
    Simple seasonality strength proxy in [0,1]: peak power / (peak power + rest).
    - f_peak_window: (f_lo, f_hi) to look for the dominant seasonal band (e.g., around 1/day).
      If None, just takes global max (excluding DC).
    """
    x = np.asarray(x).astype(np.float64)
    N = x.shape[0]
    if N < 8:
        return np.nan
    if detrend == 'mean':
        x = x - np.nanmean(x)
    elif detrend == 'lin':
        t = np.arange(N)
        A = np.vstack([t, np.ones(N)]).T
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        x = x - (m * t + b)

    w = np.hanning(N)
    X = np.fft.rfft(x * w, n=N)
    Pxx = (np.abs(X) ** 2)
    lo, hi, freqs = _band_lims(fs, N, f_low=None, f_high=None)

    if f_peak_window is not None:
        flo, fhi = f_peak_window
        lo = max(lo, np.searchsorted(freqs, flo, side='left'))
        hi = min(hi, np.searchsorted(freqs, fhi, side='right'))

    band = Pxx[lo:hi]
    if band.size == 0:
        return np.nan
    peak = float(np.max(band))
    rest = float(band.sum() - peak)
    if peak <= 0:
        return 0.0
    return peak / (peak + rest + 1e-12)

def batch_spectral_metrics(ctx, fut, fs=1.0, f_low=None, f_high=None, detrend='mean',
                           season_band=None):
    """
    ctx: np array [B, L, D] context window (what model sees)
    fut: np array [B, H, D] future ground truth window
    Returns dict of per-batch aggregates and (optionally) per-d feature arrays.
    """
    ctx = _to_np(ctx)
    fut = _to_np(fut)
    B, L, D = ctx.shape
    _, H, D2 = fut.shape
    assert D == D2

    se_ctx = np.full((B, D), np.nan)
    se_fut = np.full((B, D), np.nan)
    seas_ctx = np.full((B, D), np.nan)
    seas_fut = np.full((B, D), np.nan)

    # choose f_low based on window to avoid under-resolved low freq
    # if not provided: drop < 1/L for ctx, < 1/H for fut by passing None here and relying on detrend+DC drop
    for b in range(B):
        for d in range(D):
            x_ctx = ctx[b, :, d]
            x_fut = fut[b, :, d]

            se_ctx[b, d]  = spectral_entropy_1d(x_ctx, fs=fs, f_low=f_low, f_high=f_high, detrend=detrend)
            se_fut[b, d]  = spectral_entropy_1d(x_fut, fs=fs, f_low=f_low, f_high=f_high, detrend=detrend)
            seas_ctx[b, d] = seasonality_strength_1d(x_ctx, fs=fs, f_peak_window=season_band, detrend=detrend)
            seas_fut[b, d] = seasonality_strength_1d(x_fut, fs=fs, f_peak_window=season_band, detrend=detrend)

    out = {
        "SE_ctx_mean": float(np.nanmean(se_ctx)),
        "SE_future_mean": float(np.nanmean(se_fut)),
        "SE_ctx_med": float(np.nanmedian(se_ctx)),
        "SE_future_med": float(np.nanmedian(se_fut)),
        "Season_ctx_mean": float(np.nanmean(seas_ctx)),
        "Season_future_mean": float(np.nanmean(seas_fut)),
    }
    # also return per-d arrays for optional per-feature logging
    out_arrays = {
        "SE_ctx_per_dim": se_ctx, "SE_future_per_dim": se_fut,
        "Season_ctx_per_dim": seas_ctx, "Season_future_per_dim": seas_fut
    }
    return out, out_arrays

def quick_stats(x):
    """Return variance and a rough 'SNR-like' proxy (var/mean(|Δx|))."""
    x = _to_np(x)
    var = float(np.var(x))
    dif = np.abs(np.diff(x, axis=-2)).mean() if x.ndim >= 2 and x.shape[-2] > 1 else np.nan
    snr_like = float(var / (dif + 1e-12)) if not np.isnan(dif) else np.nan
    return var, snr_like


def MASE(pred, true, seasonality=1, eps=0):
    """
    Robust MASE:
    - pred, true: torch.Tensor or np.ndarray. Shapes supported: (B,T), (B,T,1), (T,), (T,1)
    - seasonality: integer lag m
    - returns mean MASE across series in batch
    """
    # convert torch -> numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    pred = np.squeeze(pred)
    true = np.squeeze(true)

    # ensure 2D: (B, T)
    if true.ndim == 1:
        true = true[np.newaxis, :]
        pred = pred[np.newaxis, :]
    elif true.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected true.ndim={true.ndim}, expected 1 or 2 after squeeze")

    B, T = true.shape
    m = int(seasonality)

    # compute denominator: mean absolute difference at lag m per series
    if m < 1:
        raise ValueError("seasonality must be >= 1")
    if m >= T:
        # cannot compute seasonal naive; fallback to one-step diff
        diffs = np.abs(true[:, 1:] - true[:, :-1])  # shape (B, T-1)
    else:
        diffs = np.abs(true[:, m:] - true[:, :-m])  # shape (B, T-m)

    denom = diffs.mean(axis=1)  # per-series denom, shape (B,)

    # numerator: mean abs error over forecast horizon, per series
    # if pred and true shapes differ in T, align on last axis
    if pred.shape[1] != T:
        raise ValueError(f"pred T ({pred.shape[1]}) != true T ({T})")
    mae = np.mean(np.abs(pred - true), axis=1)  # shape (B,)

    # avoid division by zero; use eps for stability
    denom_safe = np.where(denom < eps, eps, denom)
    mase_per_series = mae / denom_safe

    return mase_per_series.mean()

# Validation function
def vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric):
    model.eval()
    total_loss, total_mae_loss, total_mase_loss, total_smape_loss = [], [], [], []
    se_ctx_list, se_fut_list = [], []
    seas_ctx_list, seas_fut_list = [], []
    var_ctx_list, snr_ctx_list = [], []
    var_fut_list, snr_fut_list = [], []
    omega_ctx_list, omega_fut_list = [], []
    lle_ctx_list,   lle_fut_list   = [], []

    fs = 1.0
    season_band = (1/30.0, 1/20.0) if args.freq in ['h', 'H'] else None
    detrend_mode = 'mean'
    # Choose seasonality for MASE (adjust if you truly want daily seasonality)
    mase_m = 24 if args.freq in ['h', 'H'] else 1

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)

            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]  # [B,H,D]
            gt = batch_y[:, -args.pred_len:, f_dim:]       # [B,H,D]

            # safe casts for numpy
            pred_np = to_np_f32(outputs)     # was: outputs.detach().cpu().numpy()
            fut_np  = to_np_f32(gt)          # was: gt.detach().cpu().numpy()
            ctx_np  = to_np_f32(batch_x[:, :args.seq_len, f_dim:])

            # Forecastability (Ω & LLE)
            fcast = batch_forecastability(ctx_np, fut_np)
            omega_ctx_list.append(fcast["Omega_ctx_mean"])
            omega_fut_list.append(fcast["Omega_future_mean"])
            lle_ctx_list.append(fcast["LLE_ctx_mean"])
            lle_fut_list.append(fcast["LLE_future_mean"])

            # Spectral metrics
            spec_dict, _ = batch_spectral_metrics(
                ctx_np, fut_np, fs=fs, f_low=None, f_high=None,
                detrend=detrend_mode, season_band=season_band
            )
            se_ctx_list.append(spec_dict["SE_ctx_mean"])
            se_fut_list.append(spec_dict["SE_future_mean"])
            seas_ctx_list.append(spec_dict["Season_ctx_mean"])
            seas_fut_list.append(spec_dict["Season_future_mean"])

            v_ctx, snr_ctx = quick_stats(ctx_np)
            v_fut, snr_fut = quick_stats(fut_np)
            var_ctx_list.append(v_ctx); snr_ctx_list.append(snr_ctx)
            var_fut_list.append(v_fut); snr_fut_list.append(snr_fut)

            # Base losses
            loss = criterion(outputs, gt)
            mae_loss = mae_metric(outputs, gt)

            # Scale-free cross-dataset metrics
            mase_loss  = MASE_batched(pred_np, fut_np, insample=ctx_np, seasonality=mase_m)
            smape_loss = sMAPE_batched(pred_np, fut_np)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            total_mase_loss.append(mase_loss)
            total_smape_loss.append(smape_loss)

    avg_loss  = float(np.mean(total_loss))
    avg_mae   = float(np.mean(total_mae_loss))
    avg_mase  = float(np.mean(total_mase_loss))
    avg_smape = float(np.mean(total_smape_loss))

    Omega_ctx = float(np.nanmean(omega_ctx_list)) if omega_ctx_list else np.nan
    Omega_fut = float(np.nanmean(omega_fut_list)) if omega_fut_list else np.nan
    LLE_ctx   = float(np.nanmean(lle_ctx_list))   if lle_ctx_list   else np.nan

    se_ctx = float(np.nanmean(se_ctx_list)) if se_ctx_list else np.nan
    se_fut = float(np.nanmean(se_fut_list)) if se_fut_list else np.nan
    seas_ctx = float(np.nanmean(seas_ctx_list)) if seas_ctx_list else np.nan
    seas_fut = float(np.nanmean(seas_fut_list)) if seas_fut_list else np.nan
    var_ctx = float(np.nanmean(var_ctx_list)) if var_ctx_list else np.nan
    var_fut = float(np.nanmean(var_fut_list)) if var_fut_list else np.nan
    snr_ctx = float(np.nanmean(snr_ctx_list)) if snr_ctx_list else np.nan
    snr_fut = float(np.nanmean(snr_fut_list)) if snr_fut_list else np.nan

    if getattr(accelerator, "is_local_main_process", True) and args.use_wandb:
        wandb.log({
            "MSE loss": avg_loss,
            "MAE loss": avg_mae,
            "MASE loss": avg_mase,
            "sMAPE": avg_smape,           # replace WAPE with sMAPE
            "SE_ctx_mean": se_ctx,
            "Season_ctx_mean": seas_ctx,
            "Var_ctx": var_ctx,
            "SNR_ctx_proxy": snr_ctx,
            "Omega_ctx_mean": Omega_ctx,
            "LLE_ctx_mean": LLE_ctx
        })

    return avg_loss, avg_mae, avg_mase, avg_smape

import torch.nn.functional as F

class TorchRidge(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.alpha = alpha

    def forward(self, x):
        return self.linear(x)

    def ridge_loss(self, pred, target):
        mse = F.mse_loss(pred, target)
        l2 = self.alpha * torch.sum(self.linear.weight ** 2)
        return mse + l2
        
def visualize_example(args, accelerator, model, test_loader):
    if not accelerator.is_local_main_process:
        return

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            # Check for NaN in target data
            if torch.isnan(batch_y).any():
                print("Skipping batch with NaN in batch_y")
                continue

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)

            # Get model predictions
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Check for NaN in predictions
            if torch.isnan(outputs).any():
                print("Model outputs contain NaN for this batch")
                continue

            seq_len, pred_len = args.seq_len, args.pred_len
            f = batch_x.shape[-1]  # Number of features (1 for univariate)

            # Extract data for visualization
            ctx  = to_np_f32(batch_x[0, :seq_len, :f])
            gt   = to_np_f32(batch_y[0, -pred_len:, :f])
            pred = to_np_f32(outputs[0, -pred_len:, :f])
            
            # Final NaN check on pred
            if np.isnan(pred).any():
                print("Predictions contain NaN for this sample")
                continue

            # Construct actual and predicted arrays
            T = seq_len + pred_len
            actual = np.zeros((T, f))
            actual[:seq_len] = ctx
            actual[seq_len:] = gt

            predicted = np.full((T, f), np.nan)
            predicted[seq_len:] = pred

            # Create CSV data
            feature_names = [args.source] if f == 1 else [f'{args.source}_{i}' for i in range(f)]
            data = {}
            for i, name in enumerate(feature_names):
                data[f'{name}_actual'] = actual[:, i]
                data[f'{name}_pred'] = predicted[:, i]

            df = pd.DataFrame(data, index=np.arange(T))
            csv_path = f'visuals/visualize_{args.model_id}_{args.llm_model}_{args.source}Source_randinit{args.rand_init}_h{args.heldout}_seed{args.seed}_initseed{args.init_seed}.csv'
            df.to_csv(csv_path, index_label='time_step')
            print(f"Visualization saved to {csv_path}")
            break  # Process only one valid batch

if __name__ == '__main__':
    # Argument parser

    parser = argparse.ArgumentParser(description='Time-LLM')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    #parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    #parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, DLinear]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--checkpoint_path', type=str, required=True, default='None', help='where trained model is stored')
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--data_pretrain', type=str, default='None', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--data_path_test', type=str, default='None', help='data file, make sure is set when cov split')
    parser.add_argument('--data_path_val', type=str, default='None', help='data file for covariate split')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                            'M:multivariate predict multivariate, S: univariate predict univariate, '
                            'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                            'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                            'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--pretrain', type=int, default=0)

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--dsampfactor', type=int, default=1, help='for downsampling purposes')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='Mamba', help='LLM model') # LLAMA, GPT2, BERT, Mamba
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')#Mamba:768 LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--num_params', type=str, default='130m', help='string of our param size to append to huggingface')
    parser.add_argument('--rand_init', type=int, default=0, help='if nonzero, initialize weights of LLM randomly')
    parser.add_argument('--init_seed', type=int, default=0, help='seed for rand_init only')
    parser.add_argument('--finetune_llm', type=int, default=0, help='if nonzero, allow LLM weights to be trained')
    parser.add_argument('--boundary_file', type=str, default=None, help='if not None, prevents training windows to be takena cross a concatenated datafile')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--col_percent', type=int, default=100)
    parser.add_argument('--train_percent', type=int, default=100)
    parser.add_argument('--split_type', type=str, default="temporal")
    parser.add_argument('--source', type=str, default="None")
    parser.add_argument('--heldout', type=str, default="None")

    parser.add_argument('--visualize', action='store_true', help='visualize a test example after training')
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    #parser.add_argument('--saveName',type=str,default="NULL",help='for smooth pipelining')
    parser.add_argument('--early_break', type=int, default=0)
    parser.add_argument('--save_checkpoints', type=int, default=0)

    parser.add_argument('--use_classical_model', action='store_true', help='Use classical model like AutoARIMA/VAR')

    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    
    if args.llm_model == "Moirai":
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./float_ds_config.json')
    else:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    # Initialize Accelerator
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    print("accelerator device: ", accelerator.device)
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

     # Initialize wandb if enabled
    if args.use_wandb:
        try:
            repo = git.Repo(search_parent_directories=True)
            commit_hash = repo.head.object.hexsha

            wandb.init(project = 'TimeMamba')
            #wandb.config.update(args)
            wandb.config.update({
            'git_commit': commit_hash,
            'layer count': args.llm_layers,
            'd_model': args.d_model,
            'train epochs': args.train_epochs,
            'model id': args.model_id,
            'model' : args.model,
            'LLM used': args.llm_model+"_LLM",
            'dsampfactor': args.dsampfactor,
            'percent': args.percent,
            'col_percent': args.col_percent,
            'train_percent': args.train_percent,
            'rand_init': args.rand_init,
            'seed': args.seed,
            'init_seed': args.init_seed,
            'pred_len': args.pred_len,
            'seq_len': args.seq_len, 
            'pretrain': args.pretrain,
            'finetune_llm': args.finetune_llm,
            'split_type': args.split_type,
            'source': args.source,
            'heldout': args.heldout
        })
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            args.use_wandb = 0

    # Load test data
    test_data, test_loader = data_provider(args, 'test')

    # Initialize the model
    if args.model == 'TimeLLM':
        model = TimeLLM.Model(args).float()
    elif args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'Ridge':
        model = DLinear.Model(args).float() #but realy we're going to overwrite this
    else:
        raise ValueError(f"Unknown model: {args.model}")

    trained_parameters = []
    
    train_steps = len(test_loader) #changed from train
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)


    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    if not args.llm_model == "Ridge":
        # Load model weights from checkpoint
        model.load_state_dict(torch.load(args.checkpoint_path,  map_location=lambda storage, loc: storage))
        
    test_loader,model,model_optim = accelerator.prepare(test_loader,model,model_optim)

    if args.llm_model == "Ridge":
        in_dim = args.seq_len * args.enc_in
        out_dim = args.pred_len * args.dec_in
        model = TorchRidge(in_dim, out_dim, alpha=1.0).to(accelerator.device)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        X_list, Y_list = [], []
        for batch_x, batch_y, _, _ in test_loader:
            X_list.append(batch_x.view(batch_x.size(0), -1).to(accelerator.device))  # [B, T, D] → [B, T*D]
            Y_list.append(batch_y[:, -args.pred_len:, :].reshape(batch_y.size(0), -1).to(accelerator.device))

        X = torch.cat(X_list, dim=0)
        Y = torch.cat(Y_list, dim=0)

        for _ in range(args.train_epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = model.ridge_loss(pred, Y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X)
            mse_loss = F.mse_loss(pred, Y).item()
            mae_loss = F.l1_loss(pred, Y).item()

        print(f"[TorchRidge] MSE: {mse_loss}")
        print(f"[TorchRidge] MAE: {mae_loss}")
        if args.use_wandb:
            wandb.log({"MSE loss": mse_loss, "MAE loss": mae_loss})
            wandb.finish()

        if args.visualize:
            batch_idx = 0
            x_seq = X[batch_idx].reshape(args.seq_len, args.enc_in).cpu().numpy()
            y_true_seq = Y[batch_idx].reshape(args.pred_len, args.dec_in).cpu().numpy()
            y_pred_seq = pred[batch_idx].reshape(args.pred_len, args.dec_in).cpu().numpy()

            T = args.seq_len + args.pred_len
            f = args.dec_in

            actual = np.zeros((T, f))
            actual[:args.seq_len] = x_seq
            actual[args.seq_len:] = y_true_seq

            predicted = np.full((T, f), np.nan)
            predicted[args.seq_len:] = y_pred_seq

            feature_names = ['coal', 'nat_gas', 'nuclear', 'oil', 'hydro', 'solar', 'wind', 'other']
            data = {}
            for i, name in enumerate(feature_names[:f]):
                data[f'{name}_actual'] = actual[:, i]
                data[f'{name}_pred'] = predicted[:, i]

            timestep_type = ['context'] * args.seq_len + ['prediction'] * args.pred_len
            data['timestep_type'] = timestep_type

            df = pd.DataFrame(data, index=np.arange(T))
            os.makedirs('visuals', exist_ok=True)
            csv_path = f'visuals/visualize_{args.model_id}_ridge_seed{args.seed}.csv'
            df.to_csv(csv_path, index_label='time_step')
            print(f"[Ridge] Visualization saved to {csv_path}")
        exit()


    
    
    # Define loss metrics
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

   

    earlyUnwrap = accelerator.unwrap_model(model)
    num_params=sum(p.numel() for p in earlyUnwrap.parameters())
    print(f'Total number of parameters: {num_params}')
    if args.use_wandb:
        wandb.config.update({'num_params':num_params})

    # Run evaluation
    test_loss, test_mae_loss, test_mase_loss, test_smape_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
    print(f"MSE loss: {test_loss}")
    print(f"MAE loss: {test_mae_loss}")
    print(f"MASE loss: {test_mase_loss}")
    print(f"sMAPE loss: {test_smape_loss}")
    # Visualize a test example if requested
    if args.visualize:
        visualize_example(args, accelerator, model, test_loader)
    # Log metrics to wandb if enabled
    if args.use_wandb:
        wandb.log({"MSE loss": test_loss, "MAE loss": test_mae_loss, "MASE loss": test_mase_loss, "sMAPE loss": test_smape_loss})
        wandb.finish()