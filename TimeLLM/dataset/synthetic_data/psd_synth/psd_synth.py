#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic time series with controllable spectral predictability Ω.

Outputs:
- train_val.csv : concatenated [all train chunks (80%) for each region] + [all val chunks (20%)]
- train_boundaries.json :
    {
      "boundaries": [[start_idx, end_idx], ...],
      "regions": ["Region 1", ..., "Region K", "Region 1", ..., "Region K"]
    }
  (Train boundaries first, then val boundaries, matching the concatenation order.)
- One CSV per held-out test region: region_test_om{Ω}.csv
- Calibration plots + metadata for traceability.

To reduce total length/time:
- YEARS: shorten (e.g., 5 → 0.5)
- SAMPLE_EVERY_HOURS: increase (e.g., 1 → 3)
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- Core metrics ---------------------------

def spectral_predictability(y: np.ndarray) -> float:
    """
    Ω in [0,1] via normalized spectral entropy of Hann-windowed, de-trended signal.
    Ω = 1 - H/ln(K), where K is # of rFFT bins (excluding symmetry).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 8:
        return np.nan

    t = np.arange(n)
    # remove best-fit line
    a, b = np.polyfit(t, y, 1)
    y = y - (a * t + b)

    w = np.hanning(n)
    Y = np.fft.rfft((y - y.mean()) * w, n=n)
    P = (Y.real**2 + Y.imag**2).clip(1e-20, None)
    p = P / P.sum()

    H = -np.sum(p * np.log(p))
    Hmax = np.log(len(p))
    return float(np.clip(1.0 - H / (Hmax + 1e-12), 0.0, 1.0))


# ------------------------ PSD construction -------------------------

def _clamp_bins(bins: List[int], n: int) -> np.ndarray:
    """Clamp rFFT bin indices into [1, n//2] and unique-sort."""
    b = np.asarray(bins, dtype=int)
    b = np.clip(b, 1, n // 2)
    return np.unique(b)

def make_psd(alpha: float, n: int, peak_bins: List[int], width_bins: float = 2.0) -> np.ndarray:
    """
    PSD = (1 - alpha) * flat + alpha * sum of narrow peaks (both unit mass).
    """
    K = n // 2 + 1
    k = np.arange(K)

    # flat with unit mass
    psd_flat = np.ones(K, dtype=np.float64)
    psd_flat[0] = 1e-12  # suppress DC
    psd_flat /= psd_flat.sum()

    # peaks with unit mass
    psd_peaks = np.zeros(K, dtype=np.float64)
    pb = _clamp_bins(peak_bins, n)
    for b in pb:
        psd_peaks += np.exp(-0.5 * ((k - b) / width_bins) ** 2)
    psd_peaks[0] = 0.0
    psd_peaks = np.clip(psd_peaks, 0.0, None)
    psd_peaks /= psd_peaks.sum() + 1e-12

    # exact mass split
    psd = (1.0 - alpha) * psd_flat + alpha * psd_peaks
    return psd


def synth_from_psd(psd: np.ndarray, n_out: int, seed: int | None = None) -> np.ndarray:
    """
    Sample a real-valued series whose magnitude spectrum matches psd (in expectation).
    Use irfft(..., n=n_out) so we exactly control the output length, even when n_out is odd.
    """
    rng = np.random.default_rng(seed)
    mag = np.sqrt(psd * psd.size)  # keep variance O(1)
    phase = rng.uniform(0, 2 * np.pi, size=psd.size)
    spec = mag * (np.cos(phase) + 1j * np.sin(phase))
    y = np.fft.irfft(spec, n=n_out)  # ← use n_out, not (psd.size - 1) * 2
    return y / (y.std() + 1e-12)


# ----------------------- Calibration & inversion --------------------

def calibrate_alpha_to_omega(
    n: int,
    peak_bins: List[int],
    alphas: np.ndarray = np.linspace(0, 1, 11),  # fewer points → faster
    width_bins: float = 2.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-time calibration curve: α -> Ω.
    Return (alpha_grid_sorted_by_omega, omega_grid_sorted).
    """
    pairs = []
    for a in alphas:
        psd = make_psd(a, n, peak_bins, width_bins=width_bins)
        y = synth_from_psd(psd, n, seed=seed)
        omega = spectral_predictability(y)
        pairs.append((a, omega))
    a_grid, om_grid = np.array(pairs).T
    srt = np.argsort(om_grid)  # enforce monotonic ordering by Ω
    return a_grid[srt], om_grid[srt]

def alpha_for_target_omega(omega_target: float, a_grid: np.ndarray, om_grid: np.ndarray) -> float:
    """
    Invert the calibrated curve by linear interpolation in Ω.
    Clips target to calibration range.
    """
    omega_target = float(np.clip(omega_target, om_grid.min(), om_grid.max()))
    return float(np.interp(omega_target, om_grid, a_grid))


# --------------------------- Convenience utils ----------------------

def bins_for_periods(n: int, periods_in_hours: List[float], sample_every_hours: int) -> List[int]:
    """
    Map desired periods (in hours) to rFFT bin indices for a length-n series sampled
    every `sample_every_hours`. With stride S, bin ≈ n * S / P.
    """
    S = float(sample_every_hours)
    bins = [int(round(n * S / P)) for P in periods_in_hours if P > 0]
    return [b for b in bins if 1 <= b <= n // 2]

def omega_tag(x: float) -> str:
    """Format Ω like 0p350 for filenames."""
    return f"{x:.3f}".replace(".", "p")

@dataclass
class RegionMeta:
    region_id: int | str
    role: str            # 'train' or 'test'
    omega_target: float
    alpha_used: float
    omega_achieved: float
    seed: int
    n: int
    periods_hours: List[float]
    width_bins: float
    years: float
    sample_every_hours: int


# ------------------------------- Main --------------------------------

def main():
    # ---- Speed/size knobs ----
    YEARS = 0.5              # ↓ from 5.0 → ~10× shorter
    SAMPLE_EVERY_HOURS = 1   # set to 3 (or 6) for extra 3× (or 6×) reduction

    out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    # total samples N for stride sampling
    hours_total = YEARS * 365.25 * 24
    N = int(round(hours_total / SAMPLE_EVERY_HOURS))  # e.g., 0.5y @ 1h ≈ 4.4k

    # Domain-relevant periodicities (in hours)
    periods_hours = [24, 24 * 7]  # daily, weekly
    peak_bins = bins_for_periods(N, periods_hours, sample_every_hours=SAMPLE_EVERY_HOURS)

    width_bins = 2.0
    alpha_grid_input = np.linspace(0, 1, 11)  # fewer alpha points → faster

    # Targets for Ω (training regions)
    omega_targets_train = np.linspace(0.15, 0.85, 8).tolist()

    # Test Ω (default: midpoints between sorted train Ωs)
    tr_sorted = np.sort(np.array(omega_targets_train))
    omega_targets_test = ((tr_sorted[:-1] + tr_sorted[1:]) / 2.0).tolist()

    train_frac = 0.8  # 80% train / 20% val per region

    # ---- Calibrate α -> Ω once ----
    a_grid, om_grid = calibrate_alpha_to_omega(
        N, peak_bins, alphas=alpha_grid_input, width_bins=width_bins, seed=0
    )
    pd.DataFrame({"alpha": a_grid, "omega": om_grid}).to_csv(
        os.path.join(out_dir, "calibration_alpha_to_omega.csv"), index=False
    )

    plt.figure(figsize=(5, 3))
    plt.plot(a_grid, om_grid, marker="o")
    plt.xlabel("alpha (mixing weight)")
    plt.ylabel("Omega (spectral predictability)")
    plt.title("Calibration: alpha → Ω")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_curve.png"), dpi=150)
    plt.close()

    # ---- Generate TRAIN regions (Ω targets), build train_val.csv + boundaries ----
    train_chunks = []    # DataFrames (train per region)
    val_chunks = []      # DataFrames (val per region)
    boundaries = []      # [[start, end], ...] (train first, then val)
    regions_names = []   # names aligned with boundaries
    train_metas: List[RegionMeta] = []

    split_idx = int(train_frac * N)
    val_len = N - split_idx

    # timestamps for stride sampling
    start_ts = pd.Timestamp("2024-01-01 00:00:00")
    freq_str = f"{SAMPLE_EVERY_HOURS}H"
    all_dates = pd.date_range(start_ts, periods=N, freq=freq_str)

    for ridx, om_t in enumerate(omega_targets_train, start=1):
        a = alpha_for_target_omega(om_t, a_grid, om_grid)
        seed = 100_000 + 17 * ridx  # deterministic per region

        psd = make_psd(a, N, peak_bins, width_bins=width_bins)
        y = synth_from_psd(psd, N, seed=seed)
        om_ach = spectral_predictability(y)

        # split
        y_train = y[:split_idx]
        y_val   = y[split_idx:]

        df_train = pd.DataFrame({"date": all_dates[:split_idx], "synth": y_train})
        df_val   = pd.DataFrame({"date": all_dates[split_idx:], "synth": y_val})

        train_chunks.append(df_train)
        val_chunks.append(df_val)

        # boundaries for train chunks
        start_tr = (ridx - 1) * split_idx
        end_tr   = start_tr + split_idx - 1
        boundaries.append([start_tr, end_tr])
        regions_names.append(f"Region {ridx}")

        train_metas.append(
            RegionMeta(
                region_id=ridx,
                role="train",
                omega_target=float(om_t),
                alpha_used=float(a),
                omega_achieved=float(om_ach),
                seed=seed,
                n=N,
                periods_hours=periods_hours,
                width_bins=width_bins,
                years=YEARS,
                sample_every_hours=SAMPLE_EVERY_HOURS,
            )
        )

    # boundaries for val chunks (after all train chunks)
    val_offset = len(train_chunks) * split_idx
    for ridx in range(1, len(train_chunks) + 1):
        start_v = val_offset + (ridx - 1) * val_len
        end_v   = start_v + val_len - 1
        boundaries.append([start_v, end_v])
        regions_names.append(f"Region {ridx}")

    # Concatenate: all train chunks first, then all val chunks
    train_val_df = pd.concat(train_chunks + val_chunks, ignore_index=True)
    train_val_path = os.path.join(out_dir, "train_val.csv")
    train_val_df.to_csv(train_val_path, index=False)

    # Boundary file (your exact structure)
    boundaries_path = os.path.join(out_dir, "train_boundaries.json")
    with open(boundaries_path, "w") as f:
        json.dump({"boundaries": boundaries, "regions": regions_names}, f, indent=2)

    # Save train metadata
    pd.DataFrame([asdict(m) for m in train_metas]).to_csv(
        os.path.join(out_dir, "train_metadata.csv"), index=False
    )

    # ---- Generate TEST regions (each its own CSV, name includes Ω) ----
    test_metas: List[RegionMeta] = []
    for tidx, om_t in enumerate(omega_targets_test, start=1):
        a = alpha_for_target_omega(om_t, a_grid, om_grid)
        seed = 900_000 + 9973 * tidx  # distinct seed family

        psd = make_psd(a, N, peak_bins, width_bins=width_bins)
        y = synth_from_psd(psd,N, seed=seed)
        om_ach = spectral_predictability(y)

        df_test = pd.DataFrame({"date": all_dates, "synth": y})
        fname = f"region_test_om{omega_tag(om_t)}.csv"
        df_test.to_csv(os.path.join(out_dir, fname), index=False)

        test_metas.append(
            RegionMeta(
                region_id=fname,
                role="test",
                omega_target=float(om_t),
                alpha_used=float(a),
                omega_achieved=float(om_ach),
                seed=seed,
                n=N,
                periods_hours=periods_hours,
                width_bins=width_bins,
                years=YEARS,
                sample_every_hours=SAMPLE_EVERY_HOURS,
            )
        )

    pd.DataFrame([asdict(m) for m in test_metas]).to_csv(
        os.path.join(out_dir, "test_metadata.csv"), index=False
    )

    # ---- Diagnostics (optional) ----
    plt.figure(figsize=(5, 3))
    plt.plot([m.omega_target for m in train_metas],
             [m.omega_achieved for m in train_metas], "o")
    lims = [0, 1]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Target Ω (train)")
    plt.ylabel("Achieved Ω")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "targets_vs_achieved_train.png"), dpi=150)
    plt.close()

    print(f"[OK] Wrote:\n  {train_val_path}\n  {boundaries_path}")
    print(f"  {len(test_metas)} held-out test CSVs named by Ω in '{out_dir}'")
    print("  train_metadata.csv and test_metadata.csv for traceability")
    print(f"  N per region: {N} samples (YEARS={YEARS}, SAMPLE_EVERY_HOURS={SAMPLE_EVERY_HOURS})")


if __name__ == "__main__":
    main()
