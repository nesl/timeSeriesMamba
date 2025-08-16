import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler  # only used in legacy path below

# ----------------------------- Forecastability metrics -----------------------------

def _detrend_linear(y, t_idx=None):
    t = np.arange(len(y)) if t_idx is None else t_idx[:len(y)]
    a, b = np.polyfit(t, y, 1)
    return y - (a * t + b)

def spectral_predictability(y, hann=None, t_idx=None):
    """
    Ω(y) in [0,1]: Ω = 1 - H / H_max, H_max = ln(K) for K spectral bins.
    """
    y = np.asarray(y, float)
    if y.size < 8:
        return np.nan
    # detrend + window
    t = np.arange(len(y)) if t_idx is None else t_idx[:len(y)]
    a, b = np.polyfit(t, y, 1)
    y = y - (a * t + b)
    w = np.hanning(len(y)) if (hann is None or len(hann) != len(y)) else hann
    yw = (y - y.mean()) * w

    spec = np.fft.rfft(yw)
    psd = np.clip((spec.real**2 + spec.imag**2), 1e-20, None)
    p = psd / psd.sum()

    H = -np.sum(p * np.log(p))
    H_max = np.log(len(p))
    return float(max(0.0, min(1.0, 1.0 - H / (H_max + 1e-12))))

def _auto_tau_acf(y, max_lag=200):
    y = y - y.mean()
    acf = np.correlate(y, y, mode='full')[len(y) - 1:]
    acf /= acf[0] + 1e-12
    for lag in range(2, min(max_lag, len(y) // 3)):
        if acf[lag - 1] > acf[lag] < acf[lag + 1]:
            return lag
    return 1

def largest_lyapunov_fast(y, m=4, tau=1, max_t=20, theiler=5):
    """
    Cheap Rosenstein LLE for screening. ~10–50x faster than full.
    """
    y = np.asarray(y, float)
    if y.size < 50:
        return np.nan
    y = (y - y.mean()) / (y.std() + 1e-12)

    N = len(y) - (m - 1) * tau
    if N < max(40, 2 * m * tau):
        return np.nan
    X = np.stack([y[i:i + N] for i in range(0, m * tau, tau)], axis=1)

    d0, nn_idx = [], []
    for i in range(N):
        lo = max(0, i - theiler)
        hi = min(N, i + theiler + 1)
        mask = np.ones(N, bool)
        mask[lo:hi] = False
        if not mask.any():
            continue
        D = np.linalg.norm(X[mask] - X[i], axis=1)
        j_rel = np.argmin(D)
        j = np.arange(N)[mask][j_rel]
        if D[j_rel] <= 0:
            continue
        d0.append(D[j_rel])
        nn_idx.append(j)
    if len(d0) < 8:
        return np.nan

    max_t = min(max_t, N - 1 - max(nn_idx))
    if max_t < 5:
        return np.nan

    lns = []
    for t in range(1, max_t + 1):
        vals = []
        for i, j in enumerate(nn_idx):
            ii = i
            jj = j
            if ii + t >= N or jj + t >= N:
                continue
            di = np.linalg.norm(X[ii + t] - X[jj + t])
            vals.append(np.log(di + 1e-12) - np.log(d0[i]))
        lns.append(np.mean(vals) if vals else np.nan)

    ts = np.arange(1, max_t + 1)
    ys = np.array(lns)
    msk = ~np.isnan(ys)
    if msk.sum() < 5:
        return np.nan
    A = np.vstack([ts[msk], np.ones(msk.sum())]).T
    slope, _ = np.linalg.lstsq(A, ys[msk], rcond=None)[0]
    return float(slope)

def largest_lyapunov_rosenstein(y, m=6, tau=None, max_t=50, theiler=10):
    """
    Rosenstein LLE estimator (publication-grade). Larger => more chaotic/harder.
    """
    y = np.asarray(y, float)
    if y.size < 50:
        return np.nan
    y = (y - y.mean()) / (y.std() + 1e-12)
    if tau is None:
        tau = _auto_tau_acf(y)

    N = len(y) - (m - 1) * tau
    if N < max(50, 2 * m * tau):
        return np.nan
    X = np.stack([y[i:i + N] for i in range(0, m * tau, tau)], axis=1)

    d0, nn_idx = [], []
    for i in range(N):
        lo = max(0, i - theiler)
        hi = min(N, i + theiler + 1)
        mask = np.ones(N, bool)
        mask[lo:hi] = False
        if not mask.any():
            continue
        D = np.linalg.norm(X[mask] - X[i], axis=1)
        j_rel = np.argmin(D)
        j = np.arange(N)[mask][j_rel]
        if D[j_rel] <= 0:
            continue
        d0.append(D[j_rel])
        nn_idx.append(j)
    if len(d0) < 10:
        return np.nan

    max_t = min(max_t, N - 1 - max(nn_idx))
    if max_t < 5:
        return np.nan

    lns = []
    for t in range(1, max_t + 1):
        vals = []
        for i, j in enumerate(nn_idx):
            ii = i
            jj = j
            if ii + t >= N or jj + t >= N:
                continue
            di = np.linalg.norm(X[ii + t] - X[jj + t])
            vals.append(np.log(di + 1e-12) - np.log(d0[i]))
        lns.append(np.mean(vals) if vals else np.nan)

    ts = np.arange(1, max_t + 1)
    ys = np.array(lns)
    msk = ~np.isnan(ys)
    if msk.sum() < 5:
        return np.nan
    A = np.vstack([ts[msk], np.ones(msk.sum())]).T
    slope, _ = np.linalg.lstsq(A, ys[msk], rcond=None)[0]
    return float(slope)

# ----------------------------- Speed helpers -----------------------------

def metric_view(y, max_len=4096, stride=6, mode="center"):
    """
    Downsample + crop to a manageable window for Ω/LLE.
    """
    yv = y[::stride] if stride > 1 else y
    if yv.size <= max_len:
        return yv
    if mode == "head":
        return yv[:max_len]
    if mode == "tail":
        return yv[-max_len:]
    s = (yv.size - max_len) // 2
    return yv[s:s + max_len]

def normalize_with_train(y, split):
    mu = float(np.mean(y[:split]))
    sd = float(np.std(y[:split]) + 1e-12)
    return (y - mu) / sd, mu, sd

# ----------------------------- Config sampler -----------------------------

def make_cfg_sampler(fixed_freqs):
    rng = np.random.default_rng()

    def _sampler(omega_hint=None):
        if omega_hint is None:
            omega_hint = 0.5

        # amplitudes template
        def amps_focused(k):
            a = rng.gamma(1.2, 0.8, size=k)
            # keep 1–2 peaks dominant
            kill = rng.choice(k, size=max(0, k-2), replace=False)
            a[kill] *= rng.uniform(0.05, 0.3, size=kill.size)
            return a

        # trend: 60% linear, 40% quadratic
        if rng.random() < 0.6:
            trend_type = 'linear'
            trend_params = float(rng.uniform(-1.5, 1.5))
        else:
            trend_type = 'polynomial'
            trend_params = [float(rng.uniform(-0.5, 0.5)),
                            float(rng.uniform(-0.6, 0.6)),
                            float(rng.uniform(-0.4, 0.4))]

        if omega_hint <= 0.2:
            # very low forecastability: almost noise
            snr = float(10 ** rng.uniform(-2.0, -0.3))      # ~0.01–0.5
            season_w = float(rng.uniform(0.02, 0.25))
            noise_exp = 0.0                                  # white
            amps = rng.uniform(0.02, 0.2, size=len(fixed_freqs))
        elif omega_hint >= 0.7:
            # very high forecastability: clear seasonality
            snr = float(10 ** rng.uniform(0.7, 1.7))        # ~5–50
            season_w = float(rng.uniform(1.0, 1.6))
            noise_exp = float(rng.choice([1.0, 2.0]))       # pink/brown
            amps = amps_focused(len(fixed_freqs))
        else:
            # mid forecastability
            snr = float(10 ** rng.uniform(-0.3, 1.0))       # ~0.5–10
            season_w = float(rng.uniform(0.3, 1.0))
            noise_exp = float(rng.choice([0.0, 1.0, 2.0]))
            amps = rng.gamma(1.2, 0.8, size=len(fixed_freqs))

        return {
            'freq': fixed_freqs,
            'amplitudes': amps.tolist(),
            'trend_type': trend_type,
            'trend_params': trend_params,
            'season_w': season_w,
            'noise_exp': noise_exp,
            'snr': snr,
        }
    return _sampler


# ----------------------------- Generator -----------------------------

class SyntheticTSGenerator:
    def __init__(self, duration_years=5, start_date=None, sample_freq='H'):
        self.duration_years = duration_years
        self.start_date = start_date or datetime.now()
        self.sample_freq = sample_freq
        self.num_points = int(duration_years * 365.25 * 24)
        self.t = np.linspace(0, duration_years, self.num_points)
        self.t_norm = self.t / duration_years
        self.timestamps = [self.start_date + timedelta(hours=i) for i in range(self.num_points)]
        self._t_idx = np.arange(self.num_points)
        self._hann = np.hanning(self.num_points)

    def colored_noise(self, exponent, std, seed=None):
        if seed is not None:
            np.random.seed(seed)
        freqs = np.fft.rfftfreq(self.num_points, d=1)
        freqs[0] = freqs[1]
        spectrum = np.power(freqs, -exponent / 2.0)
        phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))
        fft_vals = spectrum * phases
        y = np.fft.irfft(fft_vals, n=self.num_points)
        y = y / np.std(y) * std
        return y

    def calculate_spectral_entropy(self, signal):
        # legacy metric used in generate_dataset(); ok to keep SciPy if installed
        from scipy.fft import rfft
        psd = np.abs(rfft(signal)) ** 2
        psd = psd / np.sum(psd)
        entropy = -np.sum(psd * np.log2(psd + 1e-10))
        return entropy

    def generate_signal(self, freqs, amplitudes, phases=None, seed=None):
        if len(freqs) != len(amplitudes):
            raise ValueError("Number of frequencies and amplitudes must match.")
        if seed is not None:
            np.random.seed(seed)
        phases = phases or np.random.uniform(0, 2 * np.pi, len(freqs))
        y = np.zeros(self.num_points)
        for f, a, p in zip(freqs, amplitudes, phases):
            y += a * np.sin(2 * np.pi * f * self.t + p)
        return y

    def generate_trend(self, trend_type, trend_params):
        if trend_type == 'linear':
            if not isinstance(trend_params, (int, float)):
                raise ValueError("For 'linear' trend, trend_params should be a scalar slope.")
            slope = trend_params
            trend = slope * (self.t_norm - 0.5)
        elif trend_type == 'polynomial':
            if not isinstance(trend_params, list):
                raise ValueError("For 'polynomial' trend, trend_params should be a list of coefficients.")
            trend = np.zeros_like(self.t)
            for k, a_k in enumerate(trend_params):
                trend += a_k * np.power(self.t_norm, k)
        else:
            raise ValueError(f"Unknown trend_type: {trend_type}")
        return trend

    def synthesize(self, freqs, amplitudes, trend_type, trend_params, season_weight, noise_exponent, snr_target, seed=None):
        base_seasonal = self.generate_signal(freqs, amplitudes, seed=seed)
        trend = self.generate_trend(trend_type, trend_params)
        signal = season_weight * base_seasonal + trend
        noise_std = np.sqrt(np.var(signal) / snr_target)
        noise = self.colored_noise(noise_exponent, noise_std, seed=seed)
        ideal_mse = np.mean(noise ** 2)
        return signal + noise, ideal_mse

    # ----------------------- Ω-targeted acceptance sampler (fast) -----------------------

    def target_series_by_omega(self,
                               target_omega,
                               tol=0.03,
                               lambda_max=0.5,
                               max_tries=200,
                               seed_base=1000,
                               cfg_sampler=None,
                               train_frac=0.8,
                               show_progress=False,
                               proxy_len=4096,
                               proxy_stride=6):
        """
        Accept if |Ω-Ω*|<=tol on a downsampled/cropped proxy AND fast LLE <= lambda_max+margin.
        Compute full LLE once on accepted candidate (on a denser proxy).
        """
        assert cfg_sampler is not None, "Provide cfg_sampler() to explore parameter space."
        split = int(train_frac * self.num_points)
        pbar = tqdm(total=max_tries, desc=f"Ω={target_omega:.3f}", leave=False) if show_progress else None

        for k in range(max_tries):
            cfg = cfg_sampler(target_omega) if cfg_sampler.__code__.co_argcount >= 1 else cfg_sampler()
            cfg_seed = seed_base + k
            y, ideal_mse = self.synthesize(
                cfg['freq'], cfg['amplitudes'], cfg['trend_type'], cfg['trend_params'],
                cfg['season_w'], cfg['noise_exp'], cfg['snr'], seed=cfg_seed
            )

            # normalize using TRAIN portion (protocol-aligned)
            y_norm, mu, sd = normalize_with_train(y, split)

            # proxy for metrics
            yv = metric_view(y_norm, max_len=proxy_len, stride=proxy_stride, mode="center")

            # cheap Ω + cheap LLE
            om = spectral_predictability(yv)
            lle_fast = largest_lyapunov_fast(yv)

            if (np.isfinite(om) and abs(om - target_omega) <= tol and
                np.isfinite(lle_fast) and (lle_fast <= (lambda_max + 0.1))):

                # one-time full LLE on slightly denser proxy
                yv_full = metric_view(y_norm, max_len=proxy_len, stride=max(1, proxy_stride // 2), mode="center")
                lle = largest_lyapunov_rosenstein(yv_full, m=6, tau=None, max_t=50, theiler=10)

                if np.isfinite(lle) and (lle <= lambda_max):
                    ideal_mse_norm = ideal_mse / (sd ** 2)
                    if pbar:
                        pbar.close()
                    return {
                        'y': y,
                        'omega': float(om),
                        'lle': float(lle),
                        'ideal_mse': float(ideal_mse),
                        'ideal_mse_norm': float(ideal_mse_norm),
                        'scaler_mean': mu,
                        'scaler_scale': sd,
                        'cfg': cfg,
                        'seed': int(cfg_seed),
                        'target_omega': float(target_omega),
                        'lambda_max': float(lambda_max),
                        'tol': float(tol),
                        'proxy_len': int(proxy_len),
                        'proxy_stride': int(proxy_stride),
                    }
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()
        return None

    # ----------------------- Dataset builder (Ω sweep) -----------------------

    def generate_dataset_from_omega(self,
                                    omega_targets_train,
                                    omega_targets_test=None,
                                    tol=0.03,
                                    lambda_max=0.5,
                                    train_frac=0.8,
                                    cfg_sampler=None,
                                    out_dir='synthetic_data',
                                    save_plots=True,
                                    proxy_len=4096,
                                    proxy_stride=6):
        """
        Build train_val.csv + boundaries + region_metrics.csv via Ω-targeted acceptance sampling.
        """
        assert cfg_sampler is not None, "Provide cfg_sampler() to explore parameter space."
        os.makedirs(out_dir, exist_ok=True)
        split = int(train_frac * self.num_points)
        val_size = self.num_points - split

        train_list, val_list = [], []
        boundaries, regions_list = [], []
        metrics_rows = []
        config_rows = []

        # ----- sample training regions -----
        hits_train = []
        for idx, om_t in enumerate(tqdm(omega_targets_train, desc="Train Ω targets")):
            hit = self.target_series_by_omega(
                om_t, tol=tol, lambda_max=lambda_max, max_tries=200,
                seed_base=1000 + idx * 10000, cfg_sampler=cfg_sampler,
                train_frac=train_frac, show_progress=True,
                proxy_len=proxy_len, proxy_stride=proxy_stride
            )
            if hit is None:
                # relax once
                hit = self.target_series_by_omega(
                    om_t, tol=min(0.04, 2 * tol), lambda_max=lambda_max + 0.1, max_tries=200,
                    seed_base=2000 + idx * 10000, cfg_sampler=cfg_sampler,
                    train_frac=train_frac, show_progress=True,
                    proxy_len=proxy_len, proxy_stride=proxy_stride
                )
            if hit is None:
                print(f"[WARN] Could not achieve target Ω={om_t:.3f} within tolerances.")
                continue
            hits_train.append(hit)

        # assign region ids (1..K for train)
        for rid, hit in enumerate(hits_train, start=1):
            y = hit['y']
            y_train = y[:split]
            y_val = y[split:]
            df_train = pd.DataFrame({'date': self.timestamps[:split], 'synth': y_train})
            df_val = pd.DataFrame({'date': self.timestamps[split:], 'synth': y_val})
            train_list.append(df_train)
            val_list.append(df_val)

            # boundaries for concat
            start_t = (rid - 1) * split
            end_t = start_t + split - 1
            boundaries.append([start_t, end_t])
            regions_list.append(f"Region {rid}")

            # val boundaries
            val_offset = len(hits_train) * split
            start_v = val_offset + (rid - 1) * val_size
            end_v = start_v + val_size - 1
            boundaries.append([start_v, end_v])
            regions_list.append(f"Region {rid}")

            # metrics
            metrics_rows.append({
                'region': rid,
                'part': 'train_val',
                'target_omega': hit['target_omega'],
                'spectral_predictability': hit['omega'],
                'lyapunov_exponent': hit['lle'],
                'ideal_mse_norm': hit['ideal_mse_norm'],
                'scaler_mean': hit['scaler_mean'],
                'scaler_scale': hit['scaler_scale'],
                'lambda_max': hit['lambda_max'],
                'tol': hit['tol'],
                'proxy_len': hit['proxy_len'],
                'proxy_stride': hit['proxy_stride'],
            })
            cfg = hit['cfg'].copy()
            cfg['region_id'] = rid
            cfg['seed'] = hit['seed']
            config_rows.append(cfg)

            if save_plots:
                self.save_plot(df_train, rid, 'train', cfg, out_dir=os.path.join(out_dir, 'region_plots'))
                self.save_plot(df_val, rid, 'val', cfg, out_dir=os.path.join(out_dir, 'region_plots'))

        # ----- concat + write -----
        if len(train_list) == 0:
            raise RuntimeError("No training regions were generated. Loosen tol/lambda_max or adjust cfg_sampler.")
        train_val_df = pd.concat(train_list + val_list, ignore_index=True)
        train_val_df.to_csv(os.path.join(out_dir, 'train_val.csv'), index=False)
        with open(os.path.join(out_dir, 'train_boundaries.json'), 'w') as f:
            json.dump({'boundaries': boundaries, 'regions': regions_list}, f, indent=2)

        # ----- test regions (ID or mild OOD) -----
        if omega_targets_test is None:
            tr = np.array(sorted(omega_targets_train))
            omega_targets_test = (tr[:-1] + tr[1:]) / 2.0

        test_written = []
        for idx, om_t in enumerate(tqdm(omega_targets_test, desc="Test Ω targets"), start=1):
            hit = self.target_series_by_omega(
                om_t, tol=tol, lambda_max=lambda_max, max_tries=200,
                seed_base=900000 + idx * 5000, cfg_sampler=cfg_sampler,
                train_frac=train_frac, show_progress=True,
                proxy_len=proxy_len, proxy_stride=proxy_stride
            )
            if hit is None:
                hit = self.target_series_by_omega(
                    om_t, tol=min(0.04, 2 * tol), lambda_max=lambda_max + 0.1, max_tries=200,
                    seed_base=910000 + idx * 5000, cfg_sampler=cfg_sampler,
                    train_frac=train_frac, show_progress=True,
                    proxy_len=proxy_len, proxy_stride=proxy_stride
                )
            if hit is None:
                print(f"[WARN] Could not achieve TEST target Ω={om_t:.3f} within tolerances.")
                continue

            y = hit['y']
            df_test = pd.DataFrame({'date': self.timestamps, 'synth': y})
            region_name = f"region_test_om{om_t:.3f}".replace('.', 'p')
            df_test.to_csv(os.path.join(out_dir, f'{region_name}.csv'), index=False)
            test_written.append(region_name)

            metrics_rows.append({
                'region': region_name,
                'part': 'test',
                'target_omega': hit['target_omega'],
                'spectral_predictability': hit['omega'],
                'lyapunov_exponent': hit['lle'],
                'ideal_mse_norm': hit['ideal_mse_norm'],
                'scaler_mean': hit['scaler_mean'],
                'scaler_scale': hit['scaler_scale'],
                'lambda_max': hit['lambda_max'],
                'tol': hit['tol'],
                'proxy_len': hit['proxy_len'],
                'proxy_stride': hit['proxy_stride'],
            })
            cfg = hit['cfg'].copy()
            cfg['region_id'] = region_name
            cfg['seed'] = hit['seed']
            config_rows.append(cfg)

            if save_plots:
                self.save_plot(df_test, region_name, 'test', cfg, out_dir=os.path.join(out_dir, 'region_plots'))

        # ----- write metrics + configs -----
        pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir, 'region_metrics.csv'), index=False)
        with open(os.path.join(out_dir, 'region_config_details.json'), 'w') as f:
            json.dump(config_rows, f, indent=2)

        print(f"Generated train_val.csv, train_boundaries.json, {len(test_written)} test CSV(s), metrics/configs in '{out_dir}'.")

    # ----------------------- Legacy region-wise generator (kept) -----------------------

    def generate_dataset(self, regions, train_regions, test_region, train_frac=0.8, out_dir='synthetic_data'):
        os.makedirs(out_dir, exist_ok=True)
        split = int(train_frac * self.num_points)
        val_size = self.num_points - split
        train_list, val_list = [], []
        boundaries, regions_list = [], []
        metrics_list = []

        def compute_metrics(y, ideal_mse, cfg, part):
            # legacy: StandardScaler + entropy/variance
            y_train = y[:split].reshape(-1, 1)
            scaler = StandardScaler().fit(y_train)
            y_norm = scaler.transform(y.reshape(-1, 1)).flatten()
            ideal_mse_norm = ideal_mse / (scaler.scale_[0] ** 2)
            ent = self.calculate_spectral_entropy(y_norm)
            var_y = np.var(y_norm)
            metrics_list.append({
                'region': cfg['region_id'],
                'part': part,
                'snr': cfg['snr'],
                'ideal_mse_norm': ideal_mse_norm,
                'spectral_entropy_norm': ent,
                'variance_norm': var_y
            })

        # train regions
        for idx, rid in enumerate(train_regions):
            cfg = regions[rid]
            cfg['region_id'] = rid
            y, ideal_mse = self.synthesize(
                cfg['freq'], cfg['amplitudes'], cfg['trend_type'], cfg['trend_params'],
                cfg['season_w'], cfg['noise_exp'], cfg['snr'], seed=rid
            )
            compute_metrics(y, ideal_mse, cfg, 'train')

            df = pd.DataFrame({'date': self.timestamps, 'synth': y})
            train_df = df.iloc[:split]
            val_df = df.iloc[split:]
            train_list.append(train_df)
            val_list.append(val_df)

            start_t = idx * split
            end_t = start_t + split - 1
            boundaries.append([start_t, end_t])
            regions_list.append(f"Region {rid}")
            self.save_plot(train_df, rid, 'train', cfg)

        # validation plots
        val_offset = len(train_regions) * split
        for idx, rid in enumerate(train_regions):
            start_v = val_offset + idx * val_size
            end_v = start_v + val_size - 1
            boundaries.append([start_v, end_v])
            regions_list.append(f"Region {rid}")
            self.save_plot(val_list[idx], rid, 'val', regions[rid])

        # write train+val
        train_val_df = pd.concat(train_list + val_list, ignore_index=True)
        train_val_df.to_csv(os.path.join(out_dir, 'train_val.csv'), index=False)
        with open(os.path.join(out_dir, 'train_boundaries.json'), 'w') as f:
            json.dump({'boundaries': boundaries, 'regions': regions_list}, f, indent=2)

        # test region
        if test_region in regions:
            cfg = regions[test_region]
            cfg['region_id'] = test_region
            y_test, ideal_mse_test = self.synthesize(
                cfg['freq'], cfg['amplitudes'], cfg['trend_type'], cfg['trend_params'],
                cfg['season_w'], cfg['noise_exp'], cfg['snr'], seed=test_region
            )
            compute_metrics(y_test, ideal_mse_test, cfg, 'test')

            test_df = pd.DataFrame({'date': self.timestamps, 'synth': y_test})
            test_df.to_csv(os.path.join(out_dir, f'region{test_region}.csv'), index=False)
            self.save_plot(test_df, test_region, 'test', cfg)

        # save metrics/config
        pd.DataFrame(metrics_list).to_csv(os.path.join(out_dir, 'region_metrics.csv'), index=False)
        with open(os.path.join(out_dir, 'region_config_details.json'), 'w') as f:
            json.dump(regions, f, indent=2)

        print(f"Generated train_val.csv, region{rid}.csv, metrics in '{out_dir}' and plots in 'region_plots'")

# --------------------------------- Main ---------------------------------

if __name__ == "__main__":
    duration_years = 5
    start_date = datetime(2024, 1, 1)

    gen = SyntheticTSGenerator(duration_years, start_date)

    # canonical frequency set (cycles/year)
    fixed_freqs = [2 * 365.25, 365.25, 52.18, 12, 1]
    sampler = make_cfg_sampler(fixed_freqs)

    # Numerical Ω sweep (train); OOD test defaults to midpoints
    omega_targets_train = np.linspace(0.15, 0.85, 7)

    gen.generate_dataset_from_omega(
        omega_targets_train=omega_targets_train,
        omega_targets_test=None,      # midpoints (mild OOD)
        tol=0.1,                     # a bit looser => faster
        lambda_max=0.75,               # guardrail for chaos
        train_frac=0.8,
        cfg_sampler=sampler,
        out_dir='synthetic_data_omega',
        save_plots=True,
        proxy_len=4096,
        proxy_stride=6
    )

    # --- legacy region-config path (kept for reference) ---
    """
    train_regions = [1, 2, 3, 4, 5, 6]
    test_region = 14
    regions = {...}
    gen.generate_dataset(regions, train_regions, test_region, train_frac=0.8, out_dir='.')
    """
