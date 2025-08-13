import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch
from statsmodels.tsa.seasonal import STL

class SyntheticTSGenerator:
    def __init__(self, duration_years=5, start_date=None, sample_freq='H'):
        self.duration_years = duration_years
        self.start_date = start_date or datetime.now()
        self.sample_freq = sample_freq
        self.num_points = int(duration_years * 365.25 * 24)
        self.t = np.linspace(0, duration_years, self.num_points)
        self.t_norm = self.t / duration_years
        self.timestamps = [self.start_date + timedelta(hours=i) for i in range(self.num_points)]

    def colored_noise(self, exponent, std, seed=None):
        if seed is not None:
            np.random.seed(seed)
        freqs = np.fft.rfftfreq(self.num_points, d=1)
        freqs[0] = freqs[1]
        spectrum = np.power(freqs, -exponent / 2.0)
        phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))
        fft_vals = spectrum * phases
        y = np.fft.irfft(fft_vals, n=self.num_points)
        y = y / (np.std(y) if np.std(y)>0 else 1.0) * std
        return y

    def generate_signal(self, freqs, amplitudes, phases=None, seed=None):
        if len(freqs) != len(amplitudes):
            raise ValueError("freqs and amplitudes must match")
        if seed is not None:
            np.random.seed(seed)
        phases = phases or np.random.uniform(0, 2 * np.pi, len(freqs))
        y = np.zeros(self.num_points)
        for f, a, p in zip(freqs, amplitudes, phases):
            y += a * np.sin(2 * np.pi * f * self.t + p)
        return y

    def generate_trend(self, trend_type, trend_params):
        if trend_type == 'linear':
            slope = trend_params
            trend = slope * (self.t_norm - 0.5)
        elif trend_type == 'polynomial':
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
        noise_std = np.sqrt(np.var(signal) / snr_target) if snr_target>0 else np.std(signal)
        noise = self.colored_noise(noise_exponent, noise_std, seed=seed)
        ideal_mse = float(np.mean(noise**2))
        return signal + noise, ideal_mse

    def save_plot(self, df, region_id, part, cfg, out_dir='region_plots'):
        os.makedirs(out_dir, exist_ok=True)
        snippet = df.iloc[:608]
        plt.figure(figsize=(12, 4))
        plt.plot(snippet['date'], snippet['synth'], linewidth=1)
        title = f"Region {region_id} - {part} | snr={cfg['snr']} | season_w={cfg['season_w']} | noise_exp={cfg['noise_exp']} | trend={cfg['trend_type']}"
        plt.title(title)
        plt.xlabel('Date'); plt.ylabel('Value')
        plt.tight_layout()
        fname = f"Region_{region_id}_{part}_snr{cfg['snr']}_sw{cfg['season_w']}_exp{cfg['noise_exp']}_{cfg['trend_type']}.png".replace('.', 'p')
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    def generate_dataset(self, regions, train_regions, test_region, train_frac=0.8, out_dir='synthetic_data'):
        os.makedirs(out_dir, exist_ok=True)
        split = int(train_frac * self.num_points)
        val_size = self.num_points - split
        train_list, val_list = [], []
        boundaries, regions_list = [], []
        metrics_list = []

        def spectral_entropy_welch(x, fs=1.0, nperseg=None):
            n = len(x); nper = min(n, 1024) if nperseg is None else nperseg
            f, Pxx = welch(x, fs=fs, nperseg=nper)
            P = Pxx / np.sum(Pxx)
            H = -np.sum(P * np.log(P + 1e-12))
            return float(H / np.log(len(P)))

        def dominant_period(x, fs=1.0):
            n = len(x)
            f, Pxx = welch(x, fs=fs, nperseg=min(n, 4096))
            if len(f) < 3: return 24
            idx = np.argmax(Pxx[1:]) + 1
            dom_f = f[idx]
            return int(max(2, round(fs / dom_f))) if dom_f > 0 else 24

        def seasonal_strength_stl(x, period):
            if np.allclose(x, x[0]): return 0.0
            res = STL(x, period=period, robust=True).fit()
            num = np.var(res.resid, ddof=1)
            den = np.var(res.seasonal + res.resid, ddof=1)
            if den <= 0: return 0.0
            v = 1.0 - num/den
            return float(max(0.0, v))

        def stl_snr_estimate(x, period):
            res = STL(x, period=period, robust=True).fit()
            sig = np.var(res.trend + res.seasonal, ddof=1)
            noi = np.var(res.resid, ddof=1)
            return float(sig/noi) if noi > 0 else np.inf

        def compute_metrics(y, ideal_mse, cfg, part):
            y_train = y[:split].reshape(-1, 1)
            scaler = StandardScaler().fit(y_train)
            y_norm = scaler.transform(y.reshape(-1, 1)).flatten()
            ideal_mse_norm = ideal_mse / (scaler.scale_[0] ** 2)
            per = dominant_period(y_norm[:split], fs=1.0)
            sent = spectral_entropy_welch(y_norm[:split], fs=1.0)
            sstr = seasonal_strength_stl(y_norm[:split], period=per)
            snr_est = stl_snr_estimate(y_norm[:split], period=per)
            metrics_list.append({
                'region': cfg['region_id'],
                'part': part,
                'snr_design': cfg['snr'],
                'ideal_mse_norm': ideal_mse_norm,
                'spectral_entropy': sent,
                'seasonal_strength': sstr,
                'snr_est': snr_est,
                'dom_period': per
            })

        for idx, rid in enumerate(train_regions):
            cfg = dict(regions[rid]); cfg['region_id'] = rid
            y, ideal_mse = self.synthesize(cfg['freq'], cfg['amplitudes'], cfg['trend_type'], cfg['trend_params'],
                                           cfg['season_w'], cfg['noise_exp'], cfg['snr'], seed=rid)
            compute_metrics(y, ideal_mse, cfg, 'train')
            df = pd.DataFrame({'date': self.timestamps, 'synth': y})
            train_df = df.iloc[:split]; val_df = df.iloc[split:]
            train_list.append(train_df); val_list.append(val_df)
            start_t = idx * split; end_t = start_t + split - 1
            boundaries.append([start_t, end_t]); regions_list.append(f"Region {rid}")
            self.save_plot(train_df, rid, 'train', cfg)

        val_offset = len(train_regions) * split
        for idx, rid in enumerate(train_regions):
            start_v = val_offset + idx * val_size; end_v = start_v + val_size - 1
            boundaries.append([start_v, end_v]); regions_list.append(f"Region {rid}")
            self.save_plot(val_list[idx], rid, 'val', regions[rid])

        train_val_df = pd.concat(train_list + val_list, ignore_index=True)
        os.makedirs(out_dir, exist_ok=True)
        train_val_df.to_csv(os.path.join(out_dir, 'train_val.csv'), index=False)
        with open(os.path.join(out_dir, 'train_boundaries.json'), 'w') as f:
            json.dump({'boundaries': boundaries, 'regions': regions_list}, f, indent=2)

        if test_region in regions:
            cfg = dict(regions[test_region]); cfg['region_id'] = test_region
            y_test, ideal_mse_test = self.synthesize(cfg['freq'], cfg['amplitudes'], cfg['trend_type'], cfg['trend_params'],
                                                     cfg['season_w'], cfg['noise_exp'], cfg['snr'], seed=test_region)
            compute_metrics(y_test, ideal_mse_test, cfg, 'test')
            test_df = pd.DataFrame({'date': self.timestamps, 'synth': y_test})
            test_df.to_csv(os.path.join(out_dir, f'region{test_region}.csv'), index=False)
            self.save_plot(test_df, test_region, 'test', cfg)

        pd.DataFrame(metrics_list).to_csv(os.path.join(out_dir, 'region_metrics.csv'), index=False)
        with open(os.path.join(out_dir, 'region_config_details.json'), 'w') as f:
            json.dump({k:regions[k] for k in train_regions+[test_region] if k in regions}, f, indent=2)
        print(f"Generated train_val.csv, region{test_region}.csv, metrics in '{out_dir}' and plots in 'region_plots'")

def spectral_entropy_welch(x, fs=1.0, nperseg=None):
    n = len(x); nper = min(n, 1024) if nperseg is None else nperseg
    f, Pxx = welch(x, fs=fs, nperseg=nper)
    P = Pxx / np.sum(Pxx)
    H = -np.sum(P * np.log(P + 1e-12))
    return float(H / np.log(len(P)))

def seasonal_strength_stl(x, period):
    if np.allclose(x, x[0]): return 0.0
    res = STL(x, period=period, robust=True).fit()
    num = np.var(res.resid, ddof=1)
    den = np.var(res.seasonal + res.resid, ddof=1)
    if den <= 0: return 0.0
    return float(max(0.0, 1.0 - num/den))

def stl_snr_estimate(x, period):
    res = STL(x, period=period, robust=True).fit()
    sig = np.var(res.trend + res.seasonal, ddof=1)
    noi = np.var(res.resid, ddof=1)
    return float(sig/noi) if noi > 0 else np.inf

def dominant_period(x, fs=1.0):
    n = len(x)
    f, Pxx = welch(x, fs=fs, nperseg=min(n, 4096))
    if len(f) < 3: return 24
    idx = np.argmax(Pxx[1:]) + 1
    dom_f = f[idx]
    return int(max(2, round(fs / dom_f))) if dom_f > 0 else 24

def probe_metrics(gen, regions_cfg, train_frac=0.8, fs=1.0):
    split = int(train_frac * gen.num_points)
    rows = []
    for rid, cfg in regions_cfg.items():
        y, _ = gen.synthesize(cfg['freq'], cfg['amplitudes'], cfg['trend_type'], cfg['trend_params'],
                              cfg['season_w'], cfg['noise_exp'], cfg['snr'], seed=rid)
        y_tr = y[:split]
        mu, sigma = float(np.mean(y_tr)), float(np.std(y_tr)) if np.std(y_tr)>0 else 1.0
        y_norm = (y - mu)/sigma
        per = dominant_period(y_norm[:split], fs=fs)
        sent = spectral_entropy_welch(y_norm[:split], fs=fs)
        sstr = seasonal_strength_stl(y_norm[:split], period=per)
        snr_est = stl_snr_estimate(y_norm[:split], period=per)
        rows.append({'region': rid, 'spectral_entropy': sent, 'seasonal_strength': sstr, 'snr_est': snr_est, 'dom_period': per})
    return pd.DataFrame(rows)

def select_balanced(df, k=6, seed=0):
    df = df.copy()
    s1 = df['spectral_entropy'].median()
    s2 = df['seasonal_strength'].median()
    df['bin'] = (df['spectral_entropy']>s1).astype(int)*2 + (df['seasonal_strength']>s2).astype(int)
    target = {0:k//4, 1:k//4, 2:k//4, 3:k - 3*(k//4)}
    rng = np.random.default_rng(seed)
    picks = []
    for b, m in target.items():
        cand = df[df['bin']==b].reset_index(drop=True)
        if len(cand) <= m:
            picks.extend(cand['region'].tolist())
        else:
            C = cand[['spectral_entropy','seasonal_strength']].to_numpy()
            sel = [int(rng.integers(len(cand)))]
            while len(sel) < m:
                d = ((C - C[sel][:,None,:])**2).sum(-1).min(0)
                d[sel] = -1
                sel.append(int(np.argmax(d)))
            picks.extend(cand.loc[sel,'region'].tolist())
    return sorted(picks)[:k]

def audit_plot(df, sel, out='audit_selection.png'):
    plt.figure(figsize=(5.5,4.5))
    plt.scatter(df['spectral_entropy'], df['seasonal_strength'], s=14, alpha=0.5)
    m = df['region'].isin(sel)
    plt.scatter(df.loc[m,'spectral_entropy'], df.loc[m,'seasonal_strength'], s=48, marker='x')
    plt.xlabel('Spectral Entropy'); plt.ylabel('Seasonal Strength')
    plt.tight_layout(); plt.savefig(out); plt.close()

if __name__ == "__main__":
    duration_years = 5
    start_date = datetime(2024, 1, 1)
    train_frac = 0.8
    test_region = 14
    fixed_freqs = [2 * 365.25, 365.25, 52.18, 12, 1]
    regions = {
        1: {'freq': fixed_freqs,'amplitudes': [0.5, 2.0, 1.0, 0.3, 0.2],'trend_type': 'linear','trend_params': 0.5,'season_w': 1.0,'noise_exp': 1,'snr': 5},
        2: {'freq': fixed_freqs,'amplitudes': [1.2, 1.5, 0.5, 0.4, 0.1],'trend_type': 'polynomial','trend_params': [0, 0.3, 0.1],'season_w': 0.8,'noise_exp': 1,'snr': 3},
        3: {'freq': fixed_freqs,'amplitudes': [0.2, 0.2, 1.0, 1.0, 0.5],'trend_type': 'polynomial','trend_params': [0, 0.5, 0.1],'season_w': 0.8,'noise_exp': 1,'snr': 3},
        4: {'freq': fixed_freqs,'amplitudes': [0.1, 0.5, 1.0, 1.3, 0.5],'trend_type': 'linear','trend_params': -1.5,'season_w': 1.0,'noise_exp': 1,'snr': 5},
        5: {'freq': fixed_freqs,'amplitudes': [0.3, 1.2, 1.5, 0.4, 0.1],'trend_type': 'polynomial','trend_params': [0, -0.3, 2],'season_w': 0.8,'noise_exp': 1,'snr': 3},
        6: {'freq': fixed_freqs,'amplitudes': [0.2, 0.5, 0.4, 1.2, 1.1],'trend_type': 'polynomial','trend_params': [0, 0.3, -1],'season_w': 0.8,'noise_exp': 1,'snr': 3},
        7: {'freq': fixed_freqs,'amplitudes': [0.4, 0.4, 0.8, 1.0, 0.6],'trend_type': 'polynomial','trend_params': [-0.3, 0.4, -0.2],'season_w': 0.5,'noise_exp': 1,'snr': 3},
        8: {'freq': fixed_freqs,'amplitudes': [0.5, 0.5, 0.5, 0.5, 0.5],'trend_type': 'polynomial','trend_params': [-0.3, 0.4, -0.2],'season_w': 0.5,'noise_exp': 1,'snr': 3},
        9: {'freq': fixed_freqs,'amplitudes': [0.2, 0.5, 0.5, 1.5, 0.75],'trend_type': 'polynomial','trend_params': [-0.3, 4, -0.2],'season_w': 0.5,'noise_exp': 2,'snr': 0.003},
        10:{'freq': fixed_freqs,'amplitudes': [0.2, 0.5, 0.3, 0.5, 1.2],'trend_type': 'linear','trend_params': 0,'season_w': 1,'noise_exp': 0,'snr': 1},
        11:{'freq': fixed_freqs,'amplitudes': [0.1, 0.6, 0.6, 0.8, 0.6],'trend_type': 'linear','trend_params': 2.0,'season_w': 0.6,'noise_exp': 1.8,'snr': 3},
        12:{'freq': fixed_freqs,'amplitudes': [0.1, 0.2, 0.3, 0.2, 0.2],'trend_type': 'linear','trend_params': 0.0,'season_w': 0.2,'noise_exp': 0.0,'snr': 0.5},
        13:{'freq': fixed_freqs,'amplitudes': [0.2, 4.0, 0.3, 0.1, 0.05],'trend_type': 'linear','trend_params': 0.0,'season_w': 1.2,'noise_exp': 1.0,'snr': 20},
        14:{'freq': fixed_freqs,'amplitudes': [0.1, 0.4, 0.4, 0.4, 0.2],'trend_type': 'linear','trend_params': 0.0,'season_w': 0.2,'noise_exp': 0.0,'snr': 0.1}
    }
    gen = SyntheticTSGenerator(duration_years, start_date)
    df_probe = probe_metrics(gen, regions, train_frac, fs=1.0)
    sel = select_balanced(df_probe, k=6, seed=0)
    with open('selected_train_regions.json','w') as f: json.dump({'train_regions': sel}, f, indent=2)
    audit_plot(df_probe, sel, out='audit_selection.png')
    print('Selected train regions:', sel)
    gen.generate_dataset(regions, sel, test_region, train_frac, out_dir='.')
