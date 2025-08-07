import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
        y = y / np.std(y) * std
        return y

    def calculate_spectral_entropy(self, signal):
        from scipy.fft import rfft
        psd = np.abs(rfft(signal))**2
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
        ideal_mse = np.mean(noise**2)
        return signal + noise, ideal_mse

    def save_plot(self, df, region_id, part, cfg, out_dir='region_plots'):
        os.makedirs(out_dir, exist_ok=True)
        snippet = df.iloc[:608]
        plt.figure(figsize=(12, 4))
        plt.plot(snippet['date'], snippet['synth'], linewidth=1)

        title = (
            f"Region {region_id} - {part} | snr={cfg['snr']} | season_w={cfg['season_w']} | "
            f"noise_exp={cfg['noise_exp']} | trend={cfg['trend_type']}"
        )
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.tight_layout()

        fname = (
            f"Region_{region_id}_{part}_snr{cfg['snr']}_sw{cfg['season_w']}_exp{cfg['noise_exp']}_{cfg['trend_type']}.png"
        ).replace('.', 'p')
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    def generate_dataset(self, regions, train_regions, test_region, train_frac=0.8, out_dir='synthetic_data'):
        os.makedirs(out_dir, exist_ok=True)
        split = int(train_frac * self.num_points)
        val_size = self.num_points - split
        train_list, val_list = [], []
        boundaries, regions_list = [], []
        metrics_list = []

        # helper to scale and compute metrics
        def compute_metrics(y, ideal_mse, cfg, part):
            # fit on train portion
            y_train = y[:split].reshape(-1, 1)
            scaler = StandardScaler().fit(y_train)
            y_norm = scaler.transform(y.reshape(-1, 1)).flatten()

            # normalized ideal MSE
            ideal_mse_norm = ideal_mse / (scaler.scale_[0] ** 2)
            # entropy & variance on normalized series
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

        # process train regions
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

        # save train+val
        train_val_df = pd.concat(train_list + val_list, ignore_index=True)
        train_val_df.to_csv(os.path.join(out_dir, 'train_val.csv'), index=False)
        with open(os.path.join(out_dir, 'train_boundaries.json'), 'w') as f:
            json.dump({'boundaries': boundaries, 'regions': regions_list}, f, indent=2)

        # process test region
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

        # save metrics
        pd.DataFrame(metrics_list).to_csv(os.path.join(out_dir, 'region_metrics.csv'), index=False)

        # save config
        with open(os.path.join(out_dir, 'region_config_details.json'), 'w') as f:
            json.dump(regions, f, indent=2)

        print(f"Generated train_val.csv, region{rid}.csv, metrics in '{out_dir}' and plots in 'region_plots'")

# Example usage
if __name__ == "__main__":
    duration_years = 5
    start_date = datetime(2024, 1, 1)
    train_regions = [1, 2, 3, 4, 5, 6]
    test_region = 10
    train_frac = 0.8

    # Fixed frequencies (cycles per year): half-daily, daily, weekly, monthly, yearly
    fixed_freqs = [2 * 365.25, 365.25, 52.18, 12, 1]
    
    # Regions with different amplitudes and trends
    regions = {
        1: {
            'freq': fixed_freqs,
            'amplitudes': [0.5, 2.0, 1.0, 0.3, 0.2],  # High amplitude for daily
            'trend_type': 'linear',
            'trend_params': 0.5,  # Linear slope
            'season_w': 1.0,
            'noise_exp': 1, #0 is white noise, 1 is pink noise, 2 is brown noise
            'snr': 5
        },
        2: {
            'freq': fixed_freqs,
            'amplitudes': [1.2, 1.5, 0.5, 0.4, 0.1],  # semidaily and daily
            'trend_type': 'polynomial',
            'trend_params': [0, 0.3, 0.1],  # Polynomial: a0 + a1*t + a2*t^2
            'season_w': 0.8,
            'noise_exp': 1, 
            'snr': 3 
        },
        3: {
            'freq': fixed_freqs,
            'amplitudes': [0.2, 0.2, 1.0, 1.0, 0.5],  # monthly and weekly
            'trend_type': 'polynomial',
            'trend_params': [0, 0.5, 0.1],  # Polynomial: a0 + a1*t + a2*t^2
            'season_w': 0.8,
            'noise_exp': 1,
            'snr': 3
        },
        4: {
            'freq': fixed_freqs,
            'amplitudes': [0.1, 0.5, 1.0, 1.3, 0.5],  # High amplitude for weekly and monthly
            'trend_type': 'linear',
            'trend_params': -1.5,  # Linear slope
            'season_w': 1.0,
            'noise_exp': 1, #0 is white noise, 1 is pink noise, 2 is brown noise
            'snr': 5
        },
        5: {
            'freq': fixed_freqs,
            'amplitudes': [0.3, 1.2, 1.5, 0.4, 0.1],  # daily and weekly
            'trend_type': 'polynomial',
            'trend_params': [0, -0.3, 2],  # Polynomial: a0 + a1*t + a2*t^2
            'season_w': 0.8,
            'noise_exp': 1, 
            'snr': 3 
        },
        6: {
            'freq': fixed_freqs,
            'amplitudes': [0.2, 0.5, 0.4, 1.2, 1.1],  # monthly and yearly
            'trend_type': 'polynomial',
            'trend_params': [0, 0.3, -1],  # Polynomial: a0 + a1*t + a2*t^2
            'season_w': 0.8,
            'noise_exp': 1,
            'snr': 3
        },
        7: {
            'freq': fixed_freqs,
            'amplitudes': [0.4, 0.4, 0.8, 1.0, 0.6], #increase
            'trend_type': 'polynomial',
            'trend_params': [-0.3, 0.4, -0.2],
            'season_w': 0.5, #higher will reduce the noise
            'noise_exp': 1,
            'snr': 3
        },
        8: {
            'freq': fixed_freqs,
            'amplitudes': [0.5, 0.5, 0.5, 0.5, 0.5],
            'trend_type': 'polynomial',
            'trend_params': [-0.3, 0.4, -0.2],
            'season_w': 0.5, #higher will reduce the noise
            'noise_exp': 1,
            'snr': 3
        },
        9: {
            'freq': fixed_freqs,
            'amplitudes': [0.2, 0.5, 0.5, 1.5, 0.75],
            'trend_type': 'polynomial',
            'trend_params': [-0.3, 4, -0.2],
            'season_w': 0.5, #higher will reduce the noise
            'noise_exp': 2,
            'snr': 0.003
        },
        10: {
            'freq': fixed_freqs,
            'amplitudes': [0.2, 0.5, 0.3, 0.5, 1.2],
            'trend_type': 'linear',
            'trend_params': 0,
            'season_w': 1,
            'noise_exp': 0,
            'snr': 1
        }
    }

    gen = SyntheticTSGenerator(duration_years, start_date)
    gen.generate_dataset(regions, train_regions, test_region, train_frac, out_dir='.')