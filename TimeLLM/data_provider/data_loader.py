import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def calculate_downsampling_factor(root_path, data_path, period_of_interest='1 year', timesteps=96):
    # Read the dataset to determine frequency
    timestamp_col = pd.to_datetime(pd.read_csv(os.path.join(root_path, data_path))['date'])
    sample_frequency = (timestamp_col[1] - timestamp_col[0]).total_seconds() / (60 * 60 * 24)  # samples per day

    # Calculate total samples in the period of interest
    if period_of_interest.lower() == '1 year':
        period_days = 365
    elif period_of_interest.lower() == '6 months':
        period_days = 182
    elif period_of_interest.lower() == '1 month':
        period_days = 30
    elif period_of_interest.lower() == '1 week':
        period_days = 7
    elif period_of_interest.lower() == '1 day':
        period_days = 1
    elif period_of_interest.lower() == '12 hours':
        period_days = 0.5  # Half a day
    elif period_of_interest.lower() == '6 hours':
        period_days = 0.25  # Quarter of a day
    else:
        print(f"Unsupported period_of_interest: {period_of_interest}")
        return 1

    print(f"period days:{period_days}")
    print(f'sample freq{sample_frequency}')
    total_samples_in_period = period_days / sample_frequency
    
    # Calculate the downsampling factor to get the desired number of timesteps
    print("total_samples_in_period", total_samples_in_period)
    print("timesteps", timesteps)
    downsampling_factor = max(1, int(np.floor(total_samples_in_period / timesteps)))    
    return downsampling_factor

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', col_percent=100, percent=100,
                 seasonal_patterns=None, dsampfactor=None, pretrain=False, split_type='temporal'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        assert split_type in ['temporal', 'covariate']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.col_percent = col_percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.pretrain = pretrain
        self.split_type = split_type

        self.timesteps = self.pred_len
        self.root_path = root_path
        self.data_path = data_path
        self.downsampling_factor = dsampfactor
        self.__read_data__()

        self.data_x = self.data_x[::self.downsampling_factor]
        self.data_y = self.data_y[::self.downsampling_factor]
        self.data_stamp = self.data_stamp[::self.downsampling_factor]
        
        self.enc_in = self.data_x.shape[-1]
        print("len data_x", len(self.data_x))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Handle column selection based on col_percent
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        num_cols_to_keep = int(len(cols) * (self.col_percent / 100))
        cols = cols[:num_cols_to_keep]
        df_raw = df_raw[['date'] + cols + [self.target]]

        if self.split_type == 'temporal':
            if self.pretrain:
                border1s = [0, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
                border2s = [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            else:
                border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
                border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            if self.set_type == 0:
                border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        else:  # covariate split
            # Split columns into train/val/test (70%/15%/15%)
            all_cols = cols + [self.target] if (self.features == 'M' or self.features == 'MS') else [self.target]
            train_cols, temp_cols = train_test_split(all_cols, train_size=0.7, random_state=42)
            val_cols, test_cols = train_test_split(temp_cols, train_size=0.5, random_state=42)
            split_cols = {'train': train_cols, 'val': val_cols, 'test': test_cols}
            selected_cols = split_cols['train'] if self.set_type == 0 else split_cols['val'] if self.set_type == 1 else split_cols['test']
            df_data = df_raw[['date'] + selected_cols]
            border1 = 0
            border2 = len(df_raw)
            if self.percent < 100:  # Apply percent reduction to rows
                border2 = int(len(df_raw) * (self.percent / 100))

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_data.columns[1:] if self.split_type == 'covariate' else df_raw.columns[1:]
            df_data = df_data[cols_data] if self.split_type == 'covariate' else df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            if self.split_type == 'temporal':
                train_data = df_data[border1s[0]:border2s[0]]
            else:
                train_cols = train_cols if (self.features == 'M' or self.features == 'MS') else [self.target]
                train_data = df_raw[train_cols][:int(len(df_raw) * 0.7)]  # Use train columns for scaling
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2] if self.split_type == 'temporal' else data
        self.data_y = data[border1:border2] if self.split_type == 'temporal' else data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100, col_percent=100,
                 seasonal_patterns=None, dsampfactor='None', split_type='temporal'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        assert split_type in ['temporal', 'covariate']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.col_percent = col_percent
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.split_type = split_type

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')

        # Set percentage of columns to keep
        num_cols_to_keep = int(len(cols) * (self.col_percent / 100))
        cols = cols[:num_cols_to_keep]
        df_raw = df_raw[['date'] + cols + [self.target]]

        if self.split_type == 'temporal':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:  # covariate split
            all_cols = cols + [self.target] if (self.features == 'M' or self.features == 'MS') else [self.target]
            train_cols, temp_cols = train_test_split(all_cols, train_size=0.7, random_state=42)
            val_cols, test_cols = train_test_split(temp_cols, train_size=0.5, random_state=42)
            split_cols = {'train': train_cols, 'val': val_cols, 'test': test_cols}
            selected_cols = split_cols['train'] if self.set_type == 0 else split_cols['val'] if self.set_type == 1 else split_cols['test']
            df_data = df_raw[['date'] + selected_cols]
            border1 = 0
            border2 = len(df_raw)
            if self.percent < 100:
                border2 = int(len(df_raw) * (self.percent / 100))

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_data.columns[1:] if self.split_type == 'covariate' else df_raw.columns[1:]
            df_data = df_data[cols_data] if self.split_type == 'covariate' else df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            if self.split_type == 'temporal':
                train_data = df_data[border1s[0]:border2s[0]]
            else:
                train_cols = train_cols if (self.features == 'M' or self.features == 'MS') else [self.target]
                train_data = df_raw[train_cols][:int(len(df_raw) * 0.7)]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2] if self.split_type == 'temporal' else data
        self.data_y = data[border1:border2] if self.split_type == 'temporal' else data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100, col_percent=100,
                 seasonal_patterns=None, dsampfactor='None', pretrain=0, split_type='temporal'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        assert split_type in ['temporal', 'covariate']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.col_percent = col_percent
        self.split_type = split_type

        self.timesteps = self.pred_len
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.downsampling_factor = dsampfactor
        print("downsampling factor: ", self.downsampling_factor)
        self.data_x = self.data_x[::self.downsampling_factor]
        self.data_y = self.data_y[::self.downsampling_factor]
        self.data_stamp = self.data_stamp[::self.downsampling_factor]
        
        self.enc_in = self.data_x.shape[-1]
        print("data x len", len(self.data_x))
        print("data seq len", self.seq_len)
        print("data pred len", self.pred_len)
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        cols = list(df_raw.columns)
        if self.target in cols:
            cols.remove(self.target)
        else:
            print(f"Warning: Target column '{self.target}' not found in the DataFrame. Skipping removal.")
        
        if 'date' in cols:
            cols.remove('date')
        else:
            print("Warning: 'date' column not found in the DataFrame.")

        num_cols_to_keep = int(len(cols) * (self.col_percent / 100))
        cols = cols[:num_cols_to_keep]

        # Reconstruct the DataFrame with 'date' and 'target' only if they exist
        selected_columns = []
        if 'date' in df_raw.columns:
            selected_columns.append('date')
        selected_columns.extend(cols)
        if self.target in df_raw.columns:
            selected_columns.append(self.target)

        df_raw = df_raw[selected_columns]
        print("df_raw added: ", df_raw)
        

        if self.split_type == 'temporal':
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:  # covariate split
            all_cols = cols #+ [self.target] if (self.features == 'M' or self.features == 'MS') else [self.target]
            train_cols, temp_cols = train_test_split(all_cols, train_size=0.7, random_state=42)
            val_cols, test_cols = train_test_split(temp_cols, train_size=0.5, random_state=42)
            split_cols = {'train': train_cols, 'val': val_cols, 'test': test_cols}
            selected_cols = split_cols['train'] if self.set_type == 0 else split_cols['val'] if self.set_type == 1 else split_cols['test']
            df_data = df_raw[['date'] + selected_cols]
            border1 = 0
            border2 = len(df_raw)
            if self.percent < 100:
                border2 = int(len(df_raw) * (self.percent / 100))

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_data.columns[1:] if self.split_type == 'covariate' else df_raw.columns[1:]
            df_data = df_data[cols_data] if self.split_type == 'covariate' else df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            #print("Only M is supported because no target column")

        if self.scale:
            if self.split_type == 'temporal':
                train_data = df_data[border1s[0]:border2s[0]]
            else:
                train_cols = train_cols if (self.features == 'M' or self.features == 'MS') else [self.target]
                train_data = df_raw[train_cols][:int(len(df_raw) * 0.7)]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2] if self.split_type == 'temporal' else data
        self.data_y = data[border1:border2] if self.split_type == 'temporal' else data
        self.data_stamp = data_stamp
        
        print(f"self.data_x shape (rows, covariates): {self.data_x.shape}")

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))
        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]
        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask