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

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from sklearn.model_selection import train_test_split

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=100, col_percent=100,
                 seasonal_patterns=None, dsampfactor=None,
                 pretrain=0, split_type='temporal', boundary_file=None):
        
        # seq lengths
        if size is None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train','val','test']
        assert split_type in ['temporal','covariate']
        type_map = {'train':0,'val':1,'test':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.col_percent = col_percent
        self.split_type = split_type

        # downsample factor
        try:
            self.downsampling_factor = int(dsampfactor) if dsampfactor else 1
        except:
            self.downsampling_factor = 1
        print(f"downsampling factor: {self.downsampling_factor}")

        self.root_path = root_path
        self.data_path = data_path
        self.boundary_file = boundary_file

        # load and process
        self._read_data()

        # downsample data arrays
        if self.downsampling_factor>1:
            self.data_x = self.data_x[::self.downsampling_factor]
            self.data_y = self.data_y[::self.downsampling_factor]
            self.data_stamp = self.data_stamp[::self.downsampling_factor]

        # build valid indices
        if self.boundary_file:
            with open(self.boundary_file) as f:
                orig = json.load(f).get('boundaries',[])
            # adjust boundaries
            bounds = [(s//self.downsampling_factor, e//self.downsampling_factor) for s,e in orig]
            all_starts = []
            for s,e in bounds:
                last = e - self.seq_len - self.pred_len +1
                if last> s:
                    all_starts.extend(range(s,last))
                else:
                    print(f"skip boundary ({s},{e}) too short")
            # filter any that exceed data length
            max_start = len(self.data_x)-self.seq_len-self.pred_len
            self.valid_start_indices = [i for i in all_starts if 0<=i<=max_start]
        else:
            #self.valid_start_indices = None
            # Define all possible start indices when no boundary file is provided
            max_start = len(self.data_x) - self.seq_len - self.pred_len + 1
            self.valid_start_indices = list(range(0, max_start))

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x)-self.seq_len-self.pred_len+1
        print(f"rows:{len(self.data_x)} seq_len:{self.seq_len} pred_len:{self.pred_len}")

    def _read_data(self):
        df = pd.read_csv(os.path.join(self.root_path,self.data_path))
        cols = [c for c in df.columns if c not in ['date',self.target]]
        keep = int(len(cols)*(self.col_percent/100))
        cols = cols[:keep]
        sel = []
        if 'date' in df.columns: sel.append('date')
        sel+=cols
        if self.target in df.columns: sel.append(self.target)
        df = df[sel]

        # temporal split
        if self.split_type == 'temporal':
            n = len(df)
            ntr = int(n * 0.7)
            nte = int(n * 0.2)
            nval = n - ntr - nte

            if "CarbonCast" in self.root_path: #self.boundary_file is not None:
                #print("we're in carboncast!")
                if self.boundary_file is None: #train one test one
                    ntr = int(n*4/5)
                    nval = int(n*1/5) #out of 5 years
                else:
                    ntr = int(n*4/5) #out of the 5 years and 6 regions, keeping note of the boundary file split
                    nval = int(n*1/5) 
            

            if self.set_type == 0:            # train 
                b1, b2 = 0, ntr
            elif self.set_type == 1:          # val  
                b1, b2 = ntr, ntr + nval
            else:                             # test
                
                if "CarbonCast" in self.root_path: #self.boundary_file is not None:
                    b1, b2 = 0, int(n)        # all 5 years of test to minimize variability
                else:
                
                    b1, b2 = n - nte, n     # last 20% (legacy)

            # adjust for sequence length
            b1 = max(0, b1 - self.seq_len)
        '''
        else:
            allc=cols
            tr,temp = train_test_split(allc,train_size=0.7,random_state=42)
            val,te = train_test_split(temp,train_size=0.5,random_state=42)
            split={'train':tr,'val':val,'test':te}
            df=df[['date']+split[list(type_map.keys())[self.set_type]]]
            b1=0; b2=int(len(df)*(self.percent/100))
        '''
        # data values
        if self.features in ['M','MS']:
            data_df = df.iloc[:,1:]
        else:
            data_df = df[[self.target]]

        if self.scale:
            trsl = data_df.iloc[:ntr] if self.split_type=='temporal' else data_df
            self.scaler = StandardScaler().fit(trsl.values)
            data = self.scaler.transform(data_df.values)
        else:
            data = data_df.values

        # time stamps
        raw = df['date'].iloc[b1:b2]
        raw = raw.apply(lambda x: x + ':00' if len(x.split(':')) == 2 else x)
        stamps = pd.to_datetime(raw, format='%Y-%m-%d %H:%M:%S')
        
        #stamps = pd.to_datetime(raw, infer_datetime_format=True)
        if self.timeenc==0:
            ts = pd.DataFrame({
                'month':stamps.dt.month,'day':stamps.dt.day,
                'weekday':stamps.dt.weekday,'hour':stamps.dt.hour
            })
            data_stamp=ts.values
        else:
            idx = pd.DatetimeIndex(stamps)
            data_stamp = time_features(idx, freq=self.freq).T

        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]
        self.data_stamp = data_stamp
        print(f"loaded {self.data_x.shape}")

    '''
    def __getitem__(self,index):
        if self.valid_start_indices is not None:
            s = self.valid_start_indices[index]
        else:
            feat = index//self.tot_len
            s = index%self.tot_len
        e = s+self.seq_len
        assert e-s==self.seq_len, f"got {e-s} rows, expected {self.seq_len}"
        rb = e-self.label_len
        re = rb+self.label_len+self.pred_len

        if self.valid_start_indices is not None:
            x = self.data_x[s:e]
            y = self.data_y[rb:re]
        else:
            x = self.data_x[s:e, feat:feat+1]
            y = self.data_y[rb:re, feat:feat+1]
        xm = self.data_stamp[s:e]
        ym = self.data_stamp[rb:re]

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(xm).long(),
            torch.from_numpy(ym).long()
        )
    '''
    def __getitem__(self, index):
        s = self.valid_start_indices[index]
        e = s + self.seq_len
        assert e - s == self.seq_len, f"got {e-s} rows, expected {self.seq_len}"
        rb = e - self.label_len
        re = rb + self.label_len + self.pred_len
        x = self.data_x[s:e]  # Shape: (seq_len, num_features)
        y = self.data_y[rb:re]  # Shape: (label_len + pred_len, num_features)
        xm = self.data_stamp[s:e]
        ym = self.data_stamp[rb:re]
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(xm).long(),
            torch.from_numpy(ym).long()
        )
    
    def __len__(self):
        if self.valid_start_indices is not None:
            return len(self.valid_start_indices)
        return (len(self.data_x)-self.seq_len-self.pred_len+1)*self.enc_in

    def inverse_transform(self,data):
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