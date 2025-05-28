from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'Illness': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Synthetic': Dataset_Custom,
    'Aus': Dataset_Custom
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    col_percent = args.col_percent
    dsampfactor = args.dsampfactor

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        if args.data_path_test is not "None":
            args.data_path = args.data_path_test #for the purposes of ETTh1 <-> h2
            #also use this for covariate split
    
    else:
        shuffle_flag = True #makes sense
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        if flag == 'val' and args.data_path_val is not "None": #mainly for covariate splitting into 3 diff datasets (not sure if val is elsewhere)
            args.data_path = args.data_path_val

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            col_percent=col_percent,
            seasonal_patterns=args.seasonal_patterns,
            dsampfactor=dsampfactor,
            pretrain=args.pretrain,
            split_type=args.split_type
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
