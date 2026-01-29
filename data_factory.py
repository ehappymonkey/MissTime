from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom ,UEAloader, PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, Dataset_PEMS
from torch.utils.data import DataLoader
import torch 
from data_provider.uea import collate_fn, padding_mask


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'electricity': Dataset_Custom,
    'traffic': Dataset_Custom,
    'weather': Dataset_Custom,
    'exchange': Dataset_Custom,
    'UEA': UEAloader,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'PEMS': Dataset_PEMS,
    'UEA': UEAloader,
}


def collate_fn_with_batch_mask(batch, missing_rate=0.25):

    indices = torch.tensor([b[0] for b in batch], dtype=torch.long) 
    xs = torch.stack([torch.tensor(b[1], dtype=torch.float32) for b in batch])
    ys = torch.stack([torch.tensor(b[2], dtype=torch.float32) for b in batch])
    x_marks = torch.stack([torch.tensor(b[3], dtype=torch.float32) for b in batch])
    y_marks = torch.stack([torch.tensor(b[4], dtype=torch.float32) for b in batch])

    C = xs.shape[2]
    while True:
        batch_mask = torch.rand(C) > missing_rate
        if batch_mask.any():  
            break
    batch_mask = batch_mask.float() # .unsqueeze(0).unsqueeze(0)  # (1, 1, C)
    xs_masked = xs * batch_mask

    return indices, xs_masked, ys, x_marks, y_marks, batch_mask, xs 

def collate_fn_with_batch_mask_PEMS(batch, missing_rate=0.25):

    xs = torch.stack([torch.tensor(b[0], dtype=torch.float32) for b in batch])
    ys = torch.stack([torch.tensor(b[1], dtype=torch.float32) for b in batch])
    x_marks = torch.stack([torch.tensor(b[2], dtype=torch.float32) for b in batch])
    y_marks = torch.stack([torch.tensor(b[3], dtype=torch.float32) for b in batch])

    C = xs.shape[2]
    while True:
        batch_mask = torch.rand(C) > missing_rate
        if batch_mask.any():  
            break
    batch_mask = batch_mask.float() # .unsqueeze(0).unsqueeze(0)  # (1, 1, C)
    xs_masked = xs * batch_mask

    return None, xs_masked, ys, x_marks, y_marks, batch_mask, xs 

def collate_fn_with_batch_mask_classification(batch, max_len=None, missing_rate=0.25):
    features, labels = zip(*batch)
    
    lengths = [X.shape[0] for X in features]
    if max_len is None:
        max_len = max(lengths)
    
    xs_full = torch.zeros(len(batch), max_len, features[0].shape[-1])
    for i in range(len(batch)):
        end = min(lengths[i], max_len)
        xs_full[i, :end, :] = features[i][:end, :]
    
    labels = torch.stack(labels).squeeze(-1)
    padding_masks = padding_mask(torch.tensor(lengths), max_len)  # (B, L)

    C = xs_full.shape[2]
    while True:
        batch_mask = torch.rand(C) > missing_rate
        if batch_mask.any():  
            break
    xs_masked = xs_full * batch_mask.float().unsqueeze(0).unsqueeze(0)  # (B, L, C)
    batch_mask = batch_mask.float()
    return xs_masked, xs_full, labels, batch_mask, padding_masks


def collate_fn_anomaly_detection(batch, missing_rate=0.25):
    seq_xs = [b[0] for b in batch]  # list of (L, C)
    seq_ys = [b[1] for b in batch]  # list of (L,) or (L, 1)

    xs_full = torch.stack([torch.tensor(x, dtype=torch.float32) for x in seq_xs])  # (B, L, C)
    ys = torch.stack([torch.tensor(y, dtype=torch.float32).squeeze() for y in seq_ys])  # (B, L)

    C = xs_full.shape[2]
    while True:
        batch_mask = torch.rand(C) > missing_rate
        if batch_mask.any(): 
            break
    xs_masked = xs_full * batch_mask.float().unsqueeze(0).unsqueeze(0)  # (B, L, C)
    batch_mask = batch_mask.float() 
    return xs_masked, xs_full, ys, batch_mask

def collate_fn_imputation(batch, missing_rate=0.25): 
    indices = torch.tensor([b[0] for b in batch], dtype=torch.long) 
    xs = torch.stack([torch.tensor(b[1], dtype=torch.float32) for b in batch])
    ys = torch.stack([torch.tensor(b[2], dtype=torch.float32) for b in batch])
    x_marks = torch.stack([torch.tensor(b[3], dtype=torch.float32) for b in batch])
    y_marks = torch.stack([torch.tensor(b[4], dtype=torch.float32) for b in batch])

    # 生成 batch-level mask
    C = xs.shape[2]
    while True:
        batch_mask = torch.rand(C) > missing_rate
        if batch_mask.any():
        # if 0 < batch_mask.sum() < C:
            break

    batch_mask = batch_mask.float() 
    xs_masked = xs * batch_mask

    return xs_masked, ys, x_marks, y_marks, batch_mask, xs 

def data_provider(args, flag, shuffle_flag=None, batch_size = None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1


    if shuffle_flag == None:
        shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    else:
        shuffle_flag = shuffle_flag 
    drop_last = False
    if batch_size ==None:
        batch_size = args.batch_size
    else:
        batch_size = batch_size 
    freq = args.freq

    if any(keyword in args.model_id for keyword in ['PEMS']) and args.task_name == 'long_term_forecast':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            # seasonal_patterns=None # We do not use this option.
        )
        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            collate_fn=lambda b: collate_fn_with_batch_mask_PEMS(b, args.mask_ratio), 
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    

    elif args.task_name == 'anomaly_detection':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda b: collate_fn_anomaly_detection(b, args.mask_ratio), # if flag != 'train' else collate_fn_anomaly_detection(b, 0)
            )
        return data_set, data_loader

    
    elif args.task_name == 'classification':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn_with_batch_mask_classification(x, None, missing_rate=args.mask_ratio), # if flag !='TRAIN' else collate_fn_with_batch_mask_classification(x, None, missing_rate=0.0)
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader

    elif args.task_name == 'long_term_forecast':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=None # We do not use this option.
        )
        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            collate_fn=lambda b: collate_fn_with_batch_mask(b, args.mask_ratio), 
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    
    elif args.task_name == 'imputation':
        data_set = Data(
            args = args,
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
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            collate_fn=lambda b: collate_fn_imputation(b, args.mask_ratio), # if flag != 'train' else collate_fn_imputation(b, 0),
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    
        return data_set, data_loader
