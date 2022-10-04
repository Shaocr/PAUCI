from torch.utils.data import DataLoader
from .dataset import (Dataset, TLDataset, IHTDataset, 
        NMDataset, BSDataset, ADADataset)
from .sampler import StratifiedSampler, MStratifiedSampler


def get_datasets(args):

    resampler_type = args.get('resampler_type', "None")
    if resampler_type == "None":
        TrainSet = Dataset
    elif resampler_type == 'TL':
        TrainSet = TLDataset
    elif resampler_type == 'IHT':
        TrainSet = IHTDataset
    elif resampler_type == 'NM':
        TrainSet = NMDataset
    elif resampler_type == 'BS':
        TrainSet = BSDataset
    elif resampler_type == 'ADA':
        TrainSet = ADADataset
    else:
        raise RuntimeError('Unknown sampler_type: %s'%resampler_type)

    train_set = TrainSet(args, 'train')
    val_set = Dataset(args, 'test')
    test_set = Dataset(args, 'val')

    return train_set, val_set, test_set

def get_data_loaders(train_set,
                     val_set,
                     test_set,
                     train_batch_size,
                     test_batch_size,
                     num_workers=4,
                     rpos = 1,
                     rneg = 4,
                     random_state = 1234):
    sampler = StratifiedSampler(train_set.get_labels(),
                                train_batch_size,
                                rpos = rpos,
                                rneg = rneg,
                                random_state=random_state)
                                
    train_loader = DataLoader(train_set,
                              batch_size=sampler.real_batch_size,
                            #   shuffle=True,
                              sampler=sampler,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set,
                            batch_size=test_batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader, sampler.data_num
def get_mdata_loaders(train_set,
                     val_set,
                     test_set,
                     train_batch_size,
                     test_batch_size,
                     num_workers=4,
                     rpos = 1,
                     rneg = 4,
                     random_state = 1234):
    sampler = MStratifiedSampler(train_set.get_labels(),
                                train_batch_size,
                                rpos = rpos,
                                rneg = rneg,
                                random_state=random_state)
                                
    train_loader = DataLoader(train_set,
                              batch_size=sampler.real_batch_size,
                            #   shuffle=True,
                              shuffle=False,
                              pin_memory=True, ##锁页内存
                              drop_last=True,
                              sampler=sampler,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set,
                            batch_size=test_batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader

__all__ = ['Dataset', 'get_datasets', 'get_data_loaders', 'get_mdata_loaders', 'StratifiedSampler']
