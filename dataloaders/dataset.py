import torch
import numpy as np
import os
import os.path as osp
import cv2
import pandas as pd
import time
import pickle
import lmdb
# *******************
import sys
sys.path.append(os.pardir)
# *******************
from tqdm import tqdm
from imblearn.under_sampling import TomekLinks,\
    InstanceHardnessThreshold, NearMiss
from imblearn.over_sampling import ADASYN, BorderlineSMOTE

from .base_dataset import BaseDataset
from .base_dataset import pil_loader


def build_lmdb(save_path, metas, commit_interval=1000):
    if not save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if osp.exists(save_path):
        print('Folder [{:s}] already exists.'.format(save_path))
        return

    if not osp.exists('/'.join(save_path.split('/')[:-1])):
        os.makedirs('/'.join(save_path.split('/')[:-1]))
    data_size_per_img = cv2.imread(metas[0][0], cv2.IMREAD_UNCHANGED).nbytes
    data_size = data_size_per_img * len(metas)
    env = lmdb.open(save_path, map_size=data_size * 10)
    txn = env.begin(write=True)

    shape = dict()

    print('Building lmdb...')
    for i in tqdm(range(len(metas))):
        image_filename = metas[i][0]
        img = pil_loader(filename=image_filename)
        assert img is not None and len(img.shape) == 3

        txn.put(image_filename.encode('ascii'), img.copy(order='C'))
        shape[image_filename] = '{:d}_{:d}_{:d}'.format(img.shape[0], img.shape[1], img.shape[2])

        if i % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    pickle.dump(shape, open(os.path.join(save_path, 'meta_info.pkl'), "wb"))

    txn.commit()
    env.close()
    print('Finish writing lmdb.')

def get_all_files(dir, ext):
    for e in ext:
        if dir.lower().endswith(e):
            return [dir]

    if not osp.isdir(dir):
        return []

    file_list = os.listdir(dir)
    ret = []
    for i in file_list:
        ret += get_all_files(osp.join(dir, i), ext)
    return ret


class Dataset(BaseDataset):
    def __init__(self, args, split='train', **kwargs):
        super().__init__(args)
        self.data_dir = osp.join(args.data_dir, split)
        if not 'class2id' in args.keys():
            self.class2id = dict()
            for i in range(args.num_classes):
                self.class2id[str(i)] = i
        else:
            self.class2id = args.get('class2id')
        self.split = split
        if split == 'train':
            self.transform = self.transform_train()
        elif split == 'val':
            self.transform = self.transform_validation()
        elif split == 'test':
            self.transform = self.transform_validation()
        else:
            raise ValueError

        self.args = args
        self.tmp = None 
        self.data = None 
        self.targets = None
        self.metas = []

        if self.args.get('npy_style', False):
        # **********************************************
            self.tmp = np.load('../' + self.data_dir + '.npy', allow_pickle=True).item()
            self.data = self.tmp['data']
            self.targets = self.tmp['targets']
            assert len(self.data) == len(self.targets)      
            
            ## resample data, only support tiny dataset
            data_shape = list(self.data.shape)
            self.data, self.targets = self.resample(self.data.reshape((data_shape[0], -1)), self.targets)
            data_shape[0] = self.data.shape[0]
            self.data = self.data.reshape(data_shape)
            del data_shape

            self.img_list = ['%08d'%i for i in range(len(self.data))]
            # self.metas.append((self.data[i], int(self.targets[i])))
            n_cls_p = len(set([0] + [i[1] for i in list(self.class2id.items())])) - 1
            for i in range(len(self.targets)):
                cls_id = self.class2id.get(str(self.targets[i]), 0)
                if cls_id < 0:
                    continue
                label_one_hot = np.zeros(n_cls_p)
                if cls_id > 0:
                    label_one_hot[cls_id - 1] = 1
                self.metas.append((self.data[i], label_one_hot))
            args.use_lmdb = False
            self.args.use_lmdb = False
        else:
            self.img_list = get_all_files(self.data_dir, ['jpg', 'jpeg', 'png'])
            self._gen_metas(self.img_list)

        self._num = len(self.metas)
        print('%s set has %d images' % (self.split, self.__len__()))

        if args.get('use_lmdb', False):
            self.lmdb_dir = osp.join(args.lmdb_dir, split + '.lmdb')
            build_lmdb(self.lmdb_dir, self.metas)
            self.initialized = False
            self._load_image = self._load_image_lmdb
        else:
            self._load_image = self._load_image_pil


        self._labels = [int(i[1].sum() != 0) for i in self.metas]
        # self._labels = [i[1] for i in self.metas]
        self._cls_num_list = pd.Series(self._labels).value_counts().sort_index().values
        self._freq_info = [
            num * 1.0 / sum(self._cls_num_list) for num in self._cls_num_list
        ]
        self._num_classes = len(self._cls_num_list)
        self._class_dim = len(set(self._labels))
        print('class number: ', self._cls_num_list)
        
    def resample(self, x, y):
        return x, y

    def self_paced_samples(self, mask_label=None):

        r'''
        This function is adopted to dynamically select tough samples to use to train,
            only used for self-paced learning process. 
        '''
        if mask_label is None:
            return

        assert len(self.img_list) == len(mask_label)

        img_list = [img for (img, mask) in zip(self.img_list, mask_label) if mask ]

        assert sum(mask_label) == len(img_list)

        if self.args.get('npy_style', False):
            data = [i for (i, mask) in zip(self.data, mask_label) if mask]
            targets = [i for (i, mask) in zip(self.targets, mask_label) if mask]
            assert len(data) == len(targets)
            self.img_list = ['%08d'%i for i in range(len(self.data))]
            self.metas = []
            # n_cls_p = sum([i[1] > 0 for i in list(self.class2id.items())])
            n_cls_p = len(set([0] + [i[1] for i in list(self.class2id.items())])) - 1
            for i in range(len(targets)):
                cls_id = self.class2id.get(str(targets[i]), 0)
                label_one_hot = np.zeros(n_cls_p)
                if cls_id > 0:
                    label_one_hot[cls_id - 1] = 1
                self.metas.append((data[i], label_one_hot))
            self.args.use_lmdb = False
        else:
            self._gen_metas(img_list)

        self._num = len(self.metas)
        print('%s set has %d images' % (self.split, self.__len__()))
        # logger.info('%s set has %d images' % (self.split, self.__len__()))

        self._labels = [int(i[1].sum() != 0) for i in self.metas]
        # self._labels = [i[1] for i in self.metas]
        self._cls_num_list = pd.Series(self._labels).value_counts().sort_index().values
        self._freq_info = [
            num * 1.0 / sum(self._cls_num_list) for num in self._cls_num_list
        ]
        self._num_classes = len(self._cls_num_list)
        self._class_dim = len(set(self._labels))

    def _gen_metas(self, img_list):
        self.metas = []
        # n_cls_p = sum([i[1] > 0 for i in list(self.class2id.items())])
        n_cls_p = len(set([0] + [i[1] for i in list(self.class2id.items())])) - 1
        if osp.exists(osp.join(self.data_dir, '../labels.txt')):
            name2label = dict()
            with open(osp.join(self.data_dir, '../labels.txt')) as f:
                lines = f.read().splitlines()
            for line in lines:
                name, label = line.split(',')
                label_one_hot = np.zeros(n_cls_p)
                for l in label.split('|'):
                    l = self.class2id[l]
                    if l > 0:
                        label_one_hot[l - 1] = 1
                name2label[name] = label_one_hot

            for i in img_list:
                cls_id = name2label[i.split('/')[-1]]
                self.metas.append((i, cls_id))
        else:
            for i in img_list:
                cls_id = self.class2id.get(i.split('/')[-2], 0)
                if cls_id < 0:
                    continue
                label_one_hot = np.zeros(n_cls_p)
                if cls_id > 0:
                    label_one_hot[cls_id - 1] = 1
                self.metas.append((i, label_one_hot))

    def _init_lmdb(self):
        if not self.initialized:
            env = lmdb.open(self.lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
            self.lmdb_txn = env.begin(write=False)
            self.meta_info = pickle.load(open(os.path.join(self.lmdb_dir, 'meta_info.pkl'), "rb"))
            self.initialized = True

    def _load_image_lmdb(self, img_filename):
        self._init_lmdb()
        img_buff = self.lmdb_txn.get(img_filename.encode('ascii'))
        C, H, W = [int(i) for i in self.meta_info[img_filename].split('_')]
        img = np.frombuffer(img_buff, dtype=np.uint8).reshape(C, H, W)
        return img

    def _load_image_pil(self, img_filename):
        return pil_loader(img_filename)

    def get_class_dim(self):
        return self._class_dim

    def get_labels(self):
        return self._labels

    def get_cls_num_list(self):
        return self._cls_num_list

    def get_freq_info(self):
        return self._freq_info

    def get_num_classes(self):
        return self._num_classes

    def __len__(self):
        return self._num

    def __str__(self):
        return self.args.data_dir + '  split=' + str(self.split)

    def _getitem(self, idx):
        sample = {
            'image': self._load_image(self.metas[idx][0]),
            'label': self.metas[idx][1]
        }
        sample = self.transform(sample)
        return sample['image'], sample['label'], idx

    def _getitem_npy(self, idx):
        sample = {
            'image': self.metas[idx][0],
            'label': self.metas[idx][1]
        }
        sample = self.transform(sample)
        return sample['image'], sample['label'], idx

    def __getitem__(self, idx):
        if self.args.get('npy_style', False):
            return self._getitem_npy(idx)
        return self._getitem(idx)


class TLDataset(Dataset):
    """
    Dataset resampled by TomekLinks
    """
    def resample(self, x, y):
        resampler = TomekLinks(sampling_strategy='majority')
        return resampler.fit_resample(x, y)


class IHTDataset(Dataset):
    """
    Dataset resampled by InstanceHardnessThreshold
    """
    def resample(self, x, y):
        resampler = InstanceHardnessThreshold(sampling_strategy='majority')
        return resampler.fit_resample(x, y)


class NMDataset(Dataset):
    """
    Dataset resampled by NearMiss
    """
    def resample(self, x, y):
        resampler = NearMiss(sampling_strategy='majority')
        return resampler.fit_resample(x, y)


class BSDataset(Dataset):
    """
    Dataset resampled by BorderlineSMOTE
    """
    def resample(self, x, y):
        resampler = BorderlineSMOTE()
        return resampler.fit_resample(x, y)


class ADADataset(Dataset):
    """
    Dataset resampled by ADASYN
    """
    def resample(self, x, y):
        resampler = ADASYN(sampling_strategy='minority')
        return resampler.fit_resample(x, y)
