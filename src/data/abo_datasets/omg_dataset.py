

import skimage
import os
import sys
import igl
import time
import h5py
import scipy
import torch
import contextlib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from xgutils import nputil, sysutil, geoutil, datautil
from xgutils.vis import visutil

import glob

from src.data.basic_dataset import NpyListDataset
import pandas as pd

ABO_id2label = {0: 'air conditioner', 1: 'bag', 2: 'battery charger', 3: 'bed', 4: 'bench', 5: 'birdhouse', 6: 'book or journal', 7: 'bottle rack', 8: 'bowl', 9: 'cabinet', 10: 'candle holder', 11: 'cart', 12: 'chair', 13: 'clock', 14: 'clothes hook', 15: 'clothes rack', 16: 'container or basket', 17: 'cooking pan', 18: 'cup', 19: 'dining set', 20: 'dresser', 21: 'drink coaster', 22: 'easel', 23: 'electrical cable', 24: 'exercise equipment', 25: 'exercise mat', 26: 'exercise weight', 27: 'fan', 28: 'figurine or sculpture', 29: 'file folder', 30: 'fire pit', 31: 'floor mat', 32: 'heater', 33: 'holder', 34: 'instrument stand', 35: 'jar', 36: 'ladder', 37: 'lamp', 38: 'laptop stand', 39: 'mattress', 40: 'mirror', 41: 'mount', 42: 'mouse pad', 43: 'office appliance', 44: 'ottoman', 45: 'picture frame or painting', 46: 'pillow', 47: 'plant or flower pot', 48: 'rug', 49: 'shelf', 50: 'shredder', 51: 'soap dispenser', 52: 'sofa', 53: 'speaker stand', 54: 'sports equipment', 55: 'step stool', 56: 'table', 57: 'tent', 58: 'trash can', 59: 'tray', 60: 'vanity', 61: 'vase', 62: 'wagon'}
ABO_label2id = {tup[1]:tup[0] for tup in ABO_id2label.items()}
# subABO_label2id = {0: "table", 1: "sofa", 2: "table", 3: "lamp"}
subABO_label2id = {0: "table", 1: "sofa", 2: "table", 3: "lamp"}
# chair: 03001627, table: 04379243 
shapenetid2label = {18: "chair", 47: "sofa", 49: "table", 30: "lamp"} # the order is from https://huggingface.co/datasets/ShapeNet/ShapeNetCore/tree/main
label2shapenetid= {v:k for k,v in shapenetid2label.items()}

def omg_np2th(omage, threshed=True):
    if type(omage) is torch.Tensor:
        return omage.float()

    if omage.dtype==np.uint16:
        omage = omage.astype(np.float32) / 65535.
    if threshed:
        omage[..., 3] = (omage[..., 3] >= 0.5)
        omage[ omage[...,3] ==0 ] = 0.

    # convert to pytorch
    omage = torch.from_numpy(omage).float()
    omage = omage.permute(2, 0, 1) # (C, H, W)
    omage = omage * 2 - 1 # turn [0, 1] to [-1, 1]
    return omage

def omg_th2np(omage):
    if type(omage) is np.ndarray:
        return omage
    # convert to numpy
    omage = omage.permute(1, 2, 0).cpu().numpy()
    omage = (omage + 1) / 2
    return omage

class Omages1024_ABO(Dataset):
    def __init__(self, metadf_path, split="train", split_ratio=.9, duplicate = 1, mode='', seed=314):
        self.__dict__.update(locals())
        self.meta_df = pd.read_pickle(metadf_path)
        self.data_path = os.path.dirname(metadf_path)
        self.df = self.meta_df[ self.meta_df['success'] ]
        
        splits = datautil.generate_split(N=len(self.df), splits=[split_ratio, 1-split_ratio], seed=seed)
        self.train_split, self.val_split = splits
        if 'overfit' not in self.mode:
            self.df = self.df.iloc[self.train_split] if split=="train" else self.df.iloc[self.val_split]
        
    def __len__(self):
        if 'overfit' in self.mode:
            return self.duplicate if self.split == "train" else 10
        return len(self.df) * self.duplicate
    def __getitem__(self, ind):
        ind = ind % len(self.df)
        if 'overfit' in self.mode:
            ind = 0
        # check if either omg or omage exists in the df
        omg_path = self.data_path + '/' + self.df.iloc[ind]["path"] + '/omage_tensor.npz'
        omage = np.load(omg_path)['arr_0']
        omage = omg_np2th(omage, threshed=True)
        cate_id = self.df.iloc[ind]["cate_id"]
        ditem = dict(omage=omage, ind=np.array(ind), cate_label=np.array(cate_id))
        return ditem

# dataset for omages64
class OmgABO64(Dataset):
    def __init__(self, dset_df='datasets/ABO/omages/df_p64_m02_res64', split="train", mode='threshed', split_ratio=.9, duplicate = 1, seed=314, cates='all', **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        self.meta_df = dset_df + '_meta.json'
        self.h5path = dset_df + '.h5'
        self.full_df = pd.read_json(self.meta_df, orient='index')
        self.full_df["full_dset_ind"] = np.arange(len(self.full_df), dtype=int)
        self.df = self.full_df[self.full_df['success']] # filter out shape_id that failed to process

        if type(cates) is not str:
            self.df = self.df[self.df["cate_text"].isin(cates)]
        if '4catemode' in self.mode or '4cateval' in self.mode:
            with nputil.temp_seed(self.seed): # this also works for pandas sample  
                chair_sid = self.df[self.df["cate_text"]=="chair"].sample(512).index
                sofa_sid = self.df[self.df["cate_text"]=="sofa"].sample(512).index
                table_sid = self.df[self.df["cate_text"]=="table"].sample(512).index
                lamp_sid = self.df[self.df["cate_text"]=="lamp"].sample(512).index
            all_sid = np.concatenate([chair_sid, sofa_sid, table_sid, lamp_sid])
            self.df = self.df.loc[all_sid]
        if 'allcateval10' in self.mode:
            all_sids = []
            with nputil.temp_seed(self.seed):
                for catei in range(63):
                    subdf = self.df[self.df["cate_text"]==ABO_id2label[catei]]
                    if len(subdf) < 10:
                        cate_sid = subdf.sample(10, replace=True).index
                    else:
                        cate_sid = subdf.sample(10).index
                    all_sids.append(cate_sid)
            all_sid = np.concatenate(all_sids)
            self.df = self.df.loc[all_sid]

        if ("fullsplit" not in self.mode) and ('eval' not in self.mode):
            splits = datautil.generate_split(N=len(self.df), splits=[split_ratio, 1-split_ratio], seed=seed)
            self.train_split, self.val_split = splits
            if 'overfit' not in self.mode:
                self.df = self.df.iloc[self.train_split] if split=="train" else self.df.iloc[self.val_split]
    def __len__(self):
        if 'overfit' in self.mode:
            return self.duplicate if self.split == "train" else 10
        return len(self.df) * self.duplicate
    def ind2name(self, ind):
        return self.df.index[ind]
    def ind2path(self, ind):
        return self.df.iloc[ind]["path"]

    def __getitem__(self, ind):
        ind = ind % len(self.df)
        if 'overfit' in self.mode:
            ind = ind % 10
        #omage = self.df.iloc[ind]["omage"]
        with h5py.File(self.h5path, 'r') as f:
            full_dset_ind = self.df.iloc[ind]["full_dset_ind"]
            omage = f['omage'][full_dset_ind]
        omage = omg_np2th(omage, threshed=True)
        shape_id = self.df.index[ind]
        cate_label = np.array(self.df.iloc[ind]["cate_id"])
        ditem = dict(omage=omage, ind=torch.tensor(ind), cate_label=torch.tensor(cate_label))
        return ditem

class G2M_Label_OmgABO(OmgABO64):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __getitem__(self, ind):
        ind = ind % len(self.df)
        if 'overfit' in self.mode:
            ind = 0
        ditem = super().__getitem__(ind)
        if 'nonormal' in self.mode:
            img      = ditem['omage'][7:]
        else:
            img      = ditem['omage'][4:]
        cond_img = ditem['omage'][:4]
        ditem['img']      = img
        ditem['cond_img'] = cond_img
        return ditem

class N2G_Label_OmgABO(OmgABO64):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __getitem__(self, ind):
        ind = ind % len(self.df)
        if 'overfit' in self.mode:
            ind = 0
        ditem = super().__getitem__(ind)
        img = ditem['omage'][:4]
        ditem['img'] = img
        return ditem
class N2M_Label_OmgABO(OmgABO64):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __getitem__(self, ind):
        ind = ind % len(self.df)
        if 'overfit' in self.mode:
            ind = 0
        ditem = super().__getitem__(ind)
        img = ditem['omage'][7:]
        ditem['img'] = img
        return ditem

import numpy as np
import pandas as pd
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
def generate_split(N, splits=[.8,.1,.1], shuffle=True, seed=314):
    """ return a list of splits of np.arange(N)
    """
    # split np.arange(N) into splits
    splits = np.array(splits)
    assert np.sum(splits) == 1
    index = np.arange(N)
    with temp_seed(seed):
        if shuffle:
            np.random.shuffle(index)
        accu_splits = np.round(np.cumsum(splits) * N).astype(int)
        assert accu_splits[-1] == N
        splits = np.split(index, accu_splits)[:-1]
    return splits

if __name__ == "__main__":
    pass
