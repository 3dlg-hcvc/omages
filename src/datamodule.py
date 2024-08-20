import lightning
import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from src import data
from xgutils import *
from xgutils.vis import npfvis

# new datamodule, assuming options to have (_target_, **kwargs) structure
class DataModule(lightning.LightningDataModule):
    def __init__(self, batch_size: int = 32, test_batch_size=None, val_batch_size=None, num_workers: int = 8,
                 trainset_opt={'_target_':None},
                 valset_opt={'_target_':None},
                 testset_opt={'_target_':None},
                 visualset_opt={'_target_':None},
                 modify_split=True):
        super().__init__()
        trainset_opt = copy.deepcopy(trainset_opt)
        testset_opt = copy.deepcopy(testset_opt)
        if modify_split:
            if 'split' not in trainset_opt:
                trainset_opt['split'] = 'train'
            if 'split' not in valset_opt:
                valset_opt['split'] = 'val'
            if 'split' not in testset_opt:
                testset_opt['split'] = 'test'

        self.__dict__.update(locals())
        self.test_batch_size = test_batch_size if test_batch_size is not None else batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else test_batch_size
        self.dims = (1, 1, 1)
        self.pin_memory = False

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_set, self.val_set, self.test_set = None, None, None
        self.train_set = sysutil.instantiate_from_opt(self.trainset_opt)
        self.val_set = sysutil.instantiate_from_opt(self.valset_opt)
        if stage == "test" or stage is None or self.val_set is None or "class" in self.testset_opt:
            self.test_set = sysutil.instantiate_from_opt(self.testset_opt)

        if self.valset_opt["_target_"] is None:
            self.val_set = self.test_set
            self.val_batch_size = self.test_batch_size

        if self.visualset_opt["_target_"] is None:
            self.visual_set = self.val_set
        else:
            self.visual_set = sysutil.instantiate_from_opt(self.visualset_opt)

    def train_dataloader(self, shuffle=True):
        print(self.train_set)
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self, shuffle=False):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self, shuffle=False):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def visual_dataloader(self, shuffle=False):
        return DataLoader(self.visual_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=self.pin_memory)

