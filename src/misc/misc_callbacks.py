import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
from einops import rearrange, repeat
import numpy as np

from xgutils import *



class VisTest(plutil.VisCallback):
    def __init__(self, **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)
        #self.vqvae = init_trained_model_from_ckpt(vqvae_opt)#.to("cuda")
    def compute_batch(self, batch):
        return ptutil.ths2nps(batch)
    
    def visualize_batch(self, computed):
        computed = ptutil.ths2nps(computed)
        