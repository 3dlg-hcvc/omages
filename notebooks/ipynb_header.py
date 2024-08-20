from IPython import get_ipython

# Get the current IPython instance
ipython = get_ipython()

if ipython is not None:
    # Run magic commands
    ipython.run_line_magic('matplotlib', 'inline')
    ipython.run_line_magic('config', "InlineBackend.figure_format = 'retina'")
    ipython.run_line_magic('reload_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import os
import sys
import copy 
import json
import h5py 
import warnings
import logging
import einops

import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
#import torch
#sys.path.append('../')
# chdir to the parent folder of the absolute path of this file
import pathlib
file_path = pathlib.Path(__file__).parent.resolve()
print(file_path.parents[0])
os.chdir( file_path.parents[0]  )
#from nnrecon.trainer2 import Trainer
from src.trainer import Trainer
import glob
from xgutils import nputil,sysutil,ptutil,plutil,geoutil
from xgutils.vis import visutil,fresnelvis
from xgutils.vis import plt3d
import igl
import numpy as np
import torch
#torch.cuda.set_device("cuda:1")
import skimage
import pandas as pd
