import argparse
import os
import torch
import shutil
import yaml
from datetime import datetime

from . import sysutil
#from nnrecon.options import Options
#from nnrecon.trainer import Trainer

FILE_DIR            = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT       = os.path.join(os.path.expanduser('~'), 'temp_project')
def generate_meta_info(root_dir, name, src_name='src'):
    root_dir = os.path.abspath(root_dir)
    m = argparse.Namespace()
    m.src_dir            = os.path.join(root_dir, src_name)
    m.datasets_dir       = os.path.join(root_dir, 'datasets/')
    m.experiments_dir    = os.path.join(root_dir, 'experiments/')
    #options_dir        = os.path.join(root_dir, 'datasets/')
    m.expr_dir        = os.path.join(m.experiments_dir, name)
    m.logs_dir        = os.path.join(m.expr_dir, 'logs')
    m.checkpoints_dir = os.path.join(m.expr_dir, 'checkpoints')
    m.results_dir     = os.path.join(m.expr_dir, 'results')
    m.session_name= name + '_' + datetime.now().strftime('%y%m%d_%H%M')
    meta_info = m.__dict__
    return meta_info
def get_opt(yaml, root_dir = DEFAULT_ROOT, src_name='src'):
    if type(yaml) is str: # is from file
        opt = load_option( yaml )
    elif type(yaml) is dict:
        opt = yaml
    name = opt.get('expr_name')
    if name is None: # not specified, using the yaml file name instead
        if type(yaml) is str: # is from file
            opt["expr_name"] = os.path.splitext(os.path.basename(yaml))[0]
            name = opt["expr_name"]
        else:
            raise ValueError('You should specify expr_name')
    opt['meta_info'] = generate_meta_info(root_dir=root_dir, name=name, src_name=src_name)
    return opt
def expr_mkdirs(opt):
    m = opt['meta_info']
    sysutil.mkdirs([m['expr_dir'],m['logs_dir'],m['checkpoints_dir'],m['results_dir']])
def dump(opt, target):
    with open(target, 'w') as file:
        documents = yaml.dump(opt, file)
def load_option(path, default_path=None):
    ''' Loads option file.
    Args:
        path (str): path to option file
        default_path (bool): whether to use default path
    ''' 
    # Load option from file itself
    with open(path, 'r') as f:
        this_opt = yaml.load(f, Loader=yaml.FullLoader)

    # Check if we should inherit from a option
    inherit_from = this_opt.get('inherit_from')

    # If yes, load this option first as default
    # If no, use the default_path
    if inherit_from is not None:
        full_path = os.path.abspath( os.path.join( os.path.dirname(path), inherit_from))
        if os.path.exists(full_path):
            inherit_from = full_path
        inherit_opt = load_option(inherit_from)
    else:
        inherit_opt = dict()

    # Include main option
    sysutil.dictUpdate(inherit_opt, this_opt)

    return inherit_opt


def _unit_test():
    opt = get_opt(os.path.join(FILE_DIR,'tests/options_test1.yaml'))
    print(opt)
if __name__ == '__main__':
    _unit_test()
