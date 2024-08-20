"""This module contains simple helper functions """
from __future__ import print_function
import collections.abc
import re
import os
import sys
import copy
import yaml
import shutil
import importlib
import subprocess


import h5py
import numpy as np
#import torch
import contextlib

from collections.abc import Iterable
import time
from tqdm import tqdm

EPS = 0.000000001

# python utils

@contextlib.contextmanager
def suppress_output(suppress=True):
    """ Usage:
        with suppress_output():
            print("This will not be printed.")
            # You can place any code here whose output you want to suppress.
    """
    if suppress:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
    else:
        yield
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
            
def dictAdd(dict1, dict2):
    for key in dict2:
        if key in dict1:
            dict1[key] += dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1



def dictAppend(dict1, dict2):
    for key in dict2:
        if key in dict1:
            dict1[key].append(dict2[key])
        else:
            dict1[key] = [dict2[key]]
    return dict1


def dictApply(func, dic):
    for key in dic:
        dic[key] = func(dic[key])


def dictMean(dicts):
    keys = dicts[0].keys()
    accum = {k: np.stack([x[k] for x in dicts if k in x]).mean() for k in keys}
    return accum


def dictUpdate(d1: dict, d2: dict, recursive=True, depth=1):
    """Updates dictionary recursively.

    Args:
        d1 (dict): first dictionary to be updated
        d1 (dict): second dictionary which entries should be used

    Returns:
        collections.abc.Mapping: Updated dict
    """
    for k, d2_v in d2.items():
        d1_v = d1.get(k, None)
        typematch = type(d1_v) is type(d2_v)
        recur = isinstance(d2_v, collections.abc.Mapping) and recursive == True
        if typematch and recur:
            d1[k] = dictUpdate(d1_v, d2_v, depth=depth+1)
        else:
            d1[k] = d2_v
    return d1


def prefixDictKey(dic, prefix=''):
    return dict([(prefix+key, dic[key]) for key in dic])


pj = os.path.join


def strNumericalKey(s):
    if s:
        try:
            c = re.findall('\d+', s)[0]
        except:
            c = -1
        return int(c)


def recursiveMap(container, func):
    # TODO
    raise NotImplementedError()
    # valid_container_type = [dict, tuple, list]
    # for item in container:
    #     if item in valid_container_type:
    #         yield type(item)(recursive_map(item, func))
    #     else:
    #         yield func(item)
# time

def flatten_tree_to_level(tree, k, depth=0, index=(), acc_ind=None):
    """ Flatten a tree of lists to a certain depth
        Args:
            tree (list): tree of lists
            k (int): depth to flatten to
            depth (int): current depth
            index (tuple): current index
            acc_ind (tuple): accumulated index, uniquely identify all elments in certain depth
        Returns:
            list: flattened tree with index: [(index, acc_ind, element), ...]
        Example:
            tree = [
                [1, 2, [3, 4]],
                [5, [6, [7, 8]]],
                [[9, [10, 11]], [12]]
            ]

            # Flatten the tree to level 2
            flattened_tree = flatten_to_level(tree, k=2)
            for idx, elem in flattened_tree:
                print(f"Index: {idx}, Element: {elem}")
            
            # Output:
            Index: [0, 0], Acc_ind: [0, 0], Element: 1
            Index: [0, 1], Acc_ind: [0, 1], Element: 2
            Index: [0, 2], Acc_ind: [0, 2], Element: [3, 4]
            Index: [1, 0], Acc_ind: [1, 3], Element: 5
            Index: [1, 1], Acc_ind: [1, 4], Element: [6, [7, 8]]
            Index: [2, 0], Acc_ind: [2, 5], Element: [9, [10, 11]]
            Index: [2, 1], Acc_ind: [2, 6], Element: [12]
    """
    current_depth = depth
    if acc_ind is None:
        acc_ind = [-1,] * k
    else:
        acc_ind[current_depth-1] += 1
    if current_depth == k:
        return [(index, tuple(acc_ind), tree, )]
    return [item for i, subtree in enumerate(tree)
                 for item in flatten_tree_to_level(subtree, k, depth + 1, index + (i,), acc_ind)]
    # # Above is normal implementation, below is generator implementation
    # if depth == k:
    #     return ((index, tree))
    # if not isinstance(tree, list):
    #     return ((index, tree))
    # return (item for i, subtree in enumerate(tree)
    #              for item in flatten_tree_to_level(subtree, k, depth + 1, index + (i,)))
    # #combining with list(itertools.chain.from_iterable()), much faster
import itertools

def flatten_to_level(tree, k, current_depth=0, index=()):
    """ faster version """
    if current_depth == k:
        yield (index, tree, 0)
    elif isinstance(tree, list):
        # Use chain.from_iterable to flatten one level of nested lists
        yield from itertools.chain.from_iterable(
            flatten_to_level(subtree, k, current_depth + 1, index + (i,))
            for i, subtree in enumerate(tree)
        )
    else:
        yield (index, tree)
    # # The below version will flatten even the (index, tree) pair if itertools.chain.from_iterable is used for some reason.
    # if current_depth == k:
    #     yield (index, tree)
    # elif isinstance(tree, list):
    #     for i, subtree in enumerate(tree):
    #         yield from flatten_to_level(subtree, k, current_depth + 1, index + (i,))
    # else:
    #     yield (index, tree)


class Timer():
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.time_stamps = []
        self.stamp_texts = []
        self.update(text='init', print_time=False)

    def update(self, text="", print_time=True):
        self.time_stamps.append(time.time())
        self.stamp_texts.append(text)
        if print_time and not self.quiet:
            print(f"Time stamp: #{len(self.time_stamps)-1} {text}")
            print(self.time_stamps[-1]-self.time_stamps[-2], self.time_stamps[-1]-self.time_stamps[0])
    def rank_interval(self):
        interval = np.diff(self.time_stamps)
        print(interval)
        # get argsort
        rank = np.argsort(interval)[::-1]
        # get sorted interval
        interval = interval[rank]
        # print out
        for i, (r, t) in enumerate(zip(rank, interval)):
            print(f"{r+1} {t:.4e} {self.stamp_texts[r+1]}")
        return rank, interval

# misc


def listdir(directory, return_path=True):
    """List dir and sort by numericals (if applicable)

    Args:
        directory (str): string repr of directory
        return_path (bool, optional): Whether return full file path instead of just file names. Defaults to True.

    Returns:
        list of str: see 'return_path'
    """
    filenames = os.listdir(directory)
    filenames.sort(key=strNumericalKey)
    if return_path == True:
        paths = [os.path.join(directory, filename) for filename in filenames]
        return filenames, paths
    else:
        return filenames


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory (recursively) if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    path = os.path.expanduser(path) # resolve the ~ symbol at the beginning of the path

    if not os.path.exists(path):
        os.makedirs(path)

def random_temp_filename(suffix='png', temp_dir='~/.tmp_xgutils/'):
    import random
    import string
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    fname = name + '.' + suffix
    
    temp_dir = os.path.expanduser('~/.temp/xgutils/')
    mkdir(temp_dir)
    path = os.path.join(temp_dir, fname)
    return path, name

def filename(path, suffix=False):
    name = path.split("/")[-1]
    if suffix == False:
        name = ".".join(name.split(".")[:-1])
    return name


def load_module_object(module_path, object_name):
    modulelib = importlib.import_module(module_path)
    exists = False
    for name, cls in modulelib.__dict__.items():
        if name == object_name:
            obj = cls
            if exists == True:
                print(
                    f'WARNING: multiple objects named {object_name} have been found in {module_path}')
            exists = True
    if exists == False:
        raise NameError(f'Object {object_name} not found in {module_path}')
    return obj


def load_object(object_path):
    splited = object_path.split('.')
    module_path = '.'.join(splited[:-1])
    object_name = splited[-1]
    return load_module_object(module_path, object_name)


def instantiate_from_opt(opt):
    # 2024.3.27 added hydra style instantiation 
    if '_target_' in opt and opt['_target_'] is not None:
        # duplicate opt and remove _target_
        opt2 = copy.deepcopy(opt)
        del opt2['_target_']
        print(opt2)
        return load_object(opt['_target_'])(**opt2)

    if not "class" in opt or opt["class"] is None:
        return None
    return load_object(opt["class"])(**opt.get("kwargs", dict()))


def get_filename(path):
    fname = os.path.basename(path)
    fname = ".".join(fname.split(".")[:-1])
    return fname


def runCommand(command, verbose=False):
    my_env = os.environ.copy()
    if verbose:
        print("Running command: %s" % command)
        print("Envirnoment variables:")
        print(my_env)
    import bashlex
    cmd = list(bashlex.split(command))
    # capture_output=True -> capture stdout&stderr
    p = subprocess.run(cmd, capture_output=True, env=my_env)
    stdout = p.stdout.decode('utf-8')
    stderr = p.stderr.decode('utf-8')
    return stdout, stderr, p.returncode


def array2batches(array, chunkSize=4):
    return [array[i*chunkSize:(i+1)*chunkSize] for i in range((len(array)+chunkSize-1)//chunkSize)]


def progbar(array, total=None):
    if total is None:
        if isinstance(array, enumerate):
            array = list(array)
        total = len(array)
    return tqdm(array, total=total, ascii=True)


def parallelMap(func, args, batchFunc=None, zippedIn=True, zippedOut=False, cores=-1, quiet=False):
    from pathos.multiprocessing import ProcessingPool
    """Parallel map using multiprocessing library Pathos
    This version implements batching to show progress bar, but it is not efficient as processes that finish earlier will be waiting for the rest of the processes to finish.

    the `pmap` function is an optimized version of this function, which utilized the `uimap` function of pathos library to avoid the problem mentioned above.

    Args:
        stderr (function): func
        args (arguments): [arg1s, arg2s ,..., argns](zippedIn==False) or [[arg1,arg2,...,argn], ...](zippedIn=True)
        batchFunc (func, optional): TODO. Defaults to None.
        zippedIn (bool, optional): See [args]. Defaults to True.
        zippedOut (bool, optional): See [Returns]. Defaults to False.
        cores (int, optional): How many processes. Defaults to -1.
        quiet (bool, optional): if do not print anything. Defaults to False.

    Returns:
        tuples: [out1s, out2s,..., outns](zippedOut==False) or [[out1,out2,...,outn], ...](zippedOut==True)
    """
    if batchFunc is None:
        def batchFunc(x): return x
    if zippedIn == True:
        args = list(map(list, zip(*args)))  # transpose
    if cores == -1:
        cores = os.cpu_count()
    batchIdx = list(range(len(args[0])))
    batches = array2batches(batchIdx, cores)
    out = []
    iterations = enumerate(
        batches) if quiet == True else progbar(enumerate(batches))
    with ProcessingPool(nodes=cores) as pool:
        for i, batch in iterations:
            batch_args = [[arg[i] for i in batch] for arg in args]
            out.extend(pool.map(func, *batch_args))
    if zippedOut == False:
        if type(out[0]) is not tuple:
            out = [(item,) for item in out]
        out = list(map(list, zip(*out)))
    return out
def pmap(func, args, zippedIn=True, zippedOut=False, cores=-1, quiet=False):
    """ Optimized version of parallelMap, which directly maps without batching. In order to show progress bar, the `uimap` function of pathos library is used. 
    To enable progress bar, `imap` is used as it does not block the main process as `map` does.
    The `uimap` method works in a `first-finish-first-out` manner, which makes the progress bar more accurate and evenly distributed.
    If using `imap` method, the progress bar will suddenly jump or stuck randomly.

    An `wfunc` function is used to wrap the original function to return the index of the input, which is used to sort the output in the original order.

    Args:
        func (function): function
        args (arguments): [arg1s, arg2s ,..., argns](zippedIn==False) or [[arg1,arg2,...,argn], ...](zippedIn=True)
        batchFunc (func, optional): TODO. Defaults to None.
        zippedIn (bool, optional): See [args]. Defaults to True.
        zippedOut (bool, optional): See [Returns]. Defaults to False.
        cores (int, optional): How many processes. Defaults to -1.
        quiet (bool, optional): if do not print anything. Defaults to False.

    Returns:
        tuples: [out1s, out2s,..., outns](zippedOut==False) or [[out1,out2,...,outn], ...](zippedOut==True)
    """
    #from pathos.multiprocessing import ProcessingPool
    from pathos.pools import ProcessPool
    import multiprocess.context as ctx
    ctx._force_start_method('spawn') # force spawn start method to avoid slow performance
    """ It turns out pytorch is not compatible with fork start method for multiprocessing in python.
    Even you just import torch without using it, all of the following multiprocessing will be slowed.
    Typically each of the CPU usage will be reduced. The more processes you use, the more slow it will be.
    This performance reduction will make the multiprocessing nearly the same as single process.
    Hence, we use ctx._force_start_method('spawn') to force the start method to be spawn.
    If you are not using pytorch, this will not be necessary.
    """

    if zippedIn == True:
        args = list(map(list, zip(*args)))  # transpose
    print(args)
    total = len(args[0])

    if cores == -1:
        cores = os.cpu_count()
    cores = min(cores, os.cpu_count())
    def wfunc(index, *args): return index, func(*args)
    with ProcessPool(nodes=cores) as pool:
        out = list(tqdm(pool.uimap(wfunc, np.arange(total), *args), total=total, ascii=True))
    out = sorted(out, key=lambda x: x[0])
    out = [x[1] for x in out]
    if zippedOut == False:
        if type(out[0]) is not tuple:
            out = [(item,) for item in out]
        out = list(map(list, zip(*out)))
    return out

@contextlib.contextmanager
def debug(if_debug=True):
    if if_debug==False:
        yield
    else:
        print("**** DEBUGGING ****")
        try:
            yield  # Control is transferred to the block using the context manager
        except Exception as exc:
            import traceback, pdb
            # Print exception details
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("Exception occurred:", exc_type)  # Print the exception type
            traceback.print_tb(exc_traceback)  # Print the stack trace
            print(getattr(exc, 'message', repr(exc)))  # Print the exception message if available

            # Enter post-mortem debug mode
            pdb.post_mortem(exc_traceback)  # Pass the traceback to post_mortem
            # exit after debugging
            sys.exit(1)
        finally:
            print("**** DEBUGGING ENDED ****")
def sh2option(filepath, parser, quiet=False):
    with open(filepath, 'r') as file:
        data = file.read().replace('\n', ' ')
    args = [string.lstrip().rstrip() for string in bashlex.split(data)][1:]
    args = [string for string in args if string != '']
    previous_argv = sys.argv
    sys.argv = args
    opt = parser.parse(quiet=quiet)
    sys.argv = previous_argv
    return opt


def isIterable(object):
    return isinstance(object, Iterable)


def make_funcdict(d=None, **kwargs):
    def funcdict(d=None, **kwargs):
        if d is not None:
            funcdict.__dict__.update(d)
        funcdict.__dict__.update(kwargs)
        return funcdict.__dict__
    funcdict(d, **kwargs)
    return funcdict


def pickleCopy(obj):
    import io
    import pickle
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    obj2 = pickle.load(buf)
    return obj2


def makeArchive(folder, outpath, format='zip'):
    folder = os.path.abspath(folder)
    folder_name = os.path.basename(folder)
    parent = os.path.dirname(folder)
    #print(folder, folder_name, parent)
    out_file = outpath + '.%s' % format
    if os.path.exists(out_file):
        os.remove(out_file)
    shutil.make_archive(base_name=outpath, format=format,
                        root_dir=parent,
                        base_dir=folder_name)


def yamldump(opt, target):
    with open(target, 'w') as file:
        documents = yaml.dump(opt, file)
# H5 dataset


class H5DataDict():
    def __init__(self, path, cache_max=10000000):
        self.path = path
        f = H5File(path)
        self.fkeys = f.fkeys
        self.dict = dict([(key, H5Var(path, key)) for key in self.fkeys])
        self.cache = dict([(key, {}) for key in self.fkeys])
        self.cache_counter = 0
        self.cache_max = cache_max
        f.close()

    def keys(self):
        return self.fkeys

    def __getitem__(self, values):
        if type(values) is not tuple:
            if values in self.fkeys:
                return self.dict[values]
            else:
                raise ValueError('%s does not exist' % values)
        if values[0] in self.fkeys:
            if values[1] not in self.cache[values[0]]:
                data = self.dict[values[0]][values[1]]
                if self.cache_counter < self.cache_max:
                    self.cache[values[0]][values[1]] = data
            else:
                data = self.cache[values[0]][values[1]]
            return data
        else:
            raise ValueError('%s does not exist' % values)


class H5Var():
    def __init__(self, path, datasetName):
        self.path, self.dname = path, datasetName

    def __getitem__(self, index):
        f = H5File(self.path)
        if index is None:
            data = f[(self.dname,)]
        else:
            data = f[self.dname, index]
        f.close()
        return data

    def __len__(self):
        f = H5File(self.path)
        leng = f.getLen(self.dname)
        f.close()
        return leng

    @property
    def shape(self):
        return len(self)

    def append(self, array):
        f = H5File(self.path, mode='a')
        if self.dname not in f.f.keys():
            if np.issubdtype(array[0].dtype, np.integer):
                dtype = 'i8'  # 'i' means 'i4' which only support number < 2147483648
            elif np.issubdtype(array[0].dtype, np.float):
                dtype = 'f8'
            else:
                raise ValueError('Unsupported dtype %s' % array.dtype)
            f.f.create_dataset(self.dname, (0,), dtype=dtype,
                               maxshape=(None,), chunks=(102400,))
            f.f.create_dataset('%s_serial_dataBias' % self.dname,
                               (0,), dtype='i', maxshape=(None,), chunks=(102400,))
            f.f.create_dataset('%s_serial_shape' % self.dname,
                               (0,), dtype='i', maxshape=(None,), chunks=(102400,))
            f.f.create_dataset('%s_serial_shapeBias' % self.dname,
                               (0,), dtype='i', maxshape=(None,), chunks=(102400,))
        if "%s_serial_dataBias" % self.dname not in f.f.keys():
            raise ('Appending for Non-serialized form is not implemented')
        #f.append(self.dname, array)
        serialData, dataBias, serialShape, shapeBias = serializeArray(array)
        key = self.dname
        dataTuple = [serialData, serialShape]
        for i, key in enumerate([self.dname, '%s_serial_shape' % self.dname]):
            oshape = f.f[key].shape[0]
            f.f[key].resize((oshape+dataTuple[i].shape[0],))
            f.f[key][oshape:oshape+dataTuple[i].shape[0]
                     ] = (dataTuple[i] if 'Bias' not in key else dataTuple[i]+f.f[key][-1])
        dataTuple = [dataBias, shapeBias]
        for i, key in enumerate(['%s_serial_dataBias' % self.dname, '%s_serial_shapeBias' % self.dname]):
            oshape = f.f[key].shape[0]
            if oshape == 0:
                f.f[key].resize((dataTuple[i].shape[0],))
                f.f[key][:] = dataTuple[i]
            else:
                tshape = oshape+dataTuple[i].shape[0]-1
                f.f[key].resize((oshape+dataTuple[i].shape[0]-1,))
                f.f[key][oshape:tshape] = dataTuple[i][1:]+f.f[key][oshape-1]


def get_h5row(path, datasetName, index):
    f = H5File(path)
    data = f[datasetName, index]
    f.close()
    return data


class H5File():
    def __init__(self, path, mode='r'):
        self.f = h5py.File(path, mode)
        self.fkeys = list(self.f.keys())

    def keys(self):
        return self.fkeys

    def get_serial_data(self, key, index):
        f = self.f
        serial_data = f[key]
        shapeBias = f['%s_serial_shapeBias' % key]
        dataBias = f['%s_serial_dataBias' % key]
        serial_shape = f['%s_serial_shape' % key]
        shape = np.array(serial_shape[shapeBias[index]:shapeBias[index+1]])

        data = np.array(serial_data[dataBias[index]
                        :dataBias[index+1]]).reshape(shape)
        return data

    def __getitem__(self, value):
        f, fkeys = self.f, self.fkeys
        key = value[0]
        if "%s_serial_dataBias" % key in fkeys:
            if len(value) == 1:
                serialData, dataBias, serialShape, shapeBias = np.array(f[key]), np.array(
                    f['%s_serial_dataBias' % key]), np.array(f['%s_serial_shape' % key]), np.array(f['%s_serial_shapeBias' % key])
                item = serialized2array(
                    serialData, dataBias, serialShape, shapeBias)
            else:
                if isIterable(value[1]):
                    item = np.array([self.get_serial_data(key, ind)
                                    for ind in value[1]])
                else:
                    item = self.get_serial_data(key, value[1])
        elif "%s_shapes" % key in fkeys:
            item = padded2array(f[key][value[1]], f["%s_shapes" % key])
        else:
            if len(value) == 1:
                item = np.array(f[key])
            else:
                if isIterable(value[1]):
                    ind = np.array(value[1])
                    uind, inverse = np.unique(ind, return_inverse=True)
                    sindi = np.argsort(uind)
                    sind = uind[sindi]
                    item = np.array(f[key][list(sind)])
                    item = item[sindi]
                    item = item[inverse]
                else:
                    item = np.array(f[key][value[1]])
        # print(type(item),item.shape)
        return item

    def getLen(self, key):
        if "%s_serial_dataBias" % key in self.fkeys:
            leng = self.f["%s_serial_dataBias" % key].shape[0] - 1
        else:
            leng = self.f[key].shape[0]
        return leng

    def append(self, dname, array):
        pass

    def close(self):
        self.f.close()


def readh5(path):
    dataDict = {}
    with h5py.File(path, 'r') as f:
        fkeys = f.keys()
        for key in fkeys:
            if "_serial" in key:
                continue
            if "_shapes" in key:
                continue
            # if np.array(f[key]).dtype.type is np.bytes_: # if is string (strings are stored as bytes in h5py)
            #     print(f[key])
            #     xs=np.array(f[key])
            #     print(xs, xs.dtype, xs.dtype.type)
            #     dataDict[key] = np.char.decode(np.array(f[key]), encoding='UTF-8')
            #     continue

            if "%s_serial_dataBias" % key in fkeys:
                serialData, dataBias, serialShape, shapeBias = np.array(f[key]), np.array(
                    f['%s_serial_dataBias' % key]), np.array(f['%s_serial_shape' % key]), np.array(f['%s_serial_shapeBias' % key])
                dataDict[key] = serialized2array(
                    serialData, dataBias, serialShape, shapeBias)
            elif "%s_shapes" % key in fkeys:
                dataDict[key] = padded2array(f[key], f["%s_shapes" % key])
            else:
                dataDict[key] = np.array(f[key])
    return dataDict


def writeh5(path, dataDict, mode='w', compactForm='serial', quiet=False):
    with h5py.File(path, mode) as f:
        fkeys = f.keys()
        for key in dataDict.keys():
            if key in fkeys:  # overwrite if dataset exists
                del f[key]
            else:
                if dataDict[key].dtype is np.dtype('O'):
                    if compactForm == 'padding':
                        padded, shapes = padArray(dataDict[key])
                        f[key] = padded
                        f['%s_shapes' % key] = shapes
                    elif compactForm == 'serial':
                        serialData, dataBias, serialShape, shapeBias = serializeArray(
                            dataDict[key])
                        f[key] = serialData
                        f['%s_serial_dataBias' % key] = dataBias
                        f['%s_serial_shape' % key] = serialShape
                        f['%s_serial_shapeBias' % key] = shapeBias

                elif dataDict[key].dtype.type is np.str_:
                    f[key] = np.char.encode(dataDict[key], 'UTF-8')
                else:
                    f[key] = dataDict[key]
    if quiet == False:
        print(path, 'is successfully written.')
    return dataDict


def readply(path, scaleFactor=1/256.):
    from plyfile import PlyData, PlyElement

    try:
        # print(path)
        with open(path, 'rb') as f:
            plydata = PlyData.read(f)
        vert = np.array(
            [plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        face = np.array([plydata['face']['vertex_index'][i] for i in range(
            len(plydata['face']['vertex_index']))]).astype(int)
        #vert = np.zeros((2,3))
    except:
        print('read error', path)
        return np.zeros((10, 3)), np.zeros((10, 3)).astype(int), False
    # return vert, face
    return vert*scaleFactor, face, True


def plys2h5(plyDir, h5path, indices=None, scaleFactor=1/256.):
    plynames, plypaths = listdir(plyDir, return_path=True)
    plypaths = np.array(plypaths)
    if indices is None:
        indices = np.arange(len(plynames))
    print('Total shapes: %d, selected shapes: %d' %
          (len(plynames), indices.shape[0]))
    verts, faces = [], []
    args = [(plypath,) for plypath in plypaths[indices]]
    #verts, faces = parallelMap(readply, args, zippedOut=True)
    def func(path): return readply(path, scaleFactor=scaleFactor)
    verts, faces, valids = parallelMap(readply, args, zippedOut=False)
    inv_ind = np.where(valids == False)[0]
    print('inv_ind', inv_ind)
    np.savetxt('inv_ind.txt', inv_ind)
    writeh5(h5path, dataDict={'vert': np.array(verts), 'face': np.array(
        faces), 'index': np.array(indices)}, compactForm='serial')
    return inv_ind


class Obj():
    # Just an empty class. Used for conveniently assign member variables
    def __init__(self, dataDict={}, **kwargs):
        self.update(dataDict)
        self.update(kwargs)

    def update(self, dataDict):
        self.__dict__.update(dataDict)
        return self


def unit_test():
    a = {'a': {'b': 3, 'c': 4}, 'd': 45,           'df': 0,
         'sx': {'1': {'2': {'deep': 'learning'}}}}
    b = {'a': {'b': 45},      'd': {'b': 3, 'c': 4},
         'sx': {'1': {'2': {'is': 'NB'}}}}
    checker = sysutil.dictUpdate(a, b) == {'a': {'b': 45, 'c': 4},
                                           'd': {'b': 3, 'c': 4},
                                           'df': 0,
                                           'sx': {'1': {'2': {'deep': 'learning', 'is': 'NB'}}}}
    assert (checker)


if __name__ == '__main__':
    unit_test()
