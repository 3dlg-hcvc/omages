"""This module contains useful methods for numpy and math.

The methods are gathered in these groups:
    Math utilities
    Array operations
    Linear algebra
    H5 dataset utilities
"""
from __future__ import print_function
import re
import os
import sys
import shutil
import subprocess
import contextlib

from collections.abc import Iterable

import h5py
import numpy as np
#import torch
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.special import log_softmax
import scipy

from collections.abc import Iterable
from time import time, sleep
from tqdm import tqdm
from xgutils import sysutil

EPS = 0.000000001
# randomization utils
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
# math utils
def lemniscate(t=None, start=0.5, end=2.5, num=256,
                a=1, x_scale=1., y_scale=1.):
    if t is None:
        t = np.linspace( np.pi*start, np.pi*end, 
                num, endpoint=False)
    x = a*np.sqrt(2)*np.cos(t) / (1+np.sin(t)**2)
    y = a*np.sqrt(2)*np.cos(t)*np.sin(t) / (1+np.sin(t)**2)
    return x*x_scale, y*y_scale
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x, axis=None):
    """ minus x.max to prevent over/under flow
     see https://stackoverflow.com/questions/42599498/numercially-stable-softmax/42606665#42606665
     """
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
def logsoftmax(x, **kwargs):
    return log_softmax(x, **kwargs)
def logistic(x,x0=.5,L=1,k=20):
    """
    Generalised logistic function https://en.wikipedia.org/wiki/Generalised_logistic_function
    """
    return L/(1+np.exp(-k*(x-x0)))
def s_shape_curve_0to1(x, beta=3.):
    """
        
    """
    xeps = x + EPS
    return 1 / (1+np.power(xeps/(1-xeps), -beta))
def normalize(x):
    return x/np.sqrt((x*x).sum())
def length(x):
    return np.linalg.norm(x)
def ceiling_division(n, d):
    """
    Ceiling equivalent of // operator (floor division) in Python
    Reminiscent of the Penn & Teller levitation trick, this "turns the world upside down (with negation), uses plain floor division (where the ceiling and floor have been swapped), and then turns the world right-side up (with negation again)"
    """
    return -(n // -d)

def point2lineDistance(q, p1, p2):
    d = np.linalg.norm(np.cross(p2-p1, p1-q))/np.linalg.norm(p2-p1)
    return d
def get2DRotMat(theta=90, mode='degree'):
    if mode == 'degree':
        theta = np.radians(theta)
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
def pointSegDistance(q, p1, p2):
    line_vec = p2-p1
    pnt_vec = q-p1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = normalize(line_vec)
    pnt_vec_scaled = pnt_vec * 1.0/line_len
    t = np.dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec * t
    dist = length(nearest - pnt_vec)
    nearest = nearest + p1
    return (dist, nearest)
def rescale(vertex):
    shifted = vertex-vertex.mean(-2)[..., np.newaxis, :]
    scope  = np.abs(shifted).max((-1,-2))[..., np.newaxis, np.newaxis]
    vertex = shifted / scope * 40 + np.array([50.,50.])
    print('checking scope: ' , scope)
    print('checking vertex: ', vertex.max(-2))
    return vertex
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
def sampleTriangle(vertices, sampleNum=10, noVert=False):
    # vertices: numpy array of 
    if noVert == False:
        rd_a, rd_b = np.random.rand(sampleNum-3), np.random.rand(sampleNum-3)
    else:
        rd_a, rd_b = np.random.rand(sampleNum), np.random.rand(sampleNum)
    larger_than_1 = (rd_a + rd_b > 1.)
    rd_a[larger_than_1] = 1 - rd_a[larger_than_1]
    rd_b[larger_than_1] = 1 - rd_b[larger_than_1]
    if noVert == False:
        rd_a = np.r_[0,1,0,rd_a]
        rd_b = np.r_[0,0,1,rd_b]
    samples = np.array([vertices[0] + rd_a[i]*(vertices[1]-vertices[0]) + rd_b[i]*(vertices[2]-vertices[0]) \
                            for i in range(sampleNum)])
    return samples
def randQuat(N=1):
    #Generates uniform random quaternions
    #James J. Kuffner 2004 
    #A random array 3xN
    s = np.random.rand(3,N)
    sigma1 = np.sqrt(1.0 - s[0])
    sigma2 = np.sqrt(s[0])
    theta1 = 2*np.pi*s[1]
    theta2 = 2*np.pi*s[2]
    w = np.cos(theta2)*sigma2
    x = np.sin(theta1)*sigma1
    y = np.cos(theta1)*sigma1
    z = np.sin(theta2)*sigma2
    return np.array([w, x, y, z])
def multQuat(Q1,Q2):
    # https://stackoverflow.com/a/38982314/5079705
    w0,x0,y0,z0 = Q1   # unpack
    w1,x1,y1,z1 = Q2
    return([-x1*x0 - y1*y0 - z1*z0 + w1*w0, x1*w0 + y1*z0 - z1*y0 +
    w1*x0, -x1*z0 + y1*w0 + z1*x0 + w1*y0, x1*y0 - y1*x0 + z1*w0 +
    w1*z0])
def conjugateQuat(Q):
    return np.array([Q[0],-Q[1],-Q[2],-Q[3]])
def applyQuat(V, Q):
    P = np.array([0., V[0], V[1], V[2]])
    nP = multQuat(Q, multQuat(P, conjugateQuat(Q)) )
    return nP[1:4]
def fibonacci_sphere(samples=1000):
    rnd = 1.

    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - np.power(y,2))

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        points.append([x,y,z])

    return points

def setDiffND(array1, array2, precision=1):
    array1 = np.round(array1, decimals=precision)
    array2 = np.round(array2, decimals=precision)
    print(array1, array2)
    a1_rows = array1.view([('', array1.dtype)] * array1.shape[1])
    a2_rows = array2.view([('', array2.dtype)] * array2.shape[1])
    diff = np.setdiff1d(a1_rows, a2_rows).view(array1.dtype).reshape(-1, array2.shape[1])
    return diff
def split_by_category(array:np.ndarray, category:Iterable, order='occurrence') ->np.ndarray:
    """group list of objects by category

    Args:
        array (np.ndarray): [description]
        category (Iterable): [description]
        order (str): 'occurrence' or 'default'

    Returns:
        np.ndarray: list of groups in the order of category item.
    """
    splited = []
    unique, index = np.unique(category, return_index=True)
    if order == 'occurrence':
        unique = category[np.sort(index)]
    for cate in unique:
        splited.append(array[category==cate])
    return np.array(splited)
    #return torch.stack(splited)
def array2chunks(array, chunk_size=2):
    oshape = array.shape
    chunks = array[:oshape[0]//chunk_size*chunk_size]
    if oshape[0]//chunk_size*chunk_size != oshape[0]:
        print("Warning! some data will be dropped during array2chunks (len mod chunk_size!=0)")
    chunks = chunks.reshape( oshape[0]//chunk_size, chunk_size ,*oshape[1:] )
    return chunks
# legacy from CycleGAN
def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)
def print_numpy(x, title="", val=True, printAll=False, shp=True):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    if title:
        print(title)
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape, x.dtype)
    if printAll:
        print(x)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))
# torch tools
def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

## python utils
def dictAdd(dict1, dict2):
    for key in dict2:
        if key in dict1:
            dict1[key]+= dict2[key]
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
def prefixDictKey(dic, prefix=''):
    return dict([(prefix+key, dic[key]) for key in dic])
pj=os.path.join
def strNumericalKey(s):
    if s:
        try:
            c = re.findall('\d+', s)[0]
        except:
            c = -1
        return int(c)
# List dir and sort by numericals (if applicable)
def listdir(directory, return_path=True):
    filenames = os.listdir(directory)
    filenames.sort(key=strNumericalKey)
    if return_path==True:
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
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
def runCommand(command):
    import bashlex
    cmd = list(bashlex.split(command))
    p = subprocess.run(cmd, capture_output=True) # capture_output=True -> capture stdout&stderr
    stdout = p.stdout.decode('utf-8')
    stderr = p.stderr.decode('utf-8')
    return stdout, stderr, p.returncode
def array2batches(array, chunkSize=4):
    return [array[i*chunkSize:(i+1)*chunkSize] for i in range( (len(array)+chunkSize-1)//chunkSize )]
def progbar(array, total=None):
    if total is None:
        if isinstance(array, enumerate):
            array = list(array)
        total = len(array)
    return tqdm(array,total=total, ascii=True)
def parallelMap(func, args, batchFunc=None, zippedIn=True, zippedOut=False, cores=-1, quiet=False):
    from pathos.multiprocessing import ProcessingPool
    """Parallel map using multiprocessing library Pathos

    Args:
        stderr (function): func
        args (arguments): [arg1s, arg2s ,..., argns](zippedIn==True) or [[arg1,arg2,...,argn], ...](zippedIn=False)
        batchFunc (func, optional): TODO. Defaults to None.
        zippedIn (bool, optional): See [args]. Defaults to True.
        zippedOut (bool, optional): See [Returns]. Defaults to False.
        cores (int, optional): How many processes. Defaults to -1.
        quiet (bool, optional): if do not print anything. Defaults to False.

    Returns:
        tuples: [out1s, out2s,..., outns](zippedOut==False) or [[out1,out2,...,outn], ...](zippedOut==True)
    """    
    if batchFunc is None:
        batchFunc = lambda x:x
    if zippedIn==True:
        args = list(map(list, zip(*args))) # transpose
    if cores==-1:
        cores = os.cpu_count()
    pool = ProcessingPool(nodes=cores)
    batchIdx = list(range(len(args[0])))
    batches = array2batches(batchIdx, cores)
    out = []
    iterations = enumerate(batches) if quiet==True else progbar(enumerate(batches))
    for i,batch in iterations:
        batch_args = [[arg[i] for i in batch] for arg in args]
        out.extend( pool.map(func, *batch_args) )
    if zippedOut == False:
        if type(out[0]) is not tuple:
            out=[(item,) for item in out]
        out = list(map(list, zip(*out)))
    return out


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
    out_file = outpath + '.%s'%format
    if os.path.exists(out_file):
        os.remove(out_file)
    shutil.make_archive(base_name= outpath,format=format, \
                        root_dir = parent, \
                        base_dir = folder_name)

# linear algebra & np array
def rdchoice(array, num, replace=False):
    choice = sorted(np.random.choice(array.shape[0], num, replace=replace))
    return array[choice]
def arraySupersample(array, weight = None, multiplier=10):
    sup = []
    if weight is None:
        weight = np.linspace(0.,1.,multiplier)
    interp = np.repeat(array[:-1], multiplier ,axis=0)
    interpN= np.repeat(array[1:], multiplier ,axis=0)
    weight = np.tile(weight, array.shape[0]-1 )
    weight = weight.reshape( weight.shape + (1,)*len(array.shape[1:]))
    interped = interp*(1-weight) + interpN*weight
    return interped
        
def pointInterp(queryP, points, values):
    """Interpolate on *queryP* based on the *values* of *points*

    Args:
        queryP (MxC1 np array): position to be interpolated
        points (NxC1 np array): base points
        values (NxC2 np array): value of base points

    Returns:
        MxC2 np array: value of query points
    """    
    lengths = np.zeros(len(points))
    weights = np.zeros(len(points))
    interp = 0.
    for i,p in enumerate(points):
        lengths[i] =  np.linalg.norm(p-queryP)
        #print(i,lengths)
        if lengths[i]<EPS:
            return values[i]
        weights[i] =  1/lengths[i]
        interp += weights[i] * values[i]
    interp /= weights.sum()
    return interp

def gridInterp(queries, grid, gridRange=np.array([[0.,1.],[0.,1.],[0.,1.]]), return_func=False):
    axes = [np.arange(gridRange[i,0],gridRange[i,1], 1./grid.shape[i]) for i in range(gridRange.shape[0])]
    func = RegularGridInterpolator(axes, grid)
    interp = func(queries)
    if return_func:
        return interp, func
    return interp

def points2grid(points, values, grid, method="linear", nan=0.0):
    """Interpolate a D-dim point set into a D-dim grid

    Args:
        points (np.ndarray): NxD
        values (np.ndarray): N
        grid (np.ndarray): XxYxZxD grid
        method (str, optional): interplation method (nearest, linear, bicubic). Defaults to "nearest".

    Returns:
        np.ndarray: XxYxZ
    """
    #split_points_coords = np.split(points, points.shape[-1], axis=-1)
    #split_points_coords = [splited[...,0] for splited in split_points_coords]
    split_grid_coords   = np.split(grid, grid.shape[-1], axis=-1)
    split_grid_coords = [splited[...,0] for splited in split_grid_coords]
    grid_values = scipy.interpolate.griddata(points, values, tuple(split_grid_coords), method=method)
    return np.nan_to_num( grid_values, nan=nan ) # replace nan (due to extrapolation) with number
def points2grid_unittest():
    def func(points):
        x, y = points[:,0], points[:,1]
        return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
    points = np.random.rand(3, 2)*2-1
    values = func(points)
    grid = nputil.makeGrid(bb_min=(-1,)*2, bb_max=(1,)*2, shape=(64,)*2, flatten=False)
    grid_value = nputil.points2grid(points, values, grid, method="linear")
    plt.imshow(grid_value, extent=(-1,1,-1,1), origin='lower')
    plt.show()
    grid_value


def cart2image(cart_grid):
    #assert len(cart_grid.shape)==2, f"shape should not be {cart_grid.shape}"
    image_grid = np.flip(np.swapaxes(cart_grid, 0, 1), axis=0)
    return image_grid
def testxxf():
    pass
def array2mesh(array, thresh=0., dim=3, coords=None, sigma=None):
    import mcubes
    from scipy.ndimage import gaussian_filter
    grid = array2NDCube(array, N=dim)


    verts, faces = mcubes.marching_cubes(grid, thresh)
    verts = verts[:,[1,0,2]]/(grid.shape[0]-1) # rearrange order and rescale
    if coords is not None:
        bb_min, bb_max = arrayBBox(coords)
        verts = verts*(bb_max-bb_min) + bb_min
    return verts, faces.astype(int)
def sampleMesh(vert, face, sampleN):
    resample = True
    sampled = None
    while resample:
        try:
            B,FI    = igl.random_points_on_mesh(sampleN, vert, face)
            sampled =   B[:,0:1]*vert[face[FI,0]] + \
                        B[:,1:2]*vert[face[FI,1]] + \
                        B[:,2:3]*vert[face[FI,2]]
            resample=False
            if sampled.shape[0] != sampleN:
                print('Failed to sample "sampleN" points, now resampling...', file=sys.__stdout__)
                resample=True
        except Exception as e:
            print('Error encountered during mesh sampling:', e, file=sys.__stdout__)
            print('Now resampling...', file=sys.__stdout__)
            resample = True
    return sampled
def arrayBBox(array):
    bb_min, bb_max = [], []
    for i in range(array.shape[-1]):
        bb_min.append( array[..., i].min() )
        bb_max.append( array[..., i].max() )
    return np.array([bb_min, bb_max])

def makeGrid(bb_min=[0,0,0], bb_max=[1,1,1], shape=[10,10,10], 
            mode='on', flatten=True, indexing="ij"):
    """ Make a grid of coordinates

    Args:
        bb_min (list or np.array): least coordinate for each dimension
        bb_max (list or np.array): maximum coordinate for each dimension
        shape (list or int): list of coordinate number along each dimension. If it is an int, the number
                            same for all dimensions
        mode (str, optional): 'on' to have vertices lying on the boundary and 
                              'in' for keeping vertices and its cell inside of the boundary
                              same as align_corners=True and align_corners=False
        flatten (bool, optional): Return as list of points or as a grid. Defaults to True.
        indexing (["ij" or "xy"]): default to "xy", see https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html

    Returns:
        np.array: return coordinates (XxYxZxD if flatten==False, X*Y*ZxD if flatten==True.
    """    
    coords=[]
    bb_min = np.array(bb_min)
    bb_max = np.array(bb_max)
    if type(shape) is int:
        shape = np.array([shape]*bb_min.shape[0])
    for i,si in enumerate(shape):
        if mode=='on':
            coord = np.linspace(bb_min[i], bb_max[i], si)
        elif mode=='in':
            offset = (bb_max[i]-bb_min[i])/2./si
            # 2*si*w=bmax-bmin
            # w = (bmax-bmin)/2/si
            # start, end = bmax+w, bmin-w
            coord = np.linspace(bb_min[i]+offset,bb_max[i]-offset, si)
        coords.append( coord )
    grid = np.stack(np.meshgrid(*coords,sparse=False, indexing=indexing), axis=-1)
    if flatten==True:
        grid = grid.reshape(-1,grid.shape[-1])
    return grid
def array2NDCube(array, N=3, feature_dim=0): 
    vox_dim = np.ceil( np.power(array.shape[0], 1./N) ).astype(int)
    if feature_dim==0:
        return array.reshape((vox_dim,)*N, )
    else:
        return array.reshape((vox_dim,)*N + (array.shape[-1],))
def gridPoints2levelPlanes(points, level_axis=0, decimals=8):
    points = points.round(decimals=decimals)
    uniqeFs = np.unique( points[..., level_axis] )
    levels = []
    for uniqueF in uniqeFs:
        levels.append(points[points[..., level_axis] == uniqueF])
    levels = np.array(levels)
    levels = np.delete(levels, level_axis, axis=-1)
    print([level.shape for level in levels])
    return levels
def inverseWhere(where, shape):
    mask = np.zeros(shape, dtype='bool')
    mask[where]=True
    return mask
def subsampleArray(array, sampleN, axis=0):
    if array.shape[axis] < sampleN:
        return array
    choice = np.random.choice(array.shape[axis], sampleN, replace=False)
    return np.take(array, choice, axis=axis)
def subsampleBoolArray(array, sampleN, ratio=None):
    if ratio is None:
        pos = np.where(array)[0]
        if pos.shape[0] > sampleN:
            choice = np.random.choice(pos.shape[0], sampleN, replace=False)
            pos = pos[choice]
            subArray = inverseWhere(pos, array.shape[0])
        else:
            subArray = array
    else:
        subArray = array
    return subArray
def array2slices(array):
    if array.dtype is not np.dtype(int):
        raise ValueError("Array's type is not int")
    return tuple([slice(0,si) if si>0 else 0 for si in array])
def padArray(objArray):
    if (type(objArray) is np.ndarray) and (not all( (type(obj) is np.ndarray) for obj in objArray)):
        raise ValueError('Invalid objArray! Not every array element is np.ndarray')
    dtype = objArray[0].dtype
    #shapes = np.array([array.shape for array in objArray])
    maxShapeL = np.array([len(array.shape) for array in objArray]).max()
    paddedShapes, slices = [], []
    for array in objArray:
        shape = np.zeros(maxShapeL).astype(int)
        shape[0:len(array.shape)] = np.array(array.shape)
        paddedShapes.append(shape)
        slices.append( array2slices(shape) )
    paddedShapes = np.array(paddedShapes)
    targetShape = paddedShapes.max(axis=0).astype(int)
    targetArrays = np.zeros((len(objArray), *tuple(targetShape)), dtype=dtype)
    for i, array in enumerate(objArray):
        targetArrays[i][slices[i]] = array
    return targetArrays, paddedShapes
def padAsShape(array, targetShape):
    targetSA = np.zeros(targetShape)
    combined = np.empty(2,dtype='O')
    combined[0] = targetSA
    combined[1] = array
    padded, _ = padArray(combined)
    padded = padded[1]
    return padded
def padAs(array, targetArray):
    return padAsShape(array, targetArray.shape)
def padded2array(padded, shapes, single=False):
    if single==True:
        padded = np.array([padded])
        shapes = np.array([shapes])
    targetArrays = []
    for i in range(padded.shape[0]):
        slices = array2slices(shapes[i])
        targetArrays.append( padded[i][slices] )
    if single==True:
        return np.array(targetArrays)[0]
    return np.array(targetArrays)
def unpadding(padded, token_dims=1):
    pshape = padded.shape
    none_end_dims = (len(pshape)-token_dims)
    end_token = padded[ (slice(-1,None),) * none_end_dims ]
    isnot_end  = (padded!=end_token).all(axis=-1)
    unpadded = []
    for i in range(pshape[0]):
        unpadded.append( padded[i][ isnot_end[i] ] )
    return unpadded

def value_to_occurrence(array):
    """ assuming array is 1d array """
    assert len(array.shape) == 1
    unique_elements = np.unique(array)
    # Find the indices where A matches the unique elements
    matches = (array[:, np.newaxis] == unique_elements).cumsum(axis=0)
    result = matches[np.arange(len(array)), array]-1
    return result

def serializeArray(objArray, return_dict=False):
    if (type(objArray) is np.ndarray) and (not all( (type(obj) is np.ndarray) for obj in objArray)):
        raise ValueError('Invalid objArray! Not every array element is np.ndarray')
    serialData, serialShape, dataBias, shapeBias=[], [], [0], [0]
    for array in objArray:
        #shapes.append(array.shape)
        shape=np.array(array.shape)
        serialData.append(array.reshape(-1))
        serialShape.append(shape.reshape(-1))
        shapeBias.append( shapeBias[-1] + serialShape[-1].shape[0])
        dataBias.append( dataBias[-1] + serialData[-1].shape[0])
    serialData, serialShape = np.concatenate(serialData), np.concatenate(serialShape)
    dataBias, shapeBias     = np.array(dataBias), np.array(shapeBias)
    if return_dict:
        return {'serialData':serialData, 'serialShape':serialShape, 'dataBias':dataBias, 'shapeBias':shapeBias}
    else:
        return serialData, dataBias, serialShape, shapeBias
def serialized2array(serialData, dataBias, serialShape, shapeBias):
    targetArrays=[]
    for i in progbar(range(shapeBias.shape[0]-1)):
        shape       = serialShape[shapeBias[i]:shapeBias[i+1]]
        data_serial = serialData[dataBias[i]:dataBias[i+1]]
        array       = data_serial.reshape(shape)
        targetArrays.append(array)
    return np.array(targetArrays)
def deserializeArray(serialData, dataBias, serialShape, shapeBias, quiet=False):
    targetArrays=[]
    if quiet==False:
        its = progbar(range(shapeBias.shape[0]-1))
    else:
        its = range(shapeBias.shape[0]-1)
    for i in its:
        shape       = serialShape[shapeBias[i]:shapeBias[i+1]]
        data_serial = serialData[dataBias[i]:dataBias[i+1]]
        array       = data_serial.reshape(shape)
        targetArrays.append(array)
    return targetArrays
serializeArrays, deserializeArrays = serializeArray, deserializeArray

# H5 dataset
class H5DataDict():
    def __init__(self, path, cache_max = 10000000):
        self.path = path
        f = H5File(path)
        self.fkeys = f.fkeys
        self.dict = dict([(key, H5Var(path, key)) for key in self.fkeys])
        self.cache= dict([(key, {}) for key in self.fkeys])
        self.cache_counter=0
        self.cache_max = cache_max
        f.close()
    def keys(self):
        return self.fkeys
    def __getitem__(self,values):
        if type(values) is not tuple:
            if values in self.fkeys:
                return self.dict[values]
            else:
                raise ValueError('%s does not exist'%values)
        if values[0] in self.fkeys:
            if values[1] not in self.cache[values[0]]:
                data = self.dict[values[0]][values[1]]
                if self.cache_counter < self.cache_max:
                    self.cache[values[0]][values[1]] = data
            else:
                data = self.cache[values[0]][values[1]]
            return data
        else:
            raise ValueError('%s does not exist'%values)
class H5Var():
    def __init__(self, path, datasetName):
        self.path, self.dname=path, datasetName
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
                dtype = 'i8' # 'i' means 'i4' which only support number < 2147483648
            elif np.issubdtype(array[0].dtype, np.float):
                dtype = 'f8'
            else:
                raise ValueError('Unsupported dtype %s'%array.dtype)
            f.f.create_dataset(self.dname, (0,), dtype=dtype, maxshape=(None,), chunks=(102400,))
            f.f.create_dataset('%s_serial_dataBias'%self.dname, (0,), dtype='i', maxshape=(None,), chunks=(102400,))
            f.f.create_dataset('%s_serial_shape'%self.dname, (0,), dtype='i', maxshape=(None,), chunks=(102400,))
            f.f.create_dataset('%s_serial_shapeBias'%self.dname, (0,), dtype='i', maxshape=(None,), chunks=(102400,))
        if "%s_serial_dataBias"%self.dname not in f.f.keys():
            raise('Appending for Non-serialized form is not implemented')
        #f.append(self.dname, array)
        serialData, dataBias, serialShape, shapeBias = serializeArray(array)
        key = self.dname
        dataTuple = [serialData, serialShape]
        for i,key in enumerate([self.dname, '%s_serial_shape'%self.dname]):
            oshape = f.f[key].shape[0]
            f.f[key].resize((oshape+dataTuple[i].shape[0],))
            f.f[key][oshape:oshape+dataTuple[i].shape[0]] = (dataTuple[i] if 'Bias' not in key else dataTuple[i]+f.f[key][-1])
        dataTuple = [dataBias, shapeBias]
        for i,key in enumerate(['%s_serial_dataBias'%self.dname, '%s_serial_shapeBias'%self.dname]):
            oshape = f.f[key].shape[0]
            if oshape ==0:
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
        shapeBias = f['%s_serial_shapeBias'%key]
        dataBias = f['%s_serial_dataBias'%key]
        serial_shape = f['%s_serial_shape'%key]
        shape = np.array( serial_shape[shapeBias[index]:shapeBias[index+1]] )

        data = np.array(serial_data[ dataBias[index]:dataBias[index+1]]).reshape(shape)
        return data
    def __getitem__(self, value):
        f, fkeys = self.f, self.fkeys
        key = value[0]
        if "%s_serial_dataBias"%key in fkeys:
            if len(value)==1:
                serialData, dataBias, serialShape, shapeBias = np.array(f[key]), np.array(f['%s_serial_dataBias'%key]), np.array(f['%s_serial_shape'%key]), np.array(f['%s_serial_shapeBias'%key])
                item = serialized2array(serialData, dataBias, serialShape, shapeBias)
            else:
                if isIterable(value[1]):
                    item = np.array([self.get_serial_data(key, ind) for ind in value[1]])
                else:
                    item = self.get_serial_data(key,value[1])
        elif "%s_shapes"%key in fkeys:
            item = padded2array(f[key][value[1]], f["%s_shapes"%key])
        else:
            if len(value)==1:
                item = np.array(f[key])
            else:
                if isIterable(value[1]):
                    ind  = np.array(value[1])
                    uind, inverse = np.unique(ind, return_inverse=True)
                    sindi= np.argsort(uind)
                    sind = uind[sindi]
                    item = np.array(f[key][ list(sind) ])
                    item = item[sindi]
                    item = item[inverse]
                else:
                    item = np.array(f[key][value[1]])
        #print(type(item),item.shape)
        return item
    def getLen(self, key):
        if "%s_serial_dataBias"%key in self.fkeys:
            leng = self.f["%s_serial_dataBias"%key].shape[0] - 1
        else:
            leng = self.f[key].shape[0]
        return leng
    def append(self, dname, array):
        pass

    def close(self):
        self.f.close()
def readh5(path):
    dataDict={}
    with h5py.File(path,'r') as f:
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

            if "%s_serial_dataBias"%key in fkeys:
                serialData, dataBias, serialShape, shapeBias = np.array(f[key]), np.array(f['%s_serial_dataBias'%key]), np.array(f['%s_serial_shape'%key]), np.array(f['%s_serial_shapeBias'%key])
                dataDict[key] = serialized2array(serialData, dataBias, serialShape, shapeBias)
            elif "%s_shapes"%key in fkeys:
                dataDict[key] = padded2array(f[key], f["%s_shapes"%key])
            else:
                dataDict[key] = np.array(f[key])
    return dataDict
def writeh5(path, dataDict, mode='w', compactForm='serial', quiet=False):
    path = os.path.expanduser(path)
    dirname = os.path.dirname( path )
    sysutil.mkdirs(dirname)
    #if os.path.exists(os.path.dir)

    with h5py.File(path, mode) as f:
        fkeys = f.keys()
        for key in dataDict.keys():
            if key in fkeys: # overwrite if dataset exists
                del f[key]
            else:
                if dataDict[key].dtype is np.dtype('O'):
                    if compactForm=='padding':
                        padded, shapes = padArray(dataDict[key])
                        f[key] = padded
                        f['%s_shapes'%key] = shapes
                    elif compactForm=='serial':
                        serialData, dataBias, serialShape, shapeBias = serializeArray(dataDict[key])
                        f[key] = serialData
                        f['%s_serial_dataBias'%key] = dataBias
                        f['%s_serial_shape'%key]    = serialShape
                        f['%s_serial_shapeBias'%key]= shapeBias

                elif dataDict[key].dtype.type is np.str_:
                    f[key] = np.char.encode( dataDict[key], 'UTF-8' )
                else:
                    f[key] = dataDict[key]
    if quiet==False:
        print(path, 'is successfully written.')
    return dataDict

def readply(path, scaleFactor=1/256.):
    from plyfile import PlyData, PlyElement

    try:
        #print(path)
        with open(path, 'rb') as f:
            plydata = PlyData.read(f)
        vert = np.array([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).T
        face = np.array([plydata['face']['vertex_index'][i] for i in range(len(plydata['face']['vertex_index']))]).astype(int)
        #vert = np.zeros((2,3))
    except:
        print('read error', path)
        return np.zeros((10,3)), np.zeros((10,3)).astype(int), False
    #return vert, face
    return vert*scaleFactor, face, True
def plys2h5(plyDir, h5path, indices=None, scaleFactor=1/256.):
    plynames, plypaths = listdir(plyDir, return_path=True)
    plypaths = np.array(plypaths)
    if indices is None:
        indices = np.arange(len(plynames))
    print('Total shapes: %d, selected shapes: %d'%(len(plynames), indices.shape[0]))
    verts, faces = [], []
    args = [(plypath,) for plypath in plypaths[indices]]
    #verts, faces = parallelMap(readply, args, zippedOut=True)
    func = lambda path: readply(path, scaleFactor=scaleFactor)
    verts, faces, valids = parallelMap(readply, args, zippedOut=False)
    inv_ind = np.where(valids==False)[0]
    print('inv_ind', inv_ind)
    np.savetxt('inv_ind.txt', inv_ind)
    writeh5(h5path, dataDict={'vert':np.array(verts), 'face':np.array(faces), 'index':np.array(indices)}, compactForm='serial')
    return inv_ind

# misc
def histogram_equalization(image, number_bins=512):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf
