import os
import sys
import glob
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from xgutils import nputil, sysutil
from einops import rearrange
def sampleNDSphere(shape):
    """generate random sample on n-d sphere

    Args:
        shape ([...,dim] np.array): generate samples as shape, 
            the last dimension dim is the dimension of the sphere

    Returns:
        np.array: samples
    """    
    u = torch.randn(shape)
    d=(u**2).sum(-1,keepdim=True)**.5
    u = u/d
    return u
# cuda related
def print_cuda_stats():
    stats = "Current CUDA usage: %.2f GB, Max usage %.3f GB"%(torch.cuda.memory_allocated()/(1<<30), torch.cuda.max_memory_allocated()/(1<<30))
    print(stats)
    return stats
# linear algebra
def array2NDCube(tensor, N=3): 
    vox_dim = np.ceil( tensor.shape[0]**(1./N) ).astype(int)
    return tensor.reshape(*((vox_dim,)*N))

# Type conversions
def th2np(tensor):
    if type(tensor) is np.ndarray:
        return tensor
    if type(tensor) is torch.Tensor:
        return tensor.detach().cpu().numpy()
    if issubclass(type(tensor), torch.distributions.distribution.Distribution):
        return thdist2np(tensor)
    else:
        return tensor
def np2th(array, device='cuda'):
    tensor = array
    # if numpy array and not type string
    if type(tensor) is np.ndarray and np.issubdtype(tensor.dtype, np.str_)==False:
        tensor = torch.from_numpy(tensor)
    elif type(tensor) is torch.Tensor:
        if device=='cuda':
            return tensor.cuda()
        return tensor.cpu()
    else:
        return array
def nps2ths(arrays, device="cuda"):
    if type(arrays) is dict:
        dic={}
        for key in arrays:
            if type(arrays[key]) is dict or type(arrays[key]) is list or type(arrays[key]) is tuple:
                dic[key] = nps2ths(arrays[key])
            else:
                dic[key] = np2th( arrays[key] )
        return dic
    elif type(arrays) is list or type(arrays) is tuple:
        tensors = []
        for array in arrays:
            if type(array) is dict or type(array) is list:
                tensors.append(nps2ths(array))
            else:
                tensors.append(np2th(array))
        return tuple(tensors)
    else:
        return np2th(arrays)
def ths2nps(tensors):
    if type(tensors) is dict:
        dic={}
        for key in tensors:
            if type(tensors[key]) is dict or type(tensors[key]) is list:
                dic[key] = ths2nps(tensors[key])
            else:
                dic[key] = th2np( tensors[key] )
        return dic
    elif type(tensors) is list or type(tensors) is tuple:
        arrays = []
        for tensor in tensors:
            if type(tensor) is dict or type(tensor) is list:
                arrays.append(ths2nps(tensor))
            else:
                arrays.append(th2np(tensor))
        return tuple(arrays)
    else:
        return th2np(tensors)
def th2device(tensor, device='cpu'): # ['cpu', 'cuda']
    if type(tensor) is torch.Tensor:
        return tensors.detach().cpu() if device=='cpu' else tensors.cuda()
    elif issubclass(type(tensor), torch.distributions.distribution.Distribution):
        if type(tensor) is torch.distributions.independent.Independent:
            reinterpreted_batch_ndims = tensor.reinterpreted_batch_ndims
            dist = th2device(tensor.base_dist)
            return torch.distributions.independent.Independent(dist, reinterpreted_batch_ndims )
        elif type(tensor) is torch.distributions.normal.Normal:
            loc, scale = ths2device([tensor.loc, tensor.scale])
            return torch.distributions.normal.Normal(loc=loc, scale=scale)
    else:
        raise TypeError(f'type {type(tensor)} is not supported')
def ths2device(tensors, device='cpu'):
    if type(tensors) is dict:
        tensors_device={}
        for key in tensors:
            tensor = tensors[key]
            if type(tensor) is dict or type(tensor) is list or type(tensor) is tuple:
                tensors_device[key] = ths2device(tensor)
            else:
                tensors_device[key] = tensor.cuda()
        return tensors_device
    else:
        tensorsCUDA = [tensor.cuda() for tensor in tensors]
        return tuple(tensorsCUDA)
def ths2cuda(tensors):
    if type(tensors) is dict:
        tensorsCUDA={}
        for key in tensors:
            tensor = tensors[key]
            if type(tensor) is dict or type(tensor) is list or type(tensor) is tuple:
                tensorsCUDA[key] = ths2cuda(tensor)
            else:
                tensorsCUDA[key] = tensor.float().cuda()
        return tensorsCUDA
    else:
        tensorsCUDA = [tensor.cuda() for tensor in tensors]
        return tuple(tensorsCUDA)
def ths2cpu(tensors):
    pass
def batch_select(batch, index=0):
    if type(batch) is dict:
        return dict([(key,batch[key][index]) for key in batch.keys()])
    else:
        return batch
def thdist2np(dist):
    item = {'mean': th2np(dist.mean), \
            'variance':th2np(dist.variance), \
            'entropy':th2np(dist.entropy())}
    return item
def simple_gather(tensor, axis, ind):
    pass # TODO
    # ind = torch.tensor(ind,dtype=int).view(..,).expand(-1, -1, dim_x)
    # indY = torch.tensor(ind,dtype=int).unsqueeze(-1).expand(-1, -1, dim_y)
    # subX = X.gather(1, indX)
    # subY = Y.gather(1, indY)
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                matplotlib.lines.Line2D([0], [0], color="b", lw=4),
                matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
def batch_dict(dict_list):
    keys = dict_list[0]
    dataDict = {}
    for key in keys:
        dataDict[key] = []
    for item in dict_list:
        for key in keys:
            dataDict[key].append( item[key] )
    for key in keys:
        dataDict[key] = np.array(dataDict[key])
    return dataDict


# dataset related
def dataset_to_h5(dset, outdir='~/temp/temp.h5'):
    item = dset[0]
    dict_list = []
    for i in sysutil.progbar( range(len(dset)) ):
        dict_list.append(dset[i])
    dataDict = batch_dict(dict_list)
    for key in dataDict.keys():
        if type(dataDict[key]) is torch.Tensor:
            dataDict[key] = th2np(dataDict[key])
    nputil.writeh5(outdir, dataDict)
def dataset_generator(dset, data_indices=[0,1,2], device="cuda"):
    for ind in data_indices:
        dataitem = dset.__getitem__(ind)
        batch = {}
        for key in dataitem:
            datakey = dataitem[key]
            if type(datakey) is not np.ndarray and type(datakey) is not torch.Tensor:
                continue
            datakey = dataitem[key][None,...]
            if type(datakey) is np.ndarray:
                datakey = torch.from_numpy(datakey)
            batch[key] = datakey.to(device)
        yield batch

# class DatasetProcessor():
#     def process(self):
#         shapes = glob.glob( os.path.join(data_root, "*/*.obj") )
#         #shapes = shapes[:40]
#         print("num of shapes", len(shapes))
#         #print(shapes)
#         #for shape_dir in sysutil.progbar(shape_dirs):
#         #    print(shape_dir)
#         #    voxelize_partnet_shape(shape_dir)
        
#         return_codes = sysutil.parallelMap(voxelize_dfaust_shape, [shapes], zippedIn=False)
#         np.save(f"/studio/datasets/DFAUST/voxelization_failure_code.npy",return_codes)
#         print("Percentage of failure:", np.array(return_codes).sum()/len(shapes))
#         print("Return code:", return_codes)



# Fold and Unfold
def unfold_cube(tensor, last_dims=2, size=2, step=2, flatten=True):
    unfolded = tensor
    batch_shape= tensor.shape[:-last_dims]
    batch_dims = len(batch_shape)
    for di in range(last_dims):
        unfolded = unfolded.unfold(batch_dims+di, size=size, step=step)
    if flatten==True:
        total_size = np.array(tensor.shape[-last_dims:]).prod()
        unfold_size = unfolded.shape[-1]**last_dims
        unfolded = unfolded.reshape(*(unfolded.shape[:-2*last_dims]), total_size//unfold_size, unfold_size)
    return unfolded
def fold_cube(unfolded, N=3):
    batch_shape = unfolded.shape[:-2]
    batch_dims  = len(batch_shape)
    vox_dim = np.ceil( np.power(unfolded.shape[-1], 1./N) ).astype(int)
    unfolded = unfolded.reshape(*batch_shape,*((vox_dim,)*(2*N)))
    folded = unfolded
    for i in range(N):
        folded = torch.cat(torch.split(folded,1,dim=batch_dims+i), dim=batch_dims+N+i)
    for i in range(N):
        folded = torch.squeeze(folded, dim=batch_dims)

    return folded

def compress_voxels(voxel, packbits=True):
    assert(voxel.shape[-1]==256), "Only 256-> 16x16 dims is supported"
    divided = unfold_cube(torch.from_numpy(voxel), last_dims=3, size=16, step=16).numpy()
    empty   = (divided.sum(axis=-1)==0)
    full    = (divided.sum(axis=-1)==16**3)
    partial = np.logical_and(1-full, 1-empty)
    empty_idx, full_idx, partial_idx = np.where(empty)[0], np.where(full)[0], np.where(partial)[0]
    shape_vocab = np.zeros((1+1+len(partial_idx), 16*16*16), dtype=np.bool)
    vocab_idx   = np.zeros((16*16*16), dtype=np.int16)
    # 0: empty, 1: full, >1: various parts
    shape_vocab[1] = 1
    shape_vocab[2+np.arange(len(partial_idx))] = divided[partial_idx]
    vocab_idx[partial_idx] = 2+np.arange(len(partial_idx))
    vocab_idx[full_idx]    = 1
    #shape_vocab = shape_vocab.astype(bool)
    assert ((shape_vocab[vocab_idx] != divided).sum()==0), "Invalid compression"
    if packbits==True:
        shape_vocab = np.packbits(shape_vocab, axis=-1)
    return shape_vocab, vocab_idx # uint8, int16
def decompress_voxels(shape_vocab, vocab_idx, unpackbits=True):
    # 20x + faster than compress_voxels
    if unpackbits==True:
        shape_vocab = np.unpackbits(shape_vocab, axis=-1)
    unfolded = shape_vocab[vocab_idx]
    folded   = fold_cube(torch.from_numpy(unfolded), N=3).numpy()
    return folded
def einops_compress_voxels(grid, packbits=False):
    #assert(voxel.shape[-1]==256), "Only 256-> 16x16 dims is supported"
    assert grid.shape[-1]%block_size==0, "Irregular grid (can not devide block_size)"
    block_num = grid.shape[-1]//block_size
    blocked = rearrange(grid, "(x b1) (y b2) (z b3) -> x y z (b1 b2 b3)", b1=block_size, b2=block_size, b3=block_size)
    empty   = (blocked.sum(axis=-1)     == 0)
    full    = ((1-blocked).sum(axis=-1) == 0)
    partial = np.logical_and(1-full, 1-empty)
    empty_idx, full_idx, partial_idx = np.where(empty)[0], np.where(full)[0], np.where(partial)[0]
    shape_vocab = np.zeros((1+1+len(partial_idx), block_size**3), dtype=np.float32)
    vocab_idx   = np.zeros((block_num**3), dtype=np.int16)
    # 0: empty, 1: full, >1: various parts
    shape_vocab[0] = 0
    shape_vocab[1] = 1
    shape_vocab[2+np.arange(len(partial_idx))] = blocked[partial_idx]
    vocab_idx[empty]       = 0
    vocab_idx[full_idx]    = 1
    vocab_idx[partial_idx] = 2+np.arange(len(partial_idx))
    #shape_vocab = shape_vocab.astype(bool)
    assert ((shape_vocab[vocab_idx] != blocked).any()==0), "Invalid compression"
    #if packbits==True:
    #    shape_vocab = np.packbits(shape_vocab, axis=-1)
    return shape_vocab, vocab_idx # uint8, int16
def einops_decompress_voxels(shape_vocab, vocab_idx, unpackbits=True):
    # 20x + faster than compress_voxels
    unfolded = shape_vocab[vocab_idx]
    folded   = fold_cube(torch.from_numpy(unfolded), N=3).numpy()
    return folded

def fold_unittest():
    testth = torch.tensor([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15.]])
    unfolded = unfold_cube(testth, size=2, step=2, last_dims=2)
    folded   = fold_cube(unfolded, N=2)
    assert (testth!=folded).sum()==0

    voxels = np.random.rand(256,256,256) > .5
    shape_vocab, vocab_idx = compress_voxels(voxels)
    print("shape_vocab, vocab_idx", shape_vocab.shape, shape_vocab.dtype, vocab_idx.shape, vocab_idx.dtype)
    decompress             = decompress_voxels(shape_vocab, vocab_idx)
    assert (voxels!=decompress).sum()==0
    print("All past")

# binary 
def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).type_as(x)
    #mask = 2 ** torch.arange(bits).type_as(x)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).long()
def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).type_as(b)
    #mask = 2 ** torch.arange(bits).type_as(x)
    return torch.sum(mask * b, -1)

def zorder2tree(x, dim, bits):
    #timer = sysutil.Timer()
    shape = x.shape
    #timer.update()
    x += 1 << (bits*dim )
    #timer.update()
    shifts = torch.arange(bits+1).type_as(x) * dim
    #timer.update()
    treeind = (x.unsqueeze(-1) >> shifts)
    #timer.update()
    treeind = treeind.reshape(*shape[:-1], -1)
    #timer.update()
    # torch.unique could be very slow for cpu, 
    treeind = torch.from_numpy(np.unique(treeind.numpy()))
#    treeind = torch.unique(treeind, dim=-1, sorted=True)
    #timer.update()
    return treeind

# point to index and reverse
def ravel_index(tensor, shape):
    raveled = torch.zeros(*tensor.shape[:-1]).type_as(tensor)
    if tensor.shape[-1]==2:
        raveled = tensor[..., 0]*shape[1] + tensor[..., 1]
    elif tensor.shape[-1]==3:
        raveled = tensor[..., 0]*shape[1]*shape[2] + tensor[..., 1]*shape[2] + tensor[..., 2]
    else:
        raise ValueError("shape must be 2 or 3 dimensional")
    return raveled
def unravel_index(tensor, shape):
    unraveled = torch.zeros(*tensor.shape, len(shape)).type_as(tensor)
    if len(shape)==2:
        unraveled[..., 0] = tensor // shape[1]
        unraveled[..., 1] = tensor %  shape[1]
    elif len(shape)==3:
        s12 = shape[1]*shape[2]
        unraveled[..., 0] = tensor // s12
        unraveled[..., 1] = tensor %  s12 // shape[2]
        unraveled[..., 2] = tensor %  s12 %  shape[2]
    else:
        raise ValueError("shape must be 2 or 3 dimensional")
    return unraveled
def ravel_unittest():
    idx = np.arange(9)
    npunravel = np.array(np.unravel_index(idx, (3,3))).swapaxes(0,-1)
    unraveled = unravel_index(torch.from_numpy(idx)[None,...], (3,3))
    assert ( npunravel==(unraveled[0].numpy())).all(), print(npunravel,"\n",unraveled[0].numpy())
    raveled   = ravel_index(unraveled, (3,3))
    assert ( idx==(raveled[0].numpy())).all(), print(idx,"\n",raveled[0].numpy())
    
    idx = np.arange(27)
    shape = (3,3,3)
    npunravel = np.array(np.unravel_index(idx, shape)).swapaxes(0,-1)
    unraveled = unravel_index(torch.from_numpy(idx)[None,...], shape)
    assert ( npunravel==(unraveled[0].numpy())).all(), print(npunravel,"\n",unraveled[0].numpy())
    raveled   = ravel_index(unraveled, shape)
    assert ( idx==(raveled[0].numpy())).all(), print(idx,"\n",raveled[0].numpy())
    print(unraveled)

def ravel_index_zorder(tensor, depth):
    tshape = tensor.shape
    dim = tshape[-1]
    binstrings = dec2bin(tensor, bits=depth)
    zorder_ind = binstrings.transpose(-1,-2)
    zorder_ind = zorder_ind.reshape( *tshape[:-1], -1)
    raveled_zorder = bin2dec(zorder_ind, bits=depth*dim)
    return raveled_zorder
def unravel_index_zorder(tensor, dim, depth):
    tshape = tensor.shape
    binstrings = dec2bin(tensor, bits=depth*dim)
    zorder_ind = binstrings.reshape(*tshape, -1, dim)
    zorder_ind = zorder_ind.transpose(-1,-2)
    unraveled_zorder = bin2dec(zorder_ind, bits=depth)
    return unraveled_zorder
def zorder_ravel_unittest():
    idx = torch.arange(16)
    mind= unravel_index(idx, shape=(4,4))
    raveled = ravel_index_zorder(mind, depth=2)
    unraveled=unravel_index_zorder(raveled,dim=2,depth=2)
    print("idx", idx)
    print("mind", mind)
    print("raveled", raveled)
    print("unraveled", unraveled)
    assert (mind==unraveled).all()
def flat2zorder(raveled_flat, dim, depth):
    flat_inds      = unravel_index(raveled_flat, shape=(2**depth,)*dim)
    raveled_zorder = ravel_index_zorder(flat_inds, depth=depth)
    return raveled_zorder
def point2index(points, grid_dim=32, ravel=False, ravel_type="flat", ret_relative=False):
    """Convert points in [-1,1]*dim to indices of (grid_dim,)*dim grid.
    The grid is generated using 'in' mode (pixel/voxel mode)

    Args:
        points (torch.Tensor): [*,dim]
        grid_dim (int, optional): dimension of the grid. Defaults to 32.

    Returns:
        [*,dim]: the returned indices if ravel=False
        [*,]:    raveld indices if ravel=True
    """
    pt_dim = points.shape[-1]
    # obsolete implementation
    #    offset = 1./grid_dim
    #    eps = 1e-5 # scale (1,1,...) slightly inside to avoid lying on the boundary
    #    # max loc 1 is corresponding to  2-2*offset, we need to multiply (grid_dim-1)/(2-2*offset)
    #    scale = (grid_dim-1-eps)/(2-2*offset)
    #    shift_points = (points + 1 - offset) * scale

    # first map points from [-1,1] to [0, 1], then to [-0.5, grid_dim-0.5]
    points01 = (points+1)/2
    shift_points = points01 * grid_dim - 0.5
    # round float to get index, clamping to prevent error in corner cases
    float_index = torch.clamp(torch.round(shift_points), min=0.0, max=grid_dim-1)
    index = float_index.long()
    if ravel==True:
        if ravel_type=="flat":
            index = ravel_index(index, shape=(grid_dim,)*pt_dim)
        elif ravel_type=="zorder":
            depth = np.log2(grid_dim)
            assert np.mod(depth, 1)==0., np.mod(depth,1)
            index = ravel_index_zorder(index, depth=depth)
    if ret_relative==True:
        grid_points   = index2point(float_index, grid_dim=grid_dim)
        relative_dist = points - grid_points
        return index, grid_points, relative_dist
    else:
        return index
def index2point(index, grid_dim=32):
    """ inverse of point2index
        grid index to coordinate, fall into the center of a cell
        grid_dim can be an array
    """
    #offset = 1./grid_dim
    # obsolete implementations
    #    eps = 1e-5
    #    scale = (grid_dim-1-eps)/(2-2*offset)
    #    points = index.float() / scale - 1 + offset
    points01 = ( index + .5 ) / (grid_dim)
    points = points01*2 -1
    return points

def unravel_index_to_point(tensor, shape):
    nparray=False
    if type(tensor) is np.ndarray:
        nparray = True
        tensor = torch.from_numpy(tensor)
    unraveled = unravel_index(tensor, shape)
    point = index2point(unraveled)
    if nparray == True:
        point = point.numpy()
    return point


#from sysutil import Timer
def point2tree(points, depth=6, max_length=-1):
    #timer = sysutil.Timer()
    dim = points.shape[-1]
    grid_dim = 2**depth
    #timer.update()
    zorder = point2index(points, grid_dim=grid_dim, ravel=True, ravel_type="zorder")
    #timer.update()
    tree = zorder2tree(zorder, dim=dim, bits=depth)
    #timer.update()
    if max_length>-1:
        tree = tree[:max_length]
    return tree
def tree2bboxes(tree, dim, depth):
    first_bit = (torch.log2(tree.float()).floor()).long()
    #shifts = torch.arange(depth+1).type_as(x) * dim
    #treeind = (x.unsqueeze(-1) >> shifts)
    tdepth = (first_bit / dim)
    # remove the first 1 bit (root bit)
    #significant_bit =
    treeind   = tree - 2**first_bit
    # (*, dim)
    treeind   = unravel_index_zorder(treeind, dim=dim, depth=depth)

    grid_dims = 2**tdepth
    boxcenter = index2point(treeind, grid_dim=grid_dims.unsqueeze(-1))
    boxlen = 1./(2**tdepth) # [-1,1]

    return boxcenter, boxlen, tdepth

def point2voxel(points, grid_dim=32, ret_coords=False):
    """Voxelize point cloud, [i][j][k] correspond to x, y, z directly
       [-1,1]

    Args:
        points (torch.Tensor): [B,num_pts,x_dim]
        grid_dim (int, optional): grid dimension. Defaults to 32.

    Returns:
        torch.Tensor: [B,(grid_dim,)*x_dim]
    """
    if type(points) is np.ndarray:
        points = torch.from_numpy(points).float()
    voxel = torch.zeros(points.shape[0], *((grid_dim,)*points.shape[-1])).type_as(points)
    inds = point2index(points, grid_dim)
    # make all the indices flat to avoid using for loop for batch
    # (B*num_points, x_dim)
    inds_flat = inds.view(-1,points.shape[-1])
    # [1,2,3] becomes [1,1,1,...,2,2,2,...,3,3,3,...]
    binds = torch.repeat_interleave(torch.arange(points.shape[0]).type_as(points).long(), points.shape[1])
    if points.shape[-1]==2:
        voxel[binds, inds_flat[:,0], inds_flat[:,1]] = 1
    if points.shape[-1]==3:
        voxel[binds, inds_flat[:,0], inds_flat[:,1], inds_flat[:,2]] = 1
    if ret_coords==True:
        x_dim = points.shape[-1]
        coords = nputil.makeGrid(bb_min=[-1,]*x_dim, bb_max=[1,]*x_dim, shape=[grid_dim,]*x_dim, indexing="ij")
        coords = torch.from_numpy(coords[None,...])
        return voxel, coords
    else:
        return voxel


###### MODEL RELATED ######
def split_batch(batch, sub_batch_size):
    assert type(batch) is dict, "batch must be a dict"
    splited = {}
    sbatch = []
    for key in batch:
        splited[key] = batch[key].split(sub_batch_size)
    for i in range(len(splited[key])):
        sbatch.append({})
        for key in splited:
            sbatch[i][key] = splited[key][i]
    return sbatch

def subbatch_run_model(model, batch_input, other_input=dict(), sub_batch_size=32, verbose=False):
    """Run model with sub-batch size

    Args:
        model (torch.nn.Module): model
        batch_input (dict): input for the model that has batch dimension
        other_input (dict): other input
        sub_batch_size (int): sub-batch size

    Returns:
        dict: output of the model
    """
    sub_batches = split_batch(batch_input, sub_batch_size)
    outputs = []
    if verbose==True:
        for sbatch in sysutil.progbar(sub_batches):
            output = model(**sbatch, **other_input)
            outputs.append(output)
    else:
        for sbatch in sub_batches:
            output = model(**sbatch, **other_input)
            outputs.append(output)
    assert len(outputs)>0, "No output"
    # aggregate outputs
    if type(outputs[0]) is dict:
        agg_output = {}
        for key in outputs[0]:
            agg_output[key] = torch.cat([o[key] for o in outputs], dim=0)
    elif type(outputs[0]) in [list, tuple]:
        agg_output = []
        for i in range(len(outputs[0])):
            agg_output.append( torch.cat([o[i] for o in outputs], dim=0) )
    elif type(outputs[0]) is torch.Tensor:
        agg_output = torch.cat(outputs, dim=0)
    else:
        raise TypeError(f"Type {type(outputs[0])} is not supported")

    return agg_output
subbatch_run_func = subbatch_run_model
