
import torch
import sklearn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from xgutils import nputil, sysutil

class BoxShapeDataset(Dataset):
    def __init__(self, dim=1, griddim=64, fixed_axes=[], bound=.8, size=1024, mode='default'):
        self.dim, self.bound, self.griddim = dim, bound, griddim
        self.mode = mode
        self.grid_points = torch.from_numpy(nputil.makeGrid(bb_min = -1*np.ones((dim)), bb_max = 1*np.ones((dim)), shape=[griddim,]*dim)).float()
        self.min_gap = 2./(griddim-1)
        self.fixed_axes = fixed_axes
        self.length = size
        self.randf = torch.rand
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        #if not self.fixed_corner:
        #    raise NotImplementedError('only implemented fixed corner')
        lower_corner = torch.ones((self.dim))*-self.bound
        if self.mode == 'default':
            upper_corner =  (self.bound - lower_corner)*self.randf(self.dim) + lower_corner + self.min_gap*2
        elif 'XorY' in self.mode:
            upper_corner = torch.zeros((self.dim)) - self.bound/2
            choice = np.random.randint(2)
            if 'Fixed' in self.mode:
                upper_corner[choice] = self.bound
            else:
                upper_corner[choice] = (self.bound - upper_corner[choice])*self.randf(1)[0] + upper_corner[choice] + self.min_gap*2
        for axis in self.fixed_axes:
            upper_corner[axis] = self.bound
        Xtg = self.grid_points
        #print(self.grid_points.shape, self.grid_points.dtype, lower_corner.dtype)
        Ytg = (self.grid_points >= lower_corner) & (self.grid_points <= upper_corner)
        Ytg = Ytg.all(dim=-1).float()[..., None]
        #if self.target_as_context==True:
        #    return Xtg, Ytg, Xtg, Ytg
        Xct, Yct = [], []
        if self.dim == 1:
            Xct = torch.stack([lower_corner, upper_corner])
        else:
            sampleN = 256
            for i in range(self.dim):
                xct = torch.zeros((sampleN, self.dim))
                xct[:,i]    = lower_corner[i]
                xct[:,:i]   = torch.rand(sampleN, i) \
                                * (upper_corner[None,:i] - lower_corner[None,:i]) \
                                +  lower_corner[None,:i]
                xct[:,i+1:] = torch.rand(sampleN, self.dim-i-1) \
                                * (upper_corner[None,i+1:] - lower_corner[None,i+1:]) \
                                +  lower_corner[None,i+1:]
                Xct.append(xct)
                xct = xct.clone()
                xct[:,i]    = upper_corner[i]
                xct[:,:i]   = torch.rand(sampleN, i) \
                                * (upper_corner[None,:i] - lower_corner[None,:i]) \
                                +  lower_corner[None,:i]
                xct[:,i+1:] = torch.rand(sampleN, self.dim-i-1) \
                                * (upper_corner[None,i+1:] - lower_corner[None,i+1:]) \
                                +  lower_corner[None,i+1:]
                Xct.append(xct)
            Xct = torch.cat(Xct, dim=0)
            Xct = Xct[ np.random.choice(Xct.shape[0], sampleN) ]
        #Xct = self.ct_pattern(Xct)
        Yct = torch.ones((*Xct.shape[:-1],1))*.5
        #return Xct.float(), Yct.float(), Xtg.float(), Ytg.float()
        
        item = dict(context_x=Xct.float().numpy(),
                    context_y=Yct.float().numpy(),
                    target_x= Xtg.float().numpy(),
                    target_y= Ytg.float().numpy(),
                    )
        return item
    def get_dataloader(self, batch_size=1, shuffle=False):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_test,
            num_workers=self.opt.num_workers,
        )


class ShapeDataset(Dataset):
    def __init__(self, dataset='MNISTShape' ,split='train', cate="all", zoomfac=1):
        dpath = f'datasets/{dataset}/{split}.hdf5'
        self.dataDict = dataDict = nputil.readh5(dpath)
        total_length = dataDict['context_x'].shape[0]
        all_ind = np.arange(total_length)
        if cate == "all":
            self.subset = all_ind
        else:
            self.subset = dataDict[cate]
        self.length = len(self.subset)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        index = self.subset[index]
        dataDict = self.dataDict
        # tgx ,tgy = dataDict['target_x'][index], dataDict['target_y'][index]
        # tgx[:,0] = scipy.ndimage.zoom( nputil.array2NDCube(tgx[:,0],N=), zoom_factor, order=1)
        # tgy = scipy.ndimage.zoom( tgy, zoom_factor, order=1)
        tgx = torch.from_numpy(dataDict['target_x'][index]).float()
        tgy = torch.from_numpy(dataDict['target_y'][index]).float()
        item = dict(context_x = torch.from_numpy(dataDict['context_x'][index]).float(),
                    context_y = torch.from_numpy(dataDict['context_y'][index]).float(),
                    target_x  = tgx,
                    target_y  = tgy,
                    )
        return item

