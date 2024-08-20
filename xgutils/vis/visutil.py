import os
import io
import re
import h5py
import glob
import copy
from time import time

import scipy
import numpy as np
import skimage
from PIL import Image, ImageFont, ImageDraw

import matplotlib as mpl
from matplotlib import offsetbox
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mpcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, but is otherwise unused.
import matplotlib.pyplot as plt

from xgutils import nputil as util
from xgutils import nputil, sysutil
from xgutils import miscutil

with nputil.temp_seed(315):
    unique_colors = np.random.rand( 100000, 3)
unique_colors[:4] = np.array([[1.,0,0],[0,1,0],[0,0,1],[1,1,0]])

div=8
marginIn=.3
marginOut=.4
sdfcolors_inside  = plt.cm.Spectral(  np.linspace(1., 1-marginIn, 512) )
#sdfcolors_inside[-div:,:2] = sdfcolors_inside[-div,:2]
sdfcolors_inside[-div:,:3] = np.zeros_like(sdfcolors_inside[-div,:3])+.4#sdfcolors_inside[-div,:2]
sdfcolors_outside = plt.cm.Spectral( np.linspace(marginOut, 0., 512) )
#sdfcolors_outside[:div,1:] = sdfcolors_outside[div,1:]#sdfcolors_outside[:,1:]*.7
sdfcolors_outside[:div,:3] = np.zeros_like(sdfcolors_outside[:div,:3])+.4#sdfcolors_outside[:,1:]*.7
sdf_colors = np.vstack((sdfcolors_inside, sdfcolors_outside))
sdf_cmap = mpl.colors.LinearSegmentedColormap.from_list('sdf_cmap',    sdf_colors)

tmagmaColors = plt.cm.Oranges( np.linspace(0., 1., 256) )
tmagmaColors[:,3] = util.logistic(np.linspace(0.,1.,256), x0=.3,k=10)
tmagmaColors[:40,3]=0
tmagma_cmap = mpl.colors.LinearSegmentedColormap.from_list('tmagma',    tmagmaColors)

rhotColors = plt.cm.gist_heat(  np.linspace(1., 0, 256) )
rhotColors = np.r_[np.zeros((40,4)),rhotColors]
rhot_cmap = mpl.colors.LinearSegmentedColormap.from_list('rhot',    rhotColors)

rRdYlBuColors  = plt.cm.RdYlBu(  np.linspace(1.,0., 512) )
rRdYlBu_cmap = mpl.colors.LinearSegmentedColormap.from_list('rRdYlBu',    rRdYlBuColors)
def get_random_color(size, color_dim=16):
    # 32**3 = 32767
    with nputil.temp_seed(314):
        color = np.zeros((size, 3))
        rd = np.random.randint(color_dim**3, size=(size))
        color[..., 0] = rd // (color_dim*color_dim) 
        color[..., 1] = rd % (color_dim*color_dim) // color_dim
        color[..., 2] = rd % (color_dim*color_dim) % color_dim
    fac = 256 / color_dim
    return (color*fac) / 256
def get_random_cmap(size):
    rd_cmap = mpl.colors.LinearSegmentedColormap.from_list('random',    get_random_color(size))
    return rd_cmap
#random_cmap = mpl.colors.LinearSegmentedColormap.from_list('random',    get_random_color(1024))
def blankImg(resolution=(256,256), format="RGBA"):
    return np.zeros( (*resolution, 4) )
def newFig(resolution=(400.,400.), tight=True):
    dpi = resolution[0]/4.
    fig     = plt.figure(figsize=(resolution[0]/dpi, resolution[1]/dpi), dpi=dpi, tight_layout=tight)
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig
def newPlot(resolution=(400.,400.), projection=None, withMargin=False, tight=True, \
            fig=None, ax=None, plotrange=None, ticks=None, axes_off=False, color_range=None):
    dpi = resolution[0]/4.
    if ax is None:
        fig, ax = plt.subplots(figsize=(resolution[0]/dpi, resolution[1]/dpi), dpi=dpi, tight_layout=tight)
    else:
        fig = ax.figure
    #if fig is None:
    #    fig, ax = plt.subplots(figsize=(resolution[0]/dpi, resolution[1]/dpi), dpi=dpi, tight_layout=tight)
    #elif ax is None:
    #    ax = fig.add_axes()

    # set plotrange
    if plotrange is not None:
        if type(plotrange) is np.ndarray:
            ax.set_xlim(plotrange[0,0], plotrange[0,1])
            ax.set_ylim(plotrange[1,0], plotrange[1,1])
        elif type(plotrange) is str:
            if plotrange=='square':
                plotrange = np.array([[-1.,1.],[-1.,1.]])
            elif plotrange=='psquare':
                plotrange = np.array([[0.,1.],[0.,1.]])
            else:
                raise ValueError(f"plotrange {plotrange} is not supported.")
        else:
            raise TypeError(f"plotrange type {type(plotrange)} is not supported.")
    if ticks is None:
        ax.set_xticks([])
        ax.set_yticks([])
    if axes_off==True:
        ax.axis('off')
    if color_range is not None:
        ax.set_clim(color_range)
    return fig, ax
def newPlot3D(resolution=(400.,400.), projection='3d', withMargin=False, tight=True, fig=None, ax=None):
    dpi = resolution[0]/4.
    if fig is None:
        fig = plt.figure(figsize=(resolution[0]/dpi, resolution[1]/dpi), dpi=dpi, tight_layout=tight)
    if ax is None:
        ax = fig.add_subplot('111', projection=projection)
    #ax.set_aspect('equal')
    return fig, ax
def readImg(path):
    path = os.path.expanduser(path)
    return mpimg.imread(path)
def saveFig(target, fig, dpi=None):
    # target can be output file path or a IO buffer
    if type(fig) is tuple: # if the input is fig=(fig, ax) then only keep fig
        fig = fig[0]
    fig.savefig(target, transparent=True, dpi=dpi)
def saveImg(target, img):
    # make sure img is C-contiguous
    # if img has float dtype
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0., 1.)
    img = img.copy(order='C')
    mpimg.imsave(target, img)
    return img
def saveImgs(targetDir='./', baseName='out_', imgs=[]):
    if not os.path.exists(targetDir):
        util.mkdirs(targetDir)
    if not util.isIterable(imgs):
        imgs = [imgs]
    for i,img in sysutil.progbar(enumerate(imgs)):
        saveImg(os.path.join(targetDir, '%s%d.png'%(baseName,i)), img)
def resizeImg(img, size=(256,256)):
    return skimage.transform.resize(img, size, order=1, preserve_range=True, anti_aliasing=True)
def saveFigs(targetDir='./', baseName='out_', figs=[]):
    imgs = figs2imgs(figs)
    saveImgs(targetDir, baseName=baseName, imgs=imgs)
def fig2img(fig, dpi=None, closefig=True):
    t0 = time()
    buf = io.BytesIO()
    saveFig(buf, fig, dpi=dpi) # save figure to buffer
    buf.seek(0)
    image = np.array(Image.open(buf)).astype(float)/255.
    buf.close()
    if closefig:
        plt.close(fig)
    return image
def figs2imgs(figs):
    if not util.isIterable(figs):
        figs = [figs]
    return list(map(fig2img, figs))
def showFig(fig):
    pass
def showImg(img, scale=1, title=None, cmap=None, resolution=None, vmin=0., vmax=1.):
    if resolution is not None:
        fig,ax = newPlot(resolution=resolution)
    else:
        resolution = (int(img.shape[1]*scale), int(img.shape[0]*scale))
        fig, ax = newPlot(resolution=resolution)
    ax.set_axis_off()
    if title is not None:
        fig.suptitle(title)

    ax.imshow(img, cmap, vmin=vmin, vmax=vmax)
    plt.show()
    return fig, ax

def drawText2Img(img, text="test", pos=(0,0), font_size=None, font_path=None):
    if font_path is None:
        font_path = miscutil.get_fontpath()
    pil_img = Image.fromarray( (img*255).astype(np.uint8), mode='RGBA' )
    draw = ImageDraw.Draw(pil_img)
    if font_size is None:
        font_size = max(img.shape[0]//20, 16)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, (255,0,0), font=font)
    return np.array(pil_img)/255.
draw_text = drawText2Img
def img2rgba(img):
    assert len(img.shape)>=2, "Image must have at least 2 dimensions"
    assert len(img.shape)<=3, "Image must have at most 3 dimensions"
    if len(img.shape)==2:
        img = img[...,None]
    if img.shape[-1]==1:
        # repeat 3 times
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1]==2:
        img = np.concatenate([img, np.ones((*img.shape[:-1],1))], axis=-1)
    if img.shape[-1]==3:
        img = np.concatenate([img, np.ones((*img.shape[:-1],1))], axis=-1)
    if img.shape[-1]>4:
        img = img[...,:4]
    if img.shape[-1]==4:
        return img

def imageGrid(imgs, shape=None, zoomfac=1, min_height=128):
    """ image must have value of 0 to 1. """
    for i in range(len(imgs)):
        imgs[i] = img2rgba(imgs[i])

    if zoomfac!=1:
        imgs[0] = scipy.ndimage.zoom( imgs,[zoomfac, zoomfac, 1], order=0)
    if imgs[0].shape[0] < min_height:
        nshape = (min_height, min_height/imgs[0].shape[0]*imgs[0].shape[1])
        imgs[0] = skimage.transform.resize(imgs[0], nshape, anti_aliasing = True)
    for i in range(1, len(imgs)): # resize to the same height
        if imgs[i].shape[0]!=imgs[0].shape[0]:
            nshape = np.array(imgs[i].shape)
            nshape[:2] = nshape[:2] * imgs[0].shape[0]/nshape[0]
            nshape = nshape.astype(int)
            imgs[i] = skimage.transform.resize(imgs[i], nshape[:2], anti_aliasing=True)

    numFig = len(imgs)
    assert numFig>0, "No images to show"
    if shape is None:
        assert numFig >= shape[0]*shape[1], "Not enough images to fill the grid"
        shape = np.ceil(np.sqrt(numFig)).astype(int)
        shape = np.array([shape, shape])
        shape[0] = np.ceil(numFig / shape[1]).astype(int) # remove blank row(s)
    else:
        shape = np.array(shape).T
        assert len(shape)==2, "Shape must be 2 dimensional"
        assert shape[0]!=-1 or shape[1]!=-1, "(-1,-1) not understood"
        if shape[0]==-1:
            shape[0] = np.ceil(numFig/shape[1]).astype(int)
        if shape[1]==-1:
            shape[1] = np.ceil(numFig/shape[0].astype(int))

    row_imgs = []
    for rowi in range( shape[0] ):
        row_beg = rowi*shape[1]
        row_end = (rowi+1)*shape[1]
        row_subimgs = imgs[row_beg:row_end]
        row_img = np.concatenate(row_subimgs, axis=1)
        row_imgs.append(row_img)
    if row_imgs[-1].shape[1] != row_imgs[0].shape[1]: # padd to the same width
        row_img = np.zeros((row_img[-1].shape[0], row_imgs[0].shape[1], 4))
        row_img[:row_imgs[-1].shape[0], :row_imgs[-1].shape[1]] = row_imgs[-1]
        row_imgs[-1] = row_img
    img_grid = np.concatenate(row_imgs, axis=0)
    return img_grid
    
def figGrid(figs, shape=None):
    return imageGrid(figs2imgs(figs), shape=shape)
def addCBar(target, fig, ax, nbins=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(target, cax=cax)
    if nbins is not None:
        cbar.ax.locator_params(nbins=3)
    return cbar
def standalone_colorbar():
    pl = plt
    a = np.array([[0,1]])
    pl.figure(figsize=(.12, 3))
    img = pl.imshow(a, cmap="plasma")
    pl.gca().set_visible(False)
    cax = pl.axes([0.1, 0.2, 0.8, 0.6])
    cbar = pl.colorbar(orientation="vertical", cax=cax)
    cyticks = np.array([1e-5, 1e-2, 0.1, 0.5])
    cbar.set_ticks(cyticks**expr)
    cbar.set_ticklabels(cyticks)
    #plt.savefig("colorbar.pdf", bbox_inches='tight')
    plt.show()
class Visualizer():
    def __init__(self, sample, label, pred=None):
        pass
def rescale(target, reference=None, truncate=False):
    if reference is None:
        reference = target
    rescaled = ( target - (reference.max() + reference.min())/2 ) / (reference.max() - reference.min()) + .5
    if truncate==True:
        rescaled[rescaled>1.]=1.
        rescaled[rescaled<0.]=0.
    return rescaled

def plot1D(samples=None, values=np.array([0,1,2,3]), title="plot"):
    fig, ax = newPlot()
    if samples is None:
        ax.plot(range(len(values)), values)
    fig.suptitle(title)
    return fig, ax
#def densityPlot(x,y,z,ax=plt,cmap='rainbow', squareRange=True, colorbar=True, ticks=None, norm=None):
def densityPlot(samples, values, cmap=sdf_cmap, bounds=None, squareRange=True, plotrange=np.array([[0.,1.],[0.,1.]]), colorbar=True, ticks=None, norm=None, 
    origin = 'lower', resolution=(400.,400.), fig=None, ax=None):
    """Generate density plot from samples

    Args:
        samples (np.ndarray): samples to interpolate from
        values (np.ndarray): samples
        cmap ([type], optional): [description]. Defaults to sdf_cmap.
        bounds ([type], optional): [description]. Defaults to None.
        squareRange (bool, optional): [description]. Defaults to True.
        plotrange ([type], optional): [description]. Defaults to np.array([[0.,1.],[0.,1.]]).
        colorbar (bool, optional): [description]. Defaults to True.
        ticks ([type], optional): [description]. Defaults to None.
        norm ([type], optional): [description]. Defaults to None.
        resolution (tuple, optional): [description]. Defaults to (400.,400.).
        fig ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    fig, ax = newPlot(resolution=resolution, fig=fig, ax=ax)
    # 2D samples with value
    z = values
    if bounds is None:
        vmin, vmax = z.min(), z.max()
    else:
        vmin, vmax = bounds
    if norm == 'log':
        norm = mpcolors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm == 'sdf':
        vmin, vmax = min(vmin, -0.01), max(vmax,0.99)
        norm = mpcolors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)
    elif norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if samples is None: # grid input
        zi = z
    else:
        x,y = samples[:,0], samples[:,1]
        if squareRange == True:
            #xi = yi = np.arange(0,1.002,0.002)
            pass
        if plotrange is None:
            xi = np.linspace( x.min(), x.max(), 500)
            yi = np.linspace( y.min(), y.max(), 500)
        else:
            xi = np.linspace(*plotrange[0], 500)
            yi = np.linspace(*plotrange[1], 500)
        xi,yi = np.meshgrid(xi,yi)
        zi = scipy.interpolate.griddata((x,y),z,(xi,yi), method='linear')

    im = ax.imshow(zi, origin=origin, cmap=cmap, norm=norm)
    if ticks is None:
        ax.set_xticks([])
        ax.set_yticks([])
    if colorbar==True:
            #divider = make_axes_locatable(plt.gca())
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
    plt.close(fig) #Don't show it if 
    return fig, ax
def diffDensityPlot(s1,l1,s2,l2,ax,cmap='rainbow', colorbar=True):
    xi = yi = np.arange(0,1.002,0.002)
    xi,yi = np.meshgrid(xi,yi)
    zi1 = scipy.interpolate.griddata((s1[:,0],s1[:,1]),l1,(xi,yi), method='linear')
    zi2 = scipy.interpolate.griddata((s2[:,0],s2[:,1]),l2,(xi,yi), method='linear')
    im = ax.imshow( np.abs(zi1-zi2), extent=(0,1,0,1),origin='lower', cmap=cmap)
    if colorbar==True:
        if ax is plt:
            divider = make_axes_locatable(plt.gca())
        else:
            divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    return im
def diffDensityPlots(opt, candidateIds=[1,165,187, 132], cmap='rainbow', colorbar=True):
    from data import create_dataset
    xi = yi = np.arange(0,1.002,0.002)
    xi,yi = np.meshgrid(xi,yi)
    samples, labels, zis = [], [], []
    dataset = create_dataset(opt)
    sid = dataset.dataloader.dataset.dataDict['shapeId']
    for candidateId in candidateIds:
        samples.append( dataset.dataloader.dataset.dataDict['sample'][sid==candidateId] )
        labels.append( dataset.dataloader.dataset.dataDict['label'][sid==candidateId]   )
        zis.append( scipy.interpolate.griddata((samples[-1][:,0],samples[-1][:,1]),labels[-1],(xi,yi), method='linear') )
    samples, labels, zis = np.array(samples), np.array(labels), np.array(zis)
    zi_mean = zis.mean(axis=0)
    
    fig, axes = plt.subplots(2, len(candidateIds)+1, figsize=(15,15))
    axes[1, 0].imshow(zi_mean, extent=(0,1,0,1), origin='lower', cmap=cmap)
    for i in range(samples.shape[0]):
        im = axes[1, i+1].imshow( np.abs(zi_mean - zis[i]), extent=(0,1,0,1),origin='lower', cmap=cmap)
        im = axes[0, i+1].imshow( zis[i], extent=(0,1,0,1),origin='lower', cmap=cmap)
    plt.setp(axes.flat, aspect=1.0, adjustable='box')
    fig.tight_layout()
    return im
def addColorbar2Plot(fig, ax, im, position='right'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    return cbar
def gradientPlot(x,y,z,ax=plt,cmap='rainbow', colorbar=True):
    xi = yi = np.arange(0,1.002,0.002)
    xi,yi = np.meshgrid(xi,yi)
    zi = scipy.interpolate.griddata((x,y),z,(xi,yi), method='linear')
    dx, dy = np.gradient(zi)
    zi = np.sqrt(dx**2+dy**2)
    im = ax.imshow(zi,extent=(0,1,0,1),origin='lower', cmap=cmap)
    if colorbar==True:
        if ax is plt:
            divider = make_axes_locatable(plt.gca())
        else:
            divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    return im
def simplePlot(samples, labels, preds=None, save_path=None):
    if preds is None:
       preds = np.zeros_like(labels)
    valueses=[labels, preds, np.abs(labels-preds)]
    figs = [densityPlot(samples, values=values) for values in valueses]
    grid = figGrid(figs, shape=(1,3))
    return grid

def linePlot(samples, labels=None, ax=plt, style='', c_ind=0, plotrange=np.array([[0.,1.],[0.,1.]]), ticks=None):
    """Plot line (same as plt.plot, but with my preferred settings)

    Args:
        samples ([type]): [description]
        labels ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to plt.
        style (str, optional): [description]. Defaults to ''.
        c_ind (int, optional): color index(mod by 20)(which color to use in tab20c). Defaults to 0.
        plotrange ([type], optional): [description]. Defaults to np.array([[0.,1.],[0.,1.]]).
        ticks ([type], optional): [description]. Defaults to None.
    """
    ax.plot(samples[:,0], samples[:,1], style, c = plt.cm.tab20(c_ind%20))
    if ax is plt:
        ax = plt.gca() #.set_aspect('equal', adjustable='box')
    ax.set_xlim(plotrange[0,0], plotrange[0,1])
    ax.set_ylim(plotrange[1,0], plotrange[1,1])
    ax.set_aspect('equal', adjustable='box' )
    if ticks is None:
        ax.set_xticks([])
        ax.set_yticks([])
    return ax.figure, ax

def scatterPlot(samples, labels=None, point_size = 3., cmap='rainbow', plotrange=np.array([[0.,1.],[0.,1.]]), ticks=None, colorbar=False, **kwargs):
    fig, ax = newPlot(**kwargs)
    im = ax.scatter(samples[:,0], samples[:,1], c=labels, s=point_size, cmap=cmap)
    #if ax is plt:
    #    ax = plt.gca() #.set_aspect('equal', adjustable='box')
    ax.set_xlim(plotrange[0,0], plotrange[0,1])
    ax.set_ylim(plotrange[1,0], plotrange[1,1])

    if colorbar==True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.locator_params(nbins=3)

    return fig, ax

def SDFPlotData(samples, labels=None, cmap='rainbow'):
    if labels is None:
        labels = samples[:,1]
    #print(labels.dtype,labels.shape)
    surfacePts = ((labels< .03) & (labels >=-.03))
    otherPts   = util.subsampleBoolArray(surfacePts, 4000)
    surfacePts = util.subsampleBoolArray(~surfacePts, 6000) # maximum 10000 points in the plot
    insidePts  = ((labels< .01) & otherPts)
    outsidePts = ((~insidePts) & otherPts)
    mask = (surfacePts | otherPts)
    labels[insidePts] = labels[outsidePts].min()
    scL = rescale(labels[mask],labels[mask])
    #filted= np.random.choice(ret.shape[0], 5000, replace=False)

    cmapF = mpl.cm.get_cmap(cmap)
    rgba_colors = cmapF(scL)
    rgba_colors[:, 3] = (1- np.power(scL, .1))*.76+.24
    print(rgba_colors[:, 3].min(), rgba_colors[:, 3].max())
    sizes = rgba_colors[:, 3]*10
    return mask, rgba_colors, sizes
def SDF3DPlot(samples, labels=None, rgba_colors=None, sizes=None, cmap='rainbow', title=None, ptsScale=.5, plotrange=np.array([[0.,1.],[0.,1.],[0.,1.]]), resolution=(400.,400.), fig=None, ax=None):
    fig, ax = newPlot3D(resolution=resolution, projection='3d',fig=fig, ax=ax)
    mask = np.ones(samples.shape[0], dtype=bool)
    if rgba_colors is None:
        mask, rgba_colors, sizes = SDFPlotData(samples, labels, cmap)
    ax.scatter(samples[mask,0], samples[mask,1], samples[mask,2], c=rgba_colors, s=sizes*ptsScale)#, mpl.rcParams['lines.markersize'] ** 2/5.)
    if title is None:
        title = "min:%.3f max:%.3f"%(labels.min(), labels.max())
    fig.suptitle(title)
    return fig, ax
def SDF2DPlot(gridData, pts=None, scatterSize=6, cmap=rRdYlBu_cmap, contour_mode=True, colorbar=True, 
        norm=None, ticks=None, bbox=None, im_origin="lower", indexing="xy",
        zoom=3,
        plotrange=np.array([[0.,1.],[0.,1.]]), valuerange=None, 
        resolution=(400.,400.), fig=None, ax=None):
    fig, ax = newPlot(resolution=resolution, fig=fig, ax=ax)
    
    gridDim = gridData.shape
    Z = scipy.ndimage.zoom(gridData, zoom, order=1)
    #Z = gridData
    if bbox is None:
        bbox = plotrange.T
    X, Y = np.meshgrid( np.linspace(bbox[0,0], bbox[1,0], gridDim[0]*zoom), \
                        np.linspace(bbox[0,1], bbox[1,1], gridDim[1]*zoom), \
                        indexing=indexing)
    
    vmin = min(Z.min(), -0.0001)
    vmax = max(Z.max(), 0.0001)
    if contour_mode==True:
        contour = ax.contour(1-X,Y,Z, origin=im_origin, levels=[0.], colors=('k',) ,linestyles=('-',), linewidths=(1,))
    if ticks is None:
        ax.set_xticks([])
        ax.set_yticks([])
    if contour_mode == True:
        field = ax.contourf(1-X,Y,Z, origin=im_origin, cmap=rRdYlBu_cmap, norm = mpcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax))
    else:
        field = ax.imshow(Z, origin=im_origin, cmap=rRdYlBu_cmap,  norm = mpcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax))
    #print(contourF)
    if colorbar==True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(field, cax=cax, format=mpl.ticker.FuncFormatter(lambda x,pos:'%.1f'%x))
    # if valuerange is not None:
    #     mi = valuerange[0]
    #     ma = valuerange[1]
    #     contourF.set_clim(mi, ma)
    if pts is not None:
        pc = ax.scatter(pts[:,0], pts[:,1], s=scatterSize, marker='o', c='b')
    ax.set_aspect('equal')
    return fig, ax
def FieldPlot(gridData, pts=None, extent=[[-1.,1.],[-1.,1.]], scatterSize=6, zoomfac=3., \
        cmap='Reds', valuerange=np.array([-1.,1.]), colorbar=True, **kwargs):
    fig, ax = newPlot(**kwargs)
    Z = scipy.ndimage.zoom(gridData, zoomfac, order=1)

    field = ax.imshow(np.flip(Z,axis=0), cmap=cmap, extent=[extent[0][0], extent[0][1], extent[1][0], extent[1][1]])
    field.set_clim(valuerange)
    #print(contourF)
    if colorbar==True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(field, cax=cax, format=mpl.ticker.FuncFormatter(lambda x,pos:'%.1f'%x))
        cbar.ax.locator_params(nbins=3)
    return fig, ax
def labels2rgba(labels, cmap='rainbow', return_rescaled=False, noRescale=False):
    if labels.max() - labels.min() <0.000001:
        scL = np.zeros_like(labels)
    else:
        if noRescale==True:
            scL = np.clip(labels, 0., 1.)
        else:
            scL = rescale(labels,labels)
    if type(cmap) is str:
        cmapF = mpl.cm.get_cmap(cmap)
    else:
        cmapF = cmap
    rgba_colors = cmapF(scL)
    # too_black = rgba_colors<.5
    # rgba_colors[too_black] = (rgba_colors[too_black]*2.*.7+.3)/2.
    if return_rescaled==True:
        return rgba_colors, scL
    return rgba_colors
def scatter3D(samples, labels=None, noRescale=False, maxPts=6000, ptsSize=6, axis=False, cmap='rainbow', alpha=1, title=None, plotrange=np.array([[0.,1.],[0.,1.],[0.,1.]]), resolution=(400.,400.), fig=None, ax=None):
    fig, ax = newPlot3D(resolution=resolution, projection='3d',fig=fig, ax=ax)
    all_pts = util.subsampleBoolArray(np.ones(samples.shape[0],dtype=bool), maxPts)
    if labels is None:
        #labels = samples[:,2]+samples[:,1]+samples[:,0]
        labels = samples[:,0] + 10*samples[:,1] + 3*samples[:,2] # color accoring to y 
    rgba_colors, rescaled = labels2rgba(labels[all_pts],cmap, return_rescaled=True, noRescale=noRescale)
    # print(rgba_colors[0])
    rgba_colors[...,3] *= alpha
    #rgba_colors[:,3] = 1 - rescaled*rescaled

    
    ax.set_xlim(plotrange[0,0], plotrange[0,1])
    ax.set_ylim(plotrange[1,0], plotrange[1,1])
    ax.set_zlim(plotrange[2,0], plotrange[2,1])
    # labels[insidePts] = labels[outsidePts].min()
    # scL = rescale(labels[all_pts],labels[all_pts])
    # #filted= np.random.choice(ret.shape[0], 5000, replace=False)

    # cmapF = mpl.cm.get_cmap(cmap)
    # rgba_colors = cmapF(scL)
    # rgba_colors[:, 3] = (1- np.power(scL,.1))*.7+.3
    im = ax.scatter(samples[all_pts,0], samples[all_pts,1], samples[all_pts,2], c=rgba_colors, s=ptsSize, cmap=cmap)#, s=rgba_colors[:,3]*10)#, mpl.rcParams['lines.markersize'] ** 2/5.)
    if axis==False:
        ax.set_axis_off()
    #addColorbar2Plot(fig, ax, im, 'bottom') 
    if title is None:
        title = "min:%.3f max:%.3f"%(labels.min(), labels.max())
    fig.suptitle(title)
    #plt.close(fig)
    return fig, ax 

def field3DPlot(outDir, figName, planes, cmap=sdf_cmap,sdfPlot=False, video=False):
    figDir  = os.path.join(outDir, figName)
    summaryPath = os.path.join(outDir, figName+'.png')
    util.mkdirs(figDir)
    norm = None
    if sdfPlot ==True:
        norm='sdf'
    figs = [densityPlot(samples=None,values=planes[i], cmap=cmap, norm=norm, bounds=np.array([planes.min(),planes.max()]))[0] \
                for i in range(planes.shape[0])]
    grid = figGrid(figs)
    saveImg(summaryPath, grid)
    if video:
        vidPath = os.path.join(outDir, figName+'.mp4')
        saveFigs(targetDir=figDir, baseName='', figs=figs)
        util.imgs2video(targetDir=outDir, folderName=figName)
    #plt.close()
    return grid

def plotPCs(pointclouds, cmap='rainbow', plotrange=np.array([[0.,1.],[0.,1.],[0.,1.]]), resolution=(400.,400.), fig=None, ax=None):
    fig, ax = newPlot3D(resolution=resolution, projection='3d',fig=fig, ax=ax)
    for pointcloud in pointclouds:
        all_pts = util.subsampleBoolArray(np.ones(pointcloud.shape[0],dtype=bool), 10000)#./pointclouds.shape[0])
        im = ax.scatter(pointcloud[all_pts,0], pointcloud[all_pts,1], pointcloud[all_pts,2])#, s=rgba_colors[:,3]*10)#, mpl.rcParams['lines.markersize'] ** 2/5.)
    return fig, ax
import matplotlib.patches as patches

def OctreePlot2D(tree, dim, depth, **kwargs):
    import torch
    from xgutils import ptutil
    assert dim==2
    boxcenter, boxlen, tdepth = ptutil.ths2nps(ptutil.tree2bboxes(torch.from_numpy(tree), dim=dim, depth=depth))    
    dflt_kwargs=dict(plotrange=np.array([[-1.,1.],[-1.,1.]])*1.2)
    dflt_kwargs.update(kwargs)
    maxdep = tdepth.max()
    fig, ax = newPlot( **dflt_kwargs )
    for i in range(len(tdepth)):
        dep=tdepth[i]
        length = boxlen[i]
        corner = (boxcenter[i]-boxlen[i])        
        lw=1+.5*np.exp(-dep)
        rect = patches.Rectangle(corner, 2*length, 2*length, linewidth=lw, edgecolor=plt.cm.plasma(dep/maxdep), facecolor='none')
        ax.add_patch(rect)
    return fig, ax

def seabornScatterPlot(samples, labels):
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame({'x':samples[:,0], 'y':samples[:,1], 'label':labels})
    g = sns.scatterplot(x="x", y="y", hue="label", s=16, data=df,legend='full', palette="Paired")
    g.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.gca().set_aspect('equal', adjustable='box')


def depthmapPlot(depthmap, cmap="inferno", max_clip=1e10, \
                equalize=True, plot_hist=False, transparent=False):
    """ in order to show more details in depth map, normalize the histogram first
    """
    img = depthmap - depthmap[depthmap!=0].min()
    img = np.clip(depthmap, 0., max_clip)
    img = img / img.max()
    valid = (depthmap!=0)
    #img = img**beta
    if equalize==True:
        nimg = nputil.histogram_equalization( img[valid].reshape(-1) )[0]
        img[valid] = nimg
        #mg = img*.8
    if plot_hist == True:
        plt.hist(img[valid].reshape(-1))
    img = plt.get_cmap(cmap)( 1-img )

    img[depthmap==0] = 0
    if transparent==False:
        img[...,-1] = 1.
    return img

def imgs2video(targetDir, folderName, frameRate=6):
    ''' Making video from a sequence of images
    
        Make a video from images with index, e.g. 1.png 2.png 3.png ...
        the images should be in targetDir/folderName/ 
        the output will be targetDir/folderName.mp4 .
        Args:
            targetDir: the output directory
            folderName: a folder in targetDir which contains the images
        Returns:
            stdout: stdout
            stderr: stderr
            returncode: exitcode
    '''
    imgs_dir = os.path.join(targetDir, folderName)
    command = 'ffmpeg -framerate {2} -f image2 -i {0} -c:v libx264 -pix_fmt yuv420p -r 25 {1} -y'.format(  \
            os.path.join(imgs_dir,'%d.png'),                                                \
            os.path.join(targetDir, '%s.mp4'%folderName),                                   \
            frameRate
            )
    print('Executing command: ', command)
    _, stderr, returncode = sysutil.runCommand(command)
    if returncode!=0:
        print("ERROR happened during making visRecon video:\n error code:%d"%returncode, stderr)

def imgs2video2(imgs_dir, out_path, frameRate=6, ind_pattern="%d.png", verbose=False):
    ''' Making video from a sequence of images
    
        Make a video from images with index, e.g. 1.png 2.png 3.png ...
        the images should be in imgs_dir
        the output will be at out_path
        Args:
            targetDir: the output directory
            folderName: a folder in targetDir which contains the images
        Returns:
            stdout: stdout
            stderr: stderr
            returncode: exitcode
    '''
    
    #vid_filter = '-vf "colorchannelmixer=aa=1"'
    vid_filter = ''
    command='ffmpeg -framerate {framerate} -f image2 -i {imgs} -c:v libx264 -pix_fmt yuv420p -r 25 {outpath} {vid_filter} -y'
    command = command.format(outpath=out_path,imgs=os.path.join(imgs_dir,ind_pattern),framerate=frameRate, vid_filter=vid_filter)
    if verbose == True:
        print('Executing command: ', command)
    _, stderr, returncode = sysutil.runCommand(command)
    if returncode!=0:
        print("ERROR happened during making video:\n error code:%d"%returncode, stderr)

def imgarray2video(targetPath, img_list, frameRate=6, duration = 4., extend_endframes=0, save_imgs=False):
    ''' Making video from a sequence of images '''
    if duration is not None:
        frameRate = len(img_list)/duration
    # convert rgba images to rgb
    if type(img_list[0]) is np.ndarray and img_list[0].shape[2]==4:
        #img_list = [skimage.color.rgba2rgb(img) for img in img_list]
        #img_list = [img[:,:,:3] for img in img_list]
        pass

    import random, string
    o_temp_dir = os.path.expanduser('~/.temp/xgutils/imgarray2video/')
    
    if targetPath is None:
        # generate random name
        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        targetPath = os.path.join(o_temp_dir, name + '.mp4')

    # generate a random suffix

    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    temp_dir = o_temp_dir + "_" + suffix
    n_attempts = 0
    while os.path.exists(temp_dir) and n_attempts < 100:
        suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        temp_dir = o_temp_dir + "_" + suffix
        n_attempts += 1
    if n_attempts >= 100:
        raise Exception("Failed to create temp dir")

    try:
        end_frame = img_list[-1]
        for i in range(extend_endframes):
            img_list.append(end_frame)
        if save_imgs==True:
            target_imgs_dir = targetPath.replace('.mp4', '_imgs')
            sysutil.mkdirs([target_imgs_dir])
            saveImgs(targetDir=target_imgs_dir, baseName='', imgs=img_list)
        for i, img in enumerate(img_list):
            if img_list[i].shape[-1]==4:
                img_list[i] = skimage.color.rgba2rgb(img)
                
        saveImgs(targetDir=temp_dir, baseName='', imgs=img_list)
        imgs2video2(temp_dir, targetPath, frameRate=frameRate)
    finally:
        os.system('rm -r %s'%temp_dir)
    return targetPath

if os.path.exists('/tmp/tstx')==False:
    os.mkdir('/tmp/tstx')

from skimage.transform import resize as skiresize
def imageGrid_html5(imgs, img_names, target_path, height=400, columns=5, mode="image", img_size=None, use_multiprocessing=True):
    if target_path.endswith("/"):
        target_path = target_path[:-1]

    basename = os.path.basename(target_path)
    if img_names is None:
        img_names = [str(i) for i in range(len(imgs))]
    # if type(imgs[0]) is str:
    #     img_paths = imgs
    #     imgs = []
    #     print("Reading images...")
    #     for img_path in sysutil.progbar(img_paths):
    #         imgs.append(readImg(img_path))
    # if img_size is not None:
    #     # resize all images with scipy.zoom
    #     imgs = [skiresize(img, img_size, anti_aliasing=True) for img in imgs]
    # Get the list of image filenames
    print("basename", basename)
    tar_imgdir = target_path + f"/{basename}/"
    sysutil.mkdirs([target_path, tar_imgdir])
    image_filenames = img_names

    import glob
    if mode == "video":
        out_suffix = "mp4"
    elif mode == "image":
        out_suffix = "png"
    # elif mode == "gif":
    #     suffix = "mp4"
    #     oimage_filenames = glob.glob(image_folder_path+f"/*.{suffix}")
   
    image_filenames = []
    def process_img(img, index):
        #img = imgs[index]

        if type(img) is str:
            img = readImg(img)
        if img_size is not None:
            img = skiresize(img, img_size, anti_aliasing=True)
        imgname = f"{img_names[index]}.{out_suffix}"
        imgabspath = os.path.join(tar_imgdir, imgname)
        imgrelpath = os.path.join(basename, imgname)
        saveImg(imgabspath, img)
        return imgrelpath
    if use_multiprocessing:
        image_filenames = sysutil.pmap(process_img, [imgs, list(range(len(imgs)))], zippedIn=False, zippedOut=False, cores=4)[0] # too many cores won't help since IO is the bottleneck
    else:
        for i, img in sysutil.progbar(enumerate(imgs)):
            image_filenames.append(process_img(img, i))

    #image_filenames = mp.map(process_img, list(range(len(imgs))) )
    # Open or create HTML file
    with open(f'{target_path}/{basename}.html', 'w') as html_file:
        # Write the HTML header
        html_file.write('<!DOCTYPE html>\n')
        html_file.write('<html lang="en">\n')
        html_file.write('<head>\n')
        html_file.write('    <meta charset="UTF-8">\n')
        html_file.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        html_file.write('    <title>Image Gallery</title>\n')
        html_file.write('    <style>\n')
        html_file.write('       figure {\n')
        html_file.write('           display: inline-block;\n')
        html_file.write('           text-align: center;\n')
        html_file.write('           margin: 2px; \n')
        html_file.write('       }\n')
        html_file.write('       figcaption {\n')
        html_file.write('           text-align: center;\n')
        html_file.write('       }\n')
        html_file.write('    </style>\n')
        html_file.write('</head>\n')
        html_file.write('<body>\n')
        html_file.write('    <table>\n')

        # Iterate through the images and create table rows
        img_per_row = columns
        for row in range(0, len(image_filenames), img_per_row):  # 4 images per row
            html_file.write('        <tr>\n')
            for col in range(row, min(row + img_per_row, len(image_filenames))):
                html_file.write(f'            <td style = "border-right: 1px dashed green;"><div>')
                img_path = image_filenames[col]
                if img_path.endswith(".gif") or img_path.endswith(".png"):
                    html_file.write(f'<img src="{img_path}" alt="Image {col}" height="{height}">')
                elif img_path.endswith(".mp4"):
                    html_file.write(f'<video height="{height}" controls autoplay loop muted><source src="{img_path}" type="video/mp4">Your browser does not support the video tag.</video>')
                caption = os.path.basename(img_path).split(".")[0]
                html_file.write(f"<figcaption>{caption}</figcaption>")
                html_file.write(f'</div></td>\n')
            html_file.write('        </tr>\n')

        # Write the HTML footer
        html_file.write('    </table>\n')
        html_file.write('</body>\n')
        html_file.write('</html>\n')

def visGrid_html5(html_path, num_rows, num_columns): # TODO
    # Open or create HTML file
    with open(f'html_path', 'w') as html_file:
        # Write the HTML header
        html_file.write('<!DOCTYPE html>\n')
        html_file.write('<html lang="en">\n')
        html_file.write('<head>\n')
        html_file.write('    <meta charset="UTF-8">\n')
        html_file.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        html_file.write('    <title>Image Gallery</title>\n')
        html_file.write('    <style>\n')
        html_file.write('       figure {\n')
        html_file.write('           display: inline-block;\n')
        html_file.write('           text-align: center;\n')
        html_file.write('           margin: 2px; \n')
        html_file.write('       }\n')
        html_file.write('       figcaption {\n')
        html_file.write('           text-align: center;\n')
        html_file.write('       }\n')
        html_file.write('    </style>\n')
        html_file.write('</head>\n')
        html_file.write('<body>\n')
        html_file.write('    <table>\n')

        # Iterate through the images and create table rows
        img_per_row = columns
        for row in range(0, len(image_filenames), img_per_row):  # 4 images per row
            html_file.write('        <tr>\n')
            for col in range(row, min(row + img_per_row, len(image_filenames))):
                html_file.write(f'            <td style = "border-right: 1px dashed green;"><div>')
                img_path = image_filenames[col]
                if img_path.endswith(".gif") or img_path.endswith(".png"):
                    html_file.write(f'<img src="{img_path}" alt="Image {col}" height="{height}">')
                elif img_path.endswith(".mp4"):
                    html_file.write(f'<video height="{height}" controls autoplay loop muted><source src="{img_path}" type="video/mp4">Your browser does not support the video tag.</video>')
                caption = os.path.basename(img_path)
                html_file.write(f"<figcaption>{caption}</figcaption>")
                html_file.write(f'</div></td>\n')
            html_file.write('        </tr>\n')

        # Write the HTML footer
        html_file.write('    </table>\n')
        html_file.write('</body>\n')
        html_file.write('</html>\n')


def html5_result_vis(image_folder_path, target_dir, height=400, title="Results"):
    """
        list all visuals in a folder and group them by the first number in the filename
    """
    # Get the list of image filenames
    tar_img_folder = target_dir+"/imgs/"
    sysutil.mkdirs([image_folder_path, tar_img_folder])
    image_filenames = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

    import glob
    oimage_filenames = glob.glob(image_folder_path+f"/*.*")
    img_per_row = len(glob.glob(image_folder_path+f"/0_*.*"))
    print("img_per_row", img_per_row)
    # elif mode == "gif":
    #     suffix = "mp4"
    #     oimage_filenames = glob.glob(image_folder_path+f"/*.{suffix}")
   
    image_filenames = []
    for fpath in oimage_filenames:
        if "summary" in fpath:
            continue
        # copy to tar_img_folder
        fname = os.path.basename(fpath)
        # check if is mp4 file, if in gif mode, convert .mp4 to .gif
        copyto = tar_img_folder+fname
        os.system(f"cp {fpath} {tar_img_folder}")
        image_filenames.append("imgs/"+fname)

    # sort by the first number in the filename, if same number, sort by rest of the filename
    image_filenames = sorted(image_filenames, key=lambda x: (int(x.split("/")[-1].split("_")[0]), "_".join(x.split("/")[-1].split("_")[1:])))

    # Open or create HTML file
    with open(f'{target_dir}/images_table.html', 'w') as html_file:
        # Write the HTML header
        html_file.write('<!DOCTYPE html>\n')
        html_file.write('<html lang="en">\n')
        html_file.write('<head>\n')
        html_file.write('    <meta charset="UTF-8">\n')
        html_file.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        html_file.write(f'    <title>{title}</title>\n')
        html_file.write('    <style>\n')
        html_file.write('       figure {\n')
        html_file.write('           display: inline-block;\n')
        html_file.write('           text-align: center;\n')
        html_file.write('           margin: 2px; \n')
        html_file.write('       }\n')
        html_file.write('       figcaption {\n')
        html_file.write('           text-align: center;\n')
        html_file.write('       }\n')
        html_file.write('    </style>\n')
        html_file.write('</head>\n')
        html_file.write('<body>\n')
        html_file.write('    <table>\n')

        # Iterate through the images and create table rows
        for row in range(0, len(image_filenames), img_per_row):  # 4 images per row
            html_file.write('        <tr>\n')
            for col in range(row, min(row + img_per_row, len(image_filenames))):
                html_file.write(f'            <td style = "border-right: 1px dashed green;"><div>')
                img_path = image_filenames[col]
                if img_path.endswith(".gif") or img_path.endswith(".png"):
                    html_file.write(f'<img src="{img_path}" alt="Image {col}" height="{height}">')
                elif img_path.endswith(".mp4"):
                    html_file.write(f'<video height="{height}" controls autoplay loop muted><source src="{img_path}" type="video/mp4">Your browser does not support the video tag.</video>')
                caption = os.path.basename(img_path)
                html_file.write(f"<figcaption>{caption}</figcaption>")
                html_file.write(f'</div></td>\n')
            html_file.write('        </tr>\n')

        # Write the HTML footer
        html_file.write('    </table>\n')
        html_file.write('</body>\n')
        html_file.write('</html>\n')

    # os.chdir("/studio/")
    # # Path to the directory containing images
    # image_folder_path = "dl_template/experiments/v31_cos-schedule_condpolyfusion_onshape/results/LoopfusionVisualizer/visual"
    # html5_imgGrid(image_folder_path, "/studio/dl_template/temp/v31_cos_polyloopfusion_gif/", columns=5, height=300, mode="gif")


def showVidNotebook(video_path=None, imgs=None, duration=4, loopback=False, targetPath=None):
    from IPython.core.display import display, HTML

    if video_path is None:
        img_list = imgs
        if loopback==True:
            img_list = imgs + imgs[::-1]
        video_path = imgarray2video(targetPath=targetPath, img_list=img_list, duration=duration, save_imgs=True)
    
    import base64 # thanks to chatgpt, we find a way to show video of arbitrary path
    video_encoded = base64.b64encode(open(video_path, "rb").read()).decode("ascii")

    # Create HTML string
    video_html = f"""
    <video width="640" height="480" controls autoplay muted loop>
        <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4">
    </video>
    """
    # # Create an HTML string to embed the video
    # video_html = f"""
    # <video width="320" height="240" controls>
    # <source src="{video_path}" type="video/mp4">
    # </video>
    # """

    # Display the video
    display(HTML(video_html))
    return video_path


################### Image Processing ###################
def binary2rgba(mask, color=np.array([0,0,0]), alpha=.8):
    rgba = np.stack([color]*mask.shape[0], axis=0)



################## snippets collections ##################
def shapeformer_nice_plot():
    fig, ax = visutil.newPlot(ticks=True, axes_off=False, resolution=(400,400), tight=False)
    sns.set_context('paper', font_scale=1.5)
    plt.style.use('seaborn-colorblind')
    ax.grid() 

    ax.set_xscale("log", basex=2)
    ax.set_xticks([2**(-2), 2**2, 2**6, 2**11, 2**17])
    ax.set_xlim(2**(-3.5), 2**19)
    ax.set_xlabel("KBytes")
    #ax.set_ylim(1.3, 5.2)
    ax.set_ylabel("Chamfer $L_2$  $×10^{4}$")
    ax.plot(data[:,-1], data[:,0], "D-")
    ax.plot(wo_data[:,-1], wo_data[:,0], "o--")

    ax.plot([OccNet[-1]], [OccNet[0]], "o")
    ax.plot([ConvONet[-1]], [ConvONet[0]], "o")
    ax.plot([IFNet[-1]], [IFNet[0]], "o")

    ax.legend(["Ours", "w/o quant." ,"OccNet", "ConvONet", "IFNet"], loc=0, prop={'size': 13})
    plt.savefig("ChamferL2.pdf", bbox_inches='tight')
    plt.show()


