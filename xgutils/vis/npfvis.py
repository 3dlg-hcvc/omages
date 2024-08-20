import numpy as np
import scipy
import matplotlib.pyplot as plt
from xgutils import *
from xgutils.vis import fresnelvis, visutil
def plot_1d_samples(Xct:np.ndarray, Yct:np.ndarray, Xtg:np.ndarray, Ytg:np.ndarray=None, pred_ys:np.ndarray=None):
    """plot 1d 

    Args:
        Xct (np.ndarray): context x (n_cntxt, dim_x)
        Yct (np.ndarray): context y (n_cntxt, dim_y)
        Xtg (np.ndarray): target x  (n_trgt, dim_x)
        Ytg (np.ndarray, optional): target_y. (n_trgt, dim_y),    Defaults to None.
        pred_ys (np.ndarray, optional): pred_ys. (n_samples, n_trgt, dim_y), Defaults to None.
    """
    num_target, dim_x = Xct.shape
    num_context = Xct.shape[0]
    n_samples   = pred_ys.shape[0]
    figs = []
    plots, legend = [], []
    if num_context>0:
        train = plt.plot(Xct, Yct, 'ro')
        plots.append(train)
        legend.append('Context Points')
    ground_truth = plt.plot(Xtg, Ytg)
    plots.append(ground_truth)
    legend.append('True Function')
    for ind in range(n_samples):
        cur = plt.plot(Xtg[:,0], pred_ys[ind][:, 0], '--')
        plots.append(cur)
        #legend.append('After %d Steps'%ind)
    plt.legend(plots, legend)
    limMax = Xtg.max()
    limMin = Xtg.min()
    Xlen = Xtg.max() - Xtg.min()
    plt.ylim(-1.2, 1.2)
    plt.xlim(limMin, limMax)
    img = visutil.fig2img(plt.gcf(), closefig=True)
    return img
def plot_2d_sample(Xtg, Ytg, Xct=None, style='-', scatter_size=1, scatter_c="red", zorder=2, ax=None, return_img=False):
    if ax is None:
        fig, ax = visutil.newPlot(plotrange=np.array([[-1.,1.],[-1.,1.]]))
    else:
        fig = ax.figure
    v, e = geoutil.array2mesh(Ytg, dim=2, coords=Xtg, thresh=.5)
    if v is None:
        v = np.zeros((0, Xtg.shape[-1]))
    if e is None:
        e = np.zeros((0, 2))
    if len(v)>0 and len(e)>0:
        num, label =    scipy.sparse.csgraph.connected_components(\
                            scipy.sparse.coo_matrix((np.ones(e.shape[0],dtype=int),e.T),shape=(v.shape[0],)*2).tocsr(), \
                            directed=False)
        for comp_i in range(num):
            vert_ind = np.where(label==comp_i)[0]
            vert_ind = np.r_[vert_ind,vert_ind[0]]
            fig, ax = visutil.linePlot(v[vert_ind], plotrange=np.array([[-1.,1.],[-1.,1.]]), ax=ax, style=style, c_ind=zorder)
    if Xct is not None:
        ax.scatter(Xct[:,0], Xct[:,1], c=scatter_c, s=scatter_size, zorder=zorder+1)

    if return_img==True:
        return visutil.fig2img(fig)
    else:
        return fig, ax
def plot_2d_samples(Xct:np.ndarray, Yct:np.ndarray, Xtg:np.ndarray, Ytg:np.ndarray=None, pred_ys:np.ndarray=None):
    """plot 2d 

    Args:
        Xct (np.ndarray): context x (n_cntxt, dim_x)
        Yct (np.ndarray): context y (n_cntxt, dim_y)
        Xtg (np.ndarray): target x  (n_trgt, dim_x)
        Ytg (np.ndarray, optional): target_y. (n_trgt, dim_y),    Defaults to None.
        pred_ys (np.ndarray, optional): pred_ys. (n_samples, n_trgt, dim_y), Defaults to None.
    """
    num_target, dim_x = Xct.shape
    num_context = Xct.shape[0]
    n_samples   = pred_ys.shape[0]

    fig, ax = visutil.newPlot(axes_off=True)
    ax.scatter(Xct[:,0], Xct[:,1], c='red', zorder=n_samples+1)
    ys = [(pred_ys[i][...,0], '-') for i in range(n_samples)]
    ys.append((Ytg[:,0], ':k'))
    for i in range(len(ys)):
        fig, ax = plot_2d_sample(Xtg, ys[i][0], style=ys[i][1], zorder = i, ax=ax)
    boxes_img = visutil.fig2img(fig)
    return boxes_img

def plot_3d_recon(Xtg, Ytg, if_decimate=False, camera_kwargs=dict(), meshC=fresnelvis.gray_color, samples=None, return_mesh=False):
    dflt_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
        camUp=np.array([0,1,0]),camHeight=2.414,resolution=(256,256))
    camera_kwargs = sysutil.dictUpdate(dflt_camera, camera_kwargs)

    vert, face = geoutil.array2mesh(Ytg, dim=3, coords=Xtg, thresh=.5, if_decimate=if_decimate)
    img = fresnelvis.renderMeshCloud(mesh={'vert':vert, 'face':face}, meshC=meshC, **camera_kwargs)
    if return_mesh==False:
        return img
    else:
        return img, vert, face
def plot_3d_sample(Xct:np.ndarray, Yct, Xtg, Ytg:np.ndarray=None, pred_y:np.ndarray=None, \
            cloudR=0.01, camera_kwargs=dict(), show_images=['GT','pred','context']):
    context_x, context_y, target_x, target_y, pred_target_y = Xct, Yct, Xtg, Ytg, pred_y
    dflt_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
        camUp=np.array([0,1,0]),camHeight=2.414,resolution=(256,256))
    camera_kwargs = sysutil.dictUpdate(dflt_camera, camera_kwargs)
    shape, face, vert, gtshape = None, None, None, None

    cerror_scaled = None
    imgs = []
    # plot gt shape
    if target_y is not None:
        vert, face = geoutil.array2mesh(Ytg, dim=3, coords=Xtg, thresh=.5, if_decimate=True)
        gtmesh = fresnelvis.renderMeshCloud(mesh={'vert':vert, 'face':face}, cloud=context_x, cloudC=None, cloudR=cloudR, **camera_kwargs)
        if 'GT' in show_images:
            imgs.append(gtmesh)
    if pred_target_y is not None:
        vert, face = geoutil.array2mesh(pred_target_y, dim=3, coords=Xtg, thresh=.5, if_decimate=True)
        shape = {'vert':vert, 'face':face}
        if context_x.shape[0] > 0:
            if vert.shape[0]>10 and face.shape[0]>10:
                context_error = np.abs(geoutil.signed_distance(context_x, vert, face)[0] ) #- context_y[:,0] )
            else:
                context_error = np.zeros((context_x.shape[0]))+10
            cloud = context_x
            cerror_scaled = context_error * 10
        else:
            cloud = None
            cerror_scaled = None
        meshcloud = fresnelvis.renderMeshCloud(mesh=shape, 
                cloud=cloud, cloudC=cerror_scaled, cloudR=cloudR, 
                **camera_kwargs)
        if 'pred' in show_images:
            imgs.append(meshcloud)
    #target_error = np.abs(target_y - pred_target_y)
    #target_grid = nputil.array2NDCube(array, N=3)
    #tree = cKDTree(target_x)
    #dist, ind = tree.query(vert, k = 1)
    #mesh_error = target_error[ind]
    #context_error = np.abs(context_y - pred_context_y)

    if context_x is not None:
        cloud = fresnelvis.renderMeshCloud(mesh=None,
                cloud=context_x, cloudC=cerror_scaled, cloudR=cloudR,
                **camera_kwargs)
        if 'context' in show_images:
            imgs.append(cloud)
    if len(imgs) == 1:
        imggrid=imgs[0]
    else:
        imggrid = visutil.imageGrid(imgs, shape=(1,len(imgs)))
    return imggrid, imgs

def plot_3d_samples(Xct:np.ndarray, Yct:np.ndarray, Xtg:np.ndarray, Ytg:np.ndarray=None, pred_ys:np.ndarray=None, \
        camera_kwargs={}, samples=5, return_list=False,
):    
    context_x, context_y, target_x, target_y, pred_target_ys = Xct, Yct, Xtg, Ytg, pred_ys
    imgs, pred_imgs, context_imgs  = [], [], []
    total_num = len(pred_target_ys)
    pred_target_ys_mean = pred_target_ys.mean(axis=0)
    datas = []
    _, (GT_img, predmean_img, contextmean_img) = plot_3d_sample(Xct, Yct, Xtg, Ytg, pred_target_ys_mean, \
        show_images = ['GT','pred','context'], camera_kwargs = camera_kwargs)
    
    if samples < total_num:
        choice = np.sort(np.random.choice(total_num, samples, replace=False))
        pred_target_ys  = pred_target_ys[choice]

    else:
        samples = total_num
        choice = np.ones(pred_target_ys.shape[0])
    #pred_context_ys = pred_context_ys[choice]
    
    for i in range(samples):
        #data['pred_context_y'] = pred_context_ys[i]
        #data['pred_target_y']  = pred_target_ys[i]
        
        show_images = ['pred', 'context']
        #out =   self.nnrecon3d( , show_images=show_images )
        _, (pred_img, context_img) = plot_3d_sample(Xct, Yct, Xtg, Ytg, pred_target_ys[i], \
        show_images = ['pred','context'], camera_kwargs=camera_kwargs)

        pred_imgs.append( pred_img )
        context_imgs.append( context_img )
    imglist = [GT_img, *pred_imgs, predmean_img, *context_imgs]
    if return_list==True:
        return imglist
    imgs = visutil.imageGrid(imglist, shape=(2,samples+1))
    return imgs
def plot_samples(Xct:np.ndarray, Yct:np.ndarray, Xtg:np.ndarray, Ytg:np.ndarray=None, pred_ys:np.ndarray=None, dim=1, **kwargs):
    plot_fns = [plot_1d_samples, plot_2d_samples, plot_3d_samples]
    return plot_fns[dim-1](Xct, Yct, Xtg, Ytg, pred_ys=pred_ys, **kwargs)
def plot_samples_pc(Xct:np.ndarray, Xbd:np.ndarray, Xtg:np.ndarray, Ytg:np.ndarray=None, pred_ys:np.ndarray=None, dim=1, **kwargs):
    plot_fns = [plot_1d_samples, plot_2d_samples, plot_3d_samples]
    Yct = np.zeros((*Xct.shape[:-1],1))
    return plot_fns[dim-1](Xct, Yct, Xtg, Ytg, pred_ys=pred_ys, **kwargs)

def plot_1d_samples_unittest():
    xtg=np.linspace(-1,1,100)[...,None]
    data=dict(Xct=np.array([[0]]), Yct=np.array([[0]]),
        Xtg=xtg, Ytg=np.zeros(100),
        pred_ys=np.array([xtg, xtg*2, xtg*3]))
    img= plot_1d_samples(**data)
    visutil.showImg(img)


