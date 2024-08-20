import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from xgutils import *
from xgutils.vis import fresnelvis, visutil
import einops
import copy

sample_cloudR = 0.02
vis_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
                camUp=np.array([0,1,0]),camHeight=2,resolution=(256,256), samples=32)
lowvis_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
                camUp=np.array([0,1,0]),camHeight=2,resolution=(128,128), samples=8)
HDvis_camera  = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
                camUp=np.array([0,1,0]),camHeight=2,resolution=(512,512), samples=256)

sample_cloudR = 0.02
pink_color = np.array([255, 0, 0])/256
gold_color = np.array([253, 204, 134])/256
gray_color = np.array([0.9, 0.9, 0.9])
result_color  = gray_color
partial_color = gold_color


def vis_VoxelXRay(voxel, axis=0, duration=10., target_path="/studio/temp/xray.mp4"):
    if(len(voxel.shape)==1):
        voxel = nputil.array2NDCube(voxel, N=3)
    imgs = []
    for i in sysutil.progbar(range(voxel.shape[axis])):
        nvox = voxel.copy()
        nvox[i:,:,:]=0
        voxv, voxf = geoutil.array2mesh(nvox.reshape(-1), thresh=.5, coords=nputil.makeGrid([-1,-1,-1],[1,1,1],[64,64,64], indexing="ij"))
        #voxv[]
        dflt_camera = fresnelvis.dflt_camera
        dflt_camera["camPos"]=np.array([2,2,2])
        dflt_camera["resolution"]=(256,256)
        img = fresnelvis.renderMeshCloud({"vert":voxv,"face":voxf}, **dflt_camera, axes=True)
        imgs.append(img)
    sysutil.imgarray2video(target_path, imgs, duration=duration)

def OctreePlot3D(tree, dim, depth, **kwargs):
    assert dim==2
    boxcenter, boxlen, tdepth = ptutil.ths2nps(ptutil.tree2bboxes(torch.from_numpy(tree), dim=dim, depth=depth))    
    maxdep = tdepth.max()
    renderer = fresnelvis.FresnelRenderer(camera_kwargs=dict(camPos=np.array([1.5,2,2]), resolution=(1024,1024)))#.add_mesh({"vert":vert, "face":face})
    for i in range(len(tdepth)):
        dep=tdepth[i]
        length = boxlen[i]
        bb_min = boxcenter[i]-boxlen[i]
        bb_max = boxcenter[i]+boxlen[i]
        lw=1+.5*np.exp(-dep)
        #rect = patches.Rectangle(corner, 2*length, 2*length, linewidth=lw, edgecolor=plt.cm.plasma(dep/maxdep), facecolor='none')
        renderer.add_bbox(bb_min=bb_min, bb_max=bb_max, color=plt.cm.plasma(dep/maxdep)[:3], radius=0.001*dep**1.5, solid=.0)
    img = renderer.render()
    return img

#def CloudPlot

def SparseVoxelPlot(sparse_voxel, depth=4, varying_color=False, camera_kwargs=dict(camPos=np.array([2,2,2]), resolution=(512,512))):
    resolution = camera_kwargs["resolution"]
    if len(sparse_voxel)==0:
        return np.zeros((resolution[0], resolution[1], 3))
    grid_dim = 2**depth
    box_len  = 2/grid_dim/2

    renderer = fresnelvis.FresnelRenderer(camera_kwargs=camera_kwargs)#.add_mesh({"vert":vert, "face":face})
    voxel_inds   = ptutil.unravel_index( torch.from_numpy(sparse_voxel), shape=(2**depth,)*3 )
    voxel_coords = ptutil.ths2nps(ptutil.index2point(voxel_inds, grid_dim=grid_dim))

    color = fresnelvis.gray_color
    percentage = np.arange(len(voxel_coords)) / len(voxel_coords)
    if varying_color==True:
        color = plt.cm.plasma(percentage)[...,:3]
    renderer.add_box(center=voxel_coords, spec=np.zeros((3))+box_len, color=color, solid=0.)
    img = renderer.render()
    return img
acc_unis = []
def IndexVoxelPlot(pos_ind, val_ind, val_max=1024, depth=4, 
        manual_color=None, distinctive_color=True, 
        camera_kwargs=dict(camPos=np.array([2,2,2]), resolution=(512,512)),
        **kwargs
        ):
    resolution = camera_kwargs["resolution"]
    if len(pos_ind)==0:
        return np.zeros((resolution[0], resolution[1], 3))
    grid_dim = 2**depth
    box_len  = 2/grid_dim/2

    renderer = fresnelvis.FresnelRenderer(camera_kwargs=camera_kwargs)#.add_mesh({"vert":vert, "face":face})
    voxel_inds   = ptutil.unravel_index( torch.from_numpy(pos_ind), shape=(2**depth,)*3 )
    voxel_coords = ptutil.ths2nps(ptutil.index2point(voxel_inds, grid_dim=grid_dim))
    if distinctive_color == False:
        color = plt.cm.Blues(val_ind/val_max)[..., :3]
    else:
        unique, inverse = np.unique(val_ind, return_inverse=True)
        acc_unis.append(unique)
        acc_uni_n = np.unique( np.concatenate(acc_unis) ).shape[0]
        print(acc_uni_n)
        print("num total uniques", acc_uni_n)
        print("num uniques", unique.shape[0])
        color = plt.cm.Blues(inverse/unique.shape[0])[..., :3]
    if manual_color is not None:
        color = manual_color
    renderer.add_box(center=voxel_coords, spec=np.zeros((3))+box_len, color=color, solid=0., **kwargs)
    img = renderer.render()
    return img
def CubePlot(coords, size, color=None, cmap = "Blues", 
            camera_kwargs=dict(camPos=np.array([2,2,2]), resolution=(512,512)),
            renderer = None,
            **kwargs
        ):
    resolution = camera_kwargs["resolution"]
    if coords.shape[0]==0:
        return np.zeros((resolution[0], resolution[1], 3))
    if renderer is None:
        renderer = fresnelvis.FresnelRenderer(camera_kwargs=camera_kwargs)#.add_mesh({"vert":vert, "face":face})
    if color is None:
        color = np.zeros(coords.shape[0])
    if len(color.shape)==1:
        color = plt.get_cmap(cmap)(color)[..., :3]
    renderer.add_box(center=coords, spec=np.zeros((3))+size, color=color, solid=0., **kwargs)
    img = renderer.render()
    return img
def BBoxPlot(center, scale, color=None, cmap = "Blues"):
    bbox_num=  center.shape[0]
    if color is None:
        #color = fresnelvis.gray_color
        # unique colors
        color = fresnelvis.unique_colors[:bbox_num][..., :3]
        print("color", color.shape)
    renderer = fresnelvis.FresnelRenderer(camera_kwargs=dict(camPos=np.array([2,2,2]), resolution=(512,512)))#.add_mesh({"vert":vert, "face":face})
    renderer.addBox(bb_min=center-scale, bb_max=center+scale, color=color, radius=0.001, solid=.0)
    renderer.add_cloud(center, radius=0.02, color=color)
    img = renderer.render(preview=True)
    return img

def meshCloudPlot(Ytg, Xbd):
    vert, face = geoutil.array2mesh(Ytg.reshape(-1), thresh=.5, coords=nputil.makeGrid([-1,-1,-1],[1,1,1],[64,64,64], indexing="ij"))
    
    vis_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
            camUp=np.array([0,1,0]),camHeight=2,resolution=(512,512), samples=16)
    img = fresnelvis.renderMeshCloud({"vert":vert, "face":face}, cloud=Xbd, cloudR=.01, **vis_camera)
    return img

import einops
def plot_volume(volume, show=True, contour_mode=True, slice_dim=1, zoom=3):
    # volume: (dim,dim,dim) 
    cdim = volume.shape[0]
    vis_dim = int(np.ceil(np.sqrt(cdim)))
    pixel_n = cdim*cdim*vis_dim*vis_dim
    pixels = np.zeros(pixel_n)
    if slice_dim==0:
        fvtrans = einops.rearrange(volume, "x y z -> (x y z)")
        pixels[:fvtrans.shape[0]] = fvtrans # [X Y Z] -> [ (X Y Z) ] -> [ X*Y*Z+padd ]
        rear = einops.rearrange(pixels, "(n m y z) -> (n y) (m z)", y=cdim, z=cdim, n=vis_dim, m = vis_dim)
    if slice_dim==1:
        fvtrans = einops.rearrange(volume, "x y z -> (y x z)")
        pixels[:fvtrans.shape[0]] = fvtrans # [X Y Z] -> [ (X Y Z) ] -> [ X*Y*Z+padd ]
        rear = einops.rearrange(pixels, "(n m x z) -> (n x) (m z)", x=cdim, z=cdim, n=vis_dim, m = vis_dim)
    if slice_dim==2:
        fvtrans = einops.rearrange(volume, "x y z -> (z y x)")
        pixels[:fvtrans.shape[0]] = fvtrans # [X Y Z] -> [ (X Y Z) ] -> [ X*Y*Z+padd ]
        rear = einops.rearrange(pixels, "(n m y x) -> (n y) (m x)", x=cdim, y=cdim, n=vis_dim, m = vis_dim)
    if zoom>1:
        rear = einops.rearrange(rear, "(n x) (m y) -> (n m) 1 x y", x=cdim, y=cdim, n=vis_dim, m = vis_dim)
        rear = torch.from_numpy(rear)
        rear = torch.nn.functional.interpolate(rear, scale_factor=zoom, mode="bilinear", align_corners=True)
        rear = einops.rearrange(rear, "(n m) 1 x y -> (n x) (m y)", n=vis_dim, m = vis_dim).numpy()
        
        #scipy.ndimage.zoom(rear, zoom_factor, order=1)
 
    if show==True:
        fig, ax = visutil.SDF2DPlot(gridData=rear, im_origin="upper", contour_mode=contour_mode, indexing="xy", zoom=zoom)
        # plt.imshow(rear) # simpler version without extra dependencies
        img = visutil.fig2img(fig)
        visutil.showImg(img)
    return rear

def render_polyloop_explosion(polyloops, radius=0.01, loop_vid=False, extent=[0.,.2], frames=16, camera_kwargs=fresnelvis.dflt_camera, show_face=False, camera_trajectory=None, **kwargs):
    imgs = []
    render_mode = ["vert", "edge"]
    if show_face==True:
        render_mode.append("face")
    camera_trajectory = parse_camera_trajectory(camera_trajectory, frames=frames, camera_kwargs=camera_kwargs)

    ext = np.linspace(extent[0], extent[1], frames)
    for ei, exi in sysutil.progbar(enumerate(ext)):
        #renderer = fresnelvis.FresnelRenderer(camera_kwargs=kwargs)
        epolyloops = geoutil.explode_polyloops(polyloops, exi, **kwargs)
        v, f = geoutil.polyloops2vf(epolyloops)
        if camera_trajectory is not None:
            camera_kwargs = camera_trajectory[ei]
        img = fresnelvis.render_polyloop(v, f, modes=render_mode, camera_kwargs=camera_kwargs, show_axes=True, show_bbox=True, preview=True)
        imgs.append(img)
    if loop_vid==True:
        imgs = imgs + imgs[::-1]
    return imgs

def generate_lemniscate_trajectory(camera_kwargs=fresnelvis.dfltw_camera, scale = 1., frames=32):
    trans, rot = fresnelvis.get_world2cam(camera_kwargs["camPos"], camera_kwargs["camLookat"], camera_kwargs["camUp"])
    t = np.linspace(0, 2 * np.pi, frames)

    # Lemniscate parametric equations
    lemniscate = np.zeros((frames, 3))
    lemniscate[:, 0] = np.cos(t) * scale
    lemniscate[:, 1] = np.sin(t) * np.cos(t) * scale

    # Rotate the lemniscate to the camera orientation using matmult
    trajectory = np.matmul(lemniscate, rot)
    trajectory = trajectory + trans[None, :]

    traj_camera_kwargs = []
    for i in range(len(trajectory)):
        ckwargs = copy.deepcopy(camera_kwargs)
        ckwargs["camPos"] = trajectory[i]
        ckwargs["camHeight"] = 3
        traj_camera_kwargs.append(ckwargs)

    return traj_camera_kwargs
def generate_circular_trajectory(camera_kwargs=fresnelvis.dfltw_camera, scale = 1., frames=32):
    trans, rot = fresnelvis.get_world2cam(camera_kwargs["camPos"], camera_kwargs["camLookat"], camera_kwargs["camUp"])
    t = np.linspace(0, 2 * np.pi, frames)

    # Lemniscate parametric equations
    lemniscate = np.zeros((frames, 3))
    lemniscate[:, 0] = np.cos(t) * scale
    lemniscate[:, 1] = np.sin(t) * scale

    # Rotate the lemniscate to the camera orientation using matmult
    trajectory = np.matmul(lemniscate, rot)
    trajectory = trajectory + trans[None, :]

    traj_camera_kwargs = []
    for i in range(len(trajectory)):
        ckwargs = copy.deepcopy(camera_kwargs)
        ckwargs["camPos"] = trajectory[i]
        ckwargs["camHeight"] = 3
        traj_camera_kwargs.append(ckwargs)

    return traj_camera_kwargs

def generate_round_trajectory(camera_kwargs=fresnelvis.dfltw_camera, scale = 1., frames=32):
    trans, rot = fresnelvis.get_world2cam(camera_kwargs["camPos"], camera_kwargs["camLookat"], camera_kwargs["camUp"])
    xy = (camera_kwargs["camPos"] - camera_kwargs["camLookat"])[[0,2]]
    mag = np.linalg.norm(xy)
    phase = np.arctan2(xy[1], xy[0])
    center = camera_kwargs["camLookat"][[0,2]]

    t = np.linspace(0, 2 * np.pi, frames)
    trajectory = np.zeros((frames, 3))
    trajectory[:, 0] = np.cos(phase + t) * mag + center[0]
    trajectory[:, 2] = np.sin(phase + t) * mag + center[1]
    trajectory[:, 1] = camera_kwargs["camPos"][1]
    traj_camera_kwargs = []
    for i in range(len(trajectory)):
        ckwargs = copy.deepcopy(camera_kwargs)
        ckwargs["camPos"] = trajectory[i]
        traj_camera_kwargs.append(ckwargs)

    return traj_camera_kwargs

    # # Test
    # pts = np.random.rand(128,3)*2 - 1
    # trajs = vis3d.generate_round_trajectory(scale=2, frames=64)
    # imgs = vis3d.render_camera_trajectory(func=fresnelvis.render_cloud, func_kwargs=dict(cloud=pts, radius=.05, show_axes=True, show_bbox=True, render_kwargs=dict(preview=True)), camera_trajectory=trajs)
    # visutil.showVidNotebook(imgs=imgs, duration=4)
def parse_camera_trajectory(camera_trajectory, frames=32, camera_kwargs=fresnelvis.dfltw_camera):
    ckwargs = copy.deepcopy(fresnelvis.dfltw_camera)
    ckwargs.update(camera_kwargs)
    if type(camera_trajectory) is str:
        if camera_trajectory=="lemniscate":
            camera_trajectory = generate_lemniscate_trajectory(scale=1, frames=frames, camera_kwargs=ckwargs)
        elif camera_trajectory=="circular":
            camera_trajectory = generate_circular_trajectory(scale=1,   frames=frames, camera_kwargs=ckwargs)
        elif camera_trajectory=="round":
            camera_trajectory = generate_round_trajectory(scale=1,  frames=frames, camera_kwargs=ckwargs)
    return camera_trajectory
def render_camera_trajectory(func, func_kwargs, camera_trajectory, frames=32):
    imgs = []
    if type(camera_trajectory) is str:
        if camera_trajectory=="lemniscate":
            camera_trajectory = generate_lemniscate_trajectory(scale=1, frames=frames)
        elif camera_trajectory=="circular":
            camera_trajectory = generate_circular_trajectory(scale=1,  frames=frames)
        elif camera_trajectory=="round":
            camera_trajectory = generate_round_trajectory(scale=1, frames=frames)

    for i in sysutil.progbar(range(len(camera_trajectory))):
        camera_kwargs = camera_trajectory[i]
        kwargs = copy.deepcopy(func_kwargs)
        kwargs["camera_kwargs"] = camera_kwargs
        img = func(**kwargs)
        imgs.append(img)
    return imgs

    # unit test
    # camera_kwargs = copy.deepcopy(fresnelvis.dflt_camera)
    # camera_kwargs["camPos"]    = np.array([.0,.5,.5])
    # imgs = []
    # for xi in np.linspace(-1,1,32):
    #     camera_kwargs["camLookat"] = np.array([0,xi,0])
    #     #camera_kwargs["camUp"] = np.array([0,0,1.])
    #     trajectory = generate_lemniscate_trajectory(camera_kwargs)
    #     imgs.append(fresnelvis.render_cloud(trajectory, radius=.05, camera_kwargs=dict(camHeight=3, camPos=(4,4,4.)), show_axes=True, show_bbox=True) )
    # visutil.showVidNotebook(imgs=imgs, duration=6)
