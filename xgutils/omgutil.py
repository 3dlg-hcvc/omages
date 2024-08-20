import skimage
import numpy as np
from xgutils import geoutil, nputil, sysutil
from xgutils.vis import plt3d, visutil

from xgutils.miscutil import preset_glb, preset_blend
PRESET_BLEND = preset_blend
# from xgutils import bpyutil

import os
import skimage
import scipy
import einops
#import torch

#import igl # some function requires igl
def omg2tensor(omage):
    objamage_tensor = np.concatenate([
        omage['position'][..., :3], omage['occupancy'][..., :1], omage['objnormal'][..., :3], 
        omage['color'][..., :3],
        omage['metal'][..., :1], omage['rough'][..., :1], 
        ], axis=-1)
    return objamage_tensor
def segment_omg(objamage_tensor, size_order=True):
    return segment_occ(objamage_tensor[..., 3], size_order=size_order)
def segment_occ(occupancy_map, size_order=True):
    occ_binary = occupancy_map>.5
    # if fully occupied, the lb should be all 1
    if occ_binary.all():
        lb = np.ones_like(occ_binary, int)
    else:
        lb = skimage.measure.label(occ_binary)
    if size_order:
        # sort the labels by patch size
        lb_sizes = np.bincount(lb.ravel())
        lb_sizes = np.argsort(lb_sizes[1:])[::-1]
        lb_sizes = np.concatenate([[0], lb_sizes+1])
        lb = lb_sizes[lb]
    return lb

def random_patch_mask( omage, ):
    lbs = segment_omg(omage)
    num_patch = lbs.max()
    shuffled_lb_ids = np.random.permutation(num_patch)
    keep_patch_num = np.random.randint(0, num_patch+1) 
    keep_mask = lbs.copy()
    keep_mask[lbs!=0] = shuffled_lb_ids[ lbs[lbs!=0] - 1 ] < keep_patch_num
    return keep_mask

def tensor2omg(objamage_tensor, seg_color_palette=None):
    if objamage_tensor.dtype == np.uint16:
        objamage_tensor = objamage_tensor.astype(np.float64) / 65535.
    position_map = objamage_tensor[..., :3]
    occupancy_map = objamage_tensor[..., 3:4]
    if objamage_tensor.shape[-1] == 12: # 
        objnormal_map = objamage_tensor[..., 4:7]
        color_map = objamage_tensor[..., 7:10]
        metal_map = objamage_tensor[..., 10:11]
        rough_map = objamage_tensor[..., 11:12]
    else: # if is position only
        objnormal_map = np.zeros((objamage_tensor.shape[0], objamage_tensor.shape[1], 3))
        color_map = np.zeros((objamage_tensor.shape[0], objamage_tensor.shape[1], 3))
        metal_map = np.zeros((objamage_tensor.shape[0], objamage_tensor.shape[1], 1))
        rough_map = np.zeros((objamage_tensor.shape[0], objamage_tensor.shape[1], 1))
    
    lb = skimage.measure.label(occupancy_map[...,0]>.5)
    segmentation_map = lb[...,None]
    if seg_color_palette is None:
        seg_color_palette = visutil.unique_colors
        unique2 = visutil.unique_colors.copy()
        with nputil.temp_seed(11):
            np.random.shuffle(unique2)
    lbc = seg_color_palette[lb]
    lbc[lb==0, :3] = [0,0,0]
    lbc = lbc[..., :3]
    
    omage = dict(position=position_map, occupancy=occupancy_map, objnormal=objnormal_map, color=color_map, metal=metal_map, rough=rough_map,
    segmentation = segmentation_map,
    segcolor = lbc,
    )
    return omage

def load_omg(path, resize=None, advanced_downsample=False, occ_thresh=.5, trim_omg=True, resize_order=0):
    """
    Load an omg file and return it as a numpy array.
    Args:
        path: str, path to the omg file.
        resize: int or tuple, resize the omg to the specified size.
        advanced_downsample: bool, whether to use advanced downsample method (edge snapping).
        occ_thresh: float, threshold for occupancy.
        trim_omg: bool, whether to trim the omg based on the occupancy threshold.
        resize_order: int, order of the resize function.
    """
    # if a npz is loaded, then omage = omage["arr_0"]
    if os.path.exists(path) == False:
        if path[-4:] == ".npy":
            path = path[:-4] + ".npz"
        elif path[-4:] == ".npz":
            path = path[:-4] + ".npy"
        if os.path.exists(path) == False:
            raise FileNotFoundError("File not found: "+path)
    if path[-4:] == ".npz":
        #if isinstance(omage, np.lib.npyio.NpzFile):
        omage = np.load(path, allow_pickle=True)
        omage = omage["arr_0"]
    else:
        omage = np.load(path, allow_pickle=True)
    # if omg is of uint16 type, convert to float
    if omage.dtype == np.uint16:
        omage = omage.astype(np.float64) / 65535.
        # clip to 0, 1
        omage = np.clip(omage, 0., 1.)
    if resize is not None:
        if type(resize) == int:
            resize = (resize, resize)
        #omage = skimage.transform.resize(omage, resize, order=1, preserve_range=True, anti_aliasing=True)
        # if shape is not the same, resize
        if omage.shape[0] != resize[0] or omage.shape[1] != resize[1]:
            if advanced_downsample==False:
                omage = skimage.transform.resize(omage, resize, order=resize_order, preserve_range=True, anti_aliasing=False) # important to use order 0 and disable anti_aliasing
            else:
                omage = downsample_omg(omage, factor=omage.shape[0]//resize[0])['omg_down_star']
    if trim_omg==True:
        occ = omage[...,3] > occ_thresh
        omage[~occ, :] = 0
    return omage

def omg2object(omage, name="omg", scale=1):
    import bpy
    from xgutils import bpyutil
    bpyutil.clear_collection("workbench")
    nvert, nface = meshing_objamage(omage)
    obj = bpyutil.mesh_from_pydata(nvert*scale, nface, name=name)
    return obj

def meshing_uv_map(occupancy):
    occ = occupancy.astype(bool)
    pixel_index = np.arange(occ.size).reshape(occ.shape)

    # Determine triangles' vertices
    is_tri_vert = occ & np.roll(occ, shift=-1, axis=0) & np.roll(occ, shift=-1, axis=1)
    verta = pixel_index
    vertb = np.roll(pixel_index, shift=-1, axis=1)
    vertc = np.roll(pixel_index, shift=-1, axis=0)
    face0 = np.stack([verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], axis=1)
    
    # Determine the second set of triangles' vertices
    is_tri_vert = occ & np.roll(occ, shift=1, axis=0) & np.roll(occ, shift=1, axis=1)
    verta = pixel_index
    vertb = np.roll(pixel_index, shift=1, axis=1)
    vertc = np.roll(pixel_index, shift=1, axis=0)
    face1 = np.stack([verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], axis=1)
    
    # Combine the two sets of faces
    face = np.concatenate([face0, face1], axis=0)

    return face
def meshing_objamage(omage, prune_verts=True):
    """ Turn an omage into a mesh.
    Args:
        omage: dict, containing 'position', 'occupancy', 'objnormal'.
        return_uv: bool, whether to return uv as well.
        prune_verts: bool, whether to remove unused vertices to reduce size.
    """
    # Generate pixel indices array
    if type(omage) == np.ndarray:
        omage = tensor2omg(omage)
    occ    = omage['occupancy'] > .5
    vert   = omage['position'] * 2 - 1
    objnormal = omage['objnormal'] * 2 - 1 if omage.get('objnormal', None) is not None else None
    pixel_index  = np.arange(occ.size).reshape(occ.shape)
    vert         = vert.reshape(-1, 3)
    objnormal       = objnormal.reshape(-1, 3) if objnormal is not None else None
    
    face = meshing_uv_map(occ)
    if face.shape[0] == 0: # no face, return empty mesh
        meshing_ret = dict( vert = np.zeros((0,3)), face = np.zeros((0,3)).astype(int), uv = np.zeros((0,2)))
        return meshing_ret

    # flip faces with inconsistent objnormal vs face normal
    if objnormal is not None:
        face_normal = np.cross(vert[face[:,1]] - vert[face[:,0]], vert[face[:,2]] - vert[face[:,0]])
        flip_mask = np.einsum('ij,ij->i', face_normal, objnormal[face[:,0]]) < 0
        face[flip_mask] = face[flip_mask][:,[0,2,1]]
    
    uv = nputil.makeGrid([0,0], [1,1], shape=(occ.shape[0], occ.shape[1]), mode='on')
    uv[..., [0,1]] = uv[..., [1,0]] # swap x, y to match the image coordinate system
    meshing_ret=dict( vert=vert, face=face, uv=uv)
    for key in omage:
        if key not in ['vert', 'face', 'uv'] and omage[key] is not None:
            meshing_ret[key] = omage[key].reshape(-1, omage[key].shape[-1])

    if prune_verts:
        vert, face, unique_vert_ind = geoutil.prune_unused_vertices(vert, face)
        uv = uv[unique_vert_ind]
        for key in meshing_ret:
            if key not in ['vert', 'face', 'uv']:
                print(key, meshing_ret[key].shape)
                meshing_ret[key] = meshing_ret[key][unique_vert_ind]
        meshing_ret['vert'], meshing_ret['face'], meshing_ret['uv'] = vert, face, uv
    return meshing_ret
def preview_omg(omg_tensor, geometry_only=None, omg_keys=['position', 'occupancy', 'objnormal', 'color', 'metal', 'rough'], scale_omg_to=None, return_rendered=True, combine_vomgs=True, resolution=512, render_kwargs=dict(shadow_catcher=True, samples=32, camera_position=None, ), seg_color_palette=None, decoder_kwargs=dict(render_mode='segcolor')):
    from xgutils import glbutil, bpyutil
    if scale_omg_to is not None: # can be (64, 64)
        # use nearest neighbor to resize the omg_tensor
        omg_tensor = skimage.transform.resize(omg_tensor, scale_omg_to, order=2, preserve_range=False, anti_aliasing=False)
        omg_tensor[..., 3] = (omg_tensor[..., 3] > .1).astype(np.float32)

    omg = tensor2omg(omg_tensor, seg_color_palette=seg_color_palette)
    if omg_tensor.shape[-1] == 4:
        img = visutil.imageGrid([omg['position'], omg['segcolor']], (1,2))
    else:
        img = visutil.imageGrid([omg['position'], omg['segcolor'], omg['objnormal'], omg['color'], omg['metal'], omg['rough']], (2,3))

    for key in ['position', 'occupancy', 'objnormal', 'color', 'metal', 'rough']:
        if key not in omg_keys:
            omg[key] = None

    #img = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)
    img = skimage.transform.resize(img, (resolution, resolution * img.shape[1]/img.shape[0]), order=0)
    geometry_only = (omg_tensor.shape[-1]==4) if geometry_only is None else geometry_only
    if return_rendered:
        # print(np.minimum(omg['occupancy'], 1-omg['occupancy']).max())
        # visutil.showImg( np.minimum(omg['occupancy'], 1-omg['occupancy']) )
        obj, meshing_ret = objamage_decoderv2(omg, geometry_only=geometry_only, **decoder_kwargs)
        nvert, nface = meshing_ret['vert'], meshing_ret['face']
        if nface.shape[0] == 0:
            rdimg = np.zeros((resolution, resolution, 4))
        else:
            # set color management to standard (from the default AgX)
            #bpy.context.scene.view_settings.view_transform = 'Standard'
            rdimg = bpyutil.render_scene(obj=obj, resolution=(resolution, resolution), **render_kwargs)
            if combine_vomgs:
                print(rdimg.shape, img.shape)
                img = np.concatenate([img, rdimg], axis=1)
        ret = dict()
        return img, rdimg, meshing_ret
    return img

def downsample_omg(omg, factor=16, anti_aliasing=False, visualize=False):
    """ downsample omg with edge snapping
    Args:
        omg: np.ndarray, (H, W, 4), omage tensor
        factor: int, downsample factor
        anti_aliasing: bool, whether to use anti_aliasing
        visualize: bool, whether to visualize the result
    Returns:
        dict, containing 'omg_down_star', 'omg_down', 'sov', 'edge_occ_down'
        'omg_down_star': np.ndarray, downsampled omg with edge snapping
        'omg_down': np.ndarray, downsampled omg without edge snapping
        'sov': np.ndarray, occupancy map with snapped boundaries highlighted
        'edge_occ_down': np.ndarray, edge occupancy map
    """
    # assuming omg is 1024 x 1024
    occ = omg[...,3]>=.5
    eroded_mask = scipy.ndimage.binary_erosion(occ, structure=np.ones((3,3))) # sqaure strucure is needed to get the corners
    edge_occ = ~eroded_mask & occ
    edge_omg = omg.copy()
    edge_omg[edge_occ==0] = -1.
    
    edge_occ_patches = einops.rearrange(edge_occ, '(h1 h2) (w1 w2) -> h1 w1 h2 w2', h2=factor, w2=factor)
    edge_occ_down = edge_occ_patches.max(axis=-1).max(axis=-1)
    eod_0_count  = (edge_occ_patches==0).sum(axis=-1).sum(axis=-1)
    eod_1_count  = (edge_occ_patches==1).sum(axis=-1).sum(axis=-1)
    edge_omg_patches = einops.rearrange(edge_omg, '(h1 h2) (w1 w2) c-> h1 w1 h2 w2 c', h2=factor, w2=factor)
    edge_omg_down = edge_omg_patches.sum(axis=-2).sum(axis=-2) + eod_0_count[...,None]
    edge_omg_down = np.divide(edge_omg_down, eod_1_count[...,None], out=np.zeros_like(edge_omg_down), where=eod_1_count[...,None]!=0)

    omg_down = skimage.transform.resize(omg, (omg.shape[0]//factor,)*2, order=0, preserve_range=False, anti_aliasing=anti_aliasing)
    
    omg_down_star = edge_omg_down * (edge_occ_down[...,None]) + omg_down * (1-edge_occ_down[...,None])
    star_occ = (omg_down[...,3] >= .5) | edge_occ_down
    
    sov = (omg_down[...,3] >= .5) # for visualizaton
    sov = sov * .5 + edge_occ_down.astype(float)
    sov[sov>=1.] = 1.
    if visualize==True:
        import matplotlib.pyplot as plt
        visutil.showImg(omg_down[...,:3])
        visutil.showImg(omg_down_star[...,:3])
        visutil.showImg(edge_omg_down[...,:3])
        plt.imshow(sov, cmap='gray')
        plt.show()
        plt.imshow(edge_occ_down, cmap='gray')
        plt.show()
        vomg0, rdimg, _ = omgutil.preview_omg(omg, return_rendered=True, geometry_only=False, combine_vomgs=True, decoder_kwargs=dict(render_mode='segcolor', preset_blend=PRESET_BLEND))
        vomg1, rdimg, _ = omgutil.preview_omg(omg_down, return_rendered=True, geometry_only=False, combine_vomgs=True, decoder_kwargs=dict(render_mode='segcolor', preset_blend=PRESET_BLEND))
        vomg2, rdimg, _ = omgutil.preview_omg(omg_down_star, return_rendered=True, geometry_only=False, combine_vomgs=True, decoder_kwargs=dict(render_mode='segcolor', preset_blend=PRESET_BLEND))
        igrid = visutil.imageGrid([vomg0, vomg1, vomg2], shape=(3,1))
        visutil.showImg(igrid)

    return dict(omg_down_star=omg_down_star, omg_down=omg_down,
            sov=sov, edge_occ_down=edge_occ_down, edge_occ=edge_occ)

def objamage_decoderv2(omage, save_path=None, boundary_zipping=False, boundary_zipping_eps=5e-2, preset_blend=PRESET_BLEND, geometry_only=False, render_mode='', export_glb=True, disable_maps=['normal'], keep_scene=False, obj_name = "objamage_decoded", shading_mode='smooth', **kwargs):
    """ 
    Decode an omage into a mesh in blender.
    ARGS: 
        omage: dict, containing 'position', 'occupancy', 'objnormal', 'color', 'metal', 'rough'.
        save_path: str, path to save the decoded mesh.
        preset_blend: str, path to the preset blend file.
        geometry_only: bool, whether to decode only the geometry without textures.
        render_mode: str, whether to render position map, segcolor, dist_transform, etc.
        export_glb: bool, whether to export the decoded mesh as glb.
    RETURNS:
        obj: bpy.types.Object, the decoded mesh object.
        nvert: np.ndarray, (N, 3), the vertices of the decoded mesh.
        nface: np.ndarray, (M, 3), the faces of the decoded mesh.
        meshing_ret: dict of np.ndarray, (N, c), the props of the decoded mesh.
    """
    import bpy
    from xgutils import bpyutil
    timer = sysutil.Timer()
    if keep_scene == False:
        bpyutil.load_blend(preset_blend)
        bpyutil.clear_collection("workbench")
        timer.update("loading preset blend")
    if type(omage) == np.ndarray:
        omage = tensor2omg(omage)
    meshing_ret = meshing_objamage(omage)
    nvert, nface = meshing_ret['vert'], meshing_ret['face']
    timer.update("meshing omage")
    # if empty mesh, skip
    if nface.shape[0] > 0:
        # create an object from the mesh with the uv using bpy. ... 
        obj = bpyutil.mesh_from_pydata(nvert, nface, meshing_ret['uv'], name=obj_name)
        timer.update("create mesh in blender")

        if boundary_zipping==True:
            # enter edit mode
            bpy.ops.object.mode_set(mode='EDIT')
            # vert select mode
            bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
            # select all
            bpy.ops.mesh.select_all(action='SELECT')
            # select boundaries
            bpy.ops.mesh.region_to_loop()
            # merge vertices by distance
            bpy.ops.mesh.remove_doubles(threshold=boundary_zipping_eps)
            # exit edit mode
            bpy.ops.object.mode_set(mode='OBJECT')

        # create new material (Principal BSDF) and assign the omage texture to it
        mat = bpy.data.materials.new(name="Material")
        mat.use_nodes = True
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf_node = nodes["Principled BSDF"]
        output_node = nodes["Material Output"]
        if not geometry_only:
            # create a new image texture node for rgb texture
            color = omage.get('color')
            if color is not None:
                img_node = nodes.new(type='ShaderNodeTexImage')
                img_node.image = bpyutil.npimg2bpy(color[...,:3], bpy_img=f"{obj_name}_omage_color", use_sRGB=True)
                bpycolor = bpyutil.bpyimg2np(f"{obj_name}_omage_color")
                # link the image texture node to the bsdf node
                links.new(img_node.outputs['Color'], bsdf_node.inputs['Base Color'])

            # create a new image texture node for normal texture
            normal = omage.get('objnormal')
            if normal is not None and 'normal' not in disable_maps:
                img_node = nodes.new(type='ShaderNodeTexImage')
                img_node.image = bpyutil.npimg2bpy(normal[...,:3], bpy_img=f"{obj_name}_omage_normal")
                # add a normal map node
                normal_map_node = nodes.new(type='ShaderNodeNormalMap')
                normal_map_node.space = 'WORLD'
                # link the image texture node to the normal map node
                links.new(img_node.outputs['Color'], normal_map_node.inputs['Color'])
                # link the normal map node to the bsdf node
                if (normal[...,:3].sum())>0.001: # if there is normal map
                    links.new(normal_map_node.outputs['Normal'], bsdf_node.inputs['Normal'])
                #links.new(img_node.outputs['Color'], bsdf_node.inputs['Normal'])

            # create a new image texture node for metal texture
            metal = omage.get('metal')
            if metal is not None and 'metal' not in disable_maps:
                img_node = nodes.new(type='ShaderNodeTexImage')
                img_node.image = bpyutil.npimg2bpy(metal[...,:1], bpy_img=f"{obj_name}_omage_metal")
                # add a color separation node
                sep_node = nodes.new(type='ShaderNodeSeparateColor')
                # link the image texture node to the color separation node
                links.new(img_node.outputs['Color'], sep_node.inputs['Color'])
                # link the color separation node to the bsdf node
                links.new(sep_node.outputs['Red'], bsdf_node.inputs['Metallic'])
            
            # create a new image texture node for rough texture
            rough = omage.get('rough')
            if rough is not None and 'rough' not in disable_maps:
                img_node = nodes.new(type='ShaderNodeTexImage')
                img_node.image = bpyutil.npimg2bpy(rough[...,:1], bpy_img=f"{obj_name}_omage_rough")
                # add a color separation node
                sep_node = nodes.new(type='ShaderNodeSeparateColor')
                # link the image texture node to the color separation node
                links.new(img_node.outputs['Color'], sep_node.inputs['Color'])
                # link the color separation node to the bsdf node
                links.new(sep_node.outputs['Red'], bsdf_node.inputs['Roughness'])

            # set color image as srgb
            bpy.ops.image.save_all_modified() # save all modified images
            bpy.data.images[f"{obj_name}_omage_color"].colorspace_settings.name = 'sRGB' # must put this line after save_all_modified, or else the image will just be black for unknown reason.
        else:
            if 'posmap' in render_mode:
                # create a new image texture node for position texture
                posmap = omage.get('position')
                img_node = nodes.new(type='ShaderNodeTexImage')
                img_node.image = bpyutil.npimg2bpy(posmap[...,:3], bpy_img=f"{obj_name}_omage_position", use_sRGB=False)
                links.new(img_node.outputs['Color'], bsdf_node.inputs['Base Color'])

                bpy.ops.image.save_all_modified() # save all modified images
                bpy.data.images["Objamage_pos"].colorspace_settings.name = 'sRGB' # must put this line after 
            if 'segcolor' in render_mode:
                segcolor = omage.get('segcolor')
                if 'dist_transform' in render_mode:
                    import scipy
                    occ = omage['occupancy'][...,0] > .5
                    occ = skimage.transform.resize(occ, (512,)*2, order=0, preserve_range=True, anti_aliasing=False) # important to use order=0 to preserve the binary nature of the occupancy map
                    segcolor = skimage.transform.resize(segcolor, (512,)*2, order=1, preserve_range=True, anti_aliasing=False)
                    df = scipy.ndimage.distance_transform_edt(occ.astype(float))
                    df_intensity= (np.sin( df / df.shape[0] / 2 / np.pi * 800 ) ** 2 / 2) + .5
                    segcolor = segcolor * df_intensity[..., None]
                    visutil.showImg(segcolor)
                img_node = nodes.new(type='ShaderNodeTexImage')
                img_node.image = bpyutil.npimg2bpy(segcolor, bpy_img=f"{obj_name}_omage_segcolor", use_sRGB=True)
                links.new(img_node.outputs['Color'], bsdf_node.inputs['Base Color'])
                #directly link to output node
                # links.new(img_node.outputs['Color'], output_node.inputs['Surface'])
                # set roughness to 1
                bsdf_node.inputs['Roughness'].default_value = 1.

                bpy.ops.image.save_all_modified() # save all modified images
                bpy.data.images[f"{obj_name}_omage_segcolor"].colorspace_settings.name = 'sRGB' # must put this line after
            if 'solid' in render_mode:
                bpyutil.set_obj_material(obj, 'MetalMaterial')
        if shading_mode == 'flat':
            bpy.ops.object.shade_flat()
        elif shading_mode == 'smooth':
            bpy.ops.object.shade_smooth(use_auto_smooth=True)
        timer.update("create material")

        if save_path is not None:
            if export_glb:
                bpy.ops.object.shade_smooth(use_auto_smooth=True)
                bpyutil.save_blend(save_path, over_write=True)
                bpy.ops.export_scene.gltf(filepath=save_path.replace(".blend", ".glb"), export_format='GLB', use_selection=True)
            else:
                bpyutil.save_blend(save_path, over_write=True)

        timer.update("save blend and glbs")
        timer.rank_interval()
    else:
        # create monkey head if the mesh is empty
        bpy.ops.mesh.primitive_monkey_add()
        obj = bpy.context.active_object
        if save_path is not None:
            bpyutil.save_blend(save_path, over_write=True)
            if export_glb:
                bpy.ops.export_scene.gltf(filepath=save_path.replace(".blend", ".glb"), export_format='GLB', use_selection=True)

    return obj, meshing_ret
