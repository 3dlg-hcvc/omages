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

def meshing_objamage(objamage, prune_verts=True):
    """ Turn an objamage into a mesh.
    Args:
        objamage: dict, containing 'position', 'occupancy', 'objnormal'.
        return_uv: bool, whether to return uv as well.
        prune_verts: bool, whether to remove unused vertices to reduce size.
    """
    # Generate pixel indices array
    if type(objamage) == np.ndarray:
        objamage = tensor2omg(objamage)
    occ    = objamage['occupancy'] > .5
    vert   = objamage['position'] * 2 - 1
    objnormal = objamage['objnormal'] * 2 - 1 if objamage.get('objnormal', None) is not None else None
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
    for key in objamage:
        if key not in ['vert', 'face', 'uv'] and objamage[key] is not None:
            meshing_ret[key] = objamage[key].reshape(-1, objamage[key].shape[-1])

    if prune_verts:
        vert, face, unique_vert_ind = geoutil.prune_unused_vertices(vert, face)
        uv = uv[unique_vert_ind]
        for key in meshing_ret:
            if key not in ['vert', 'face', 'uv']:
                print(key, meshing_ret[key].shape)
                meshing_ret[key] = meshing_ret[key][unique_vert_ind]
        meshing_ret['vert'], meshing_ret['face'], meshing_ret['uv'] = vert, face, uv
    return meshing_ret

def remove_small_patches(obj, min_area=1e-5, max_n_patch=128):
    import igl
    import bpy
    from xgutils import bpyutil
    bpyutil.set_active_exclusive(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    # deselect all faces, edges, verts
    bpy.ops.mesh.select_all(action='SELECT')

    ov, of, ou = bpyutil.get_trimesh(obj, return_uv=True)
    f_component_id = igl.facet_components(of) # assuming component ids have no gaps (not like 0,1,3,4)
    face_area_2d = np.abs(igl.doublearea(ou, of)) / 2
    face_area_3d = np.abs(igl.doublearea(ov, of)) / 2
    component_area_2d = np.bincount(f_component_id, weights=face_area_2d)
    component_area_3d = np.bincount(f_component_id, weights=face_area_3d)

    remove_component_mask = (component_area_2d < min_area) | (component_area_3d < min_area)

    rkfarea = np.argsort(component_area_3d)[::-1]
    print("num patch", len(rkfarea))
    # only keep K largest patches
    if len(rkfarea) > max_n_patch:
        remove_component_mask[rkfarea[max_n_patch:]] = True

    remove_face_mask = remove_component_mask[f_component_id]

    # select faces to remove
    mesh = obj.data
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT') # deselect all verts, edges as well.
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh.polygons.foreach_set('select', remove_face_mask)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.delete(type='FACE')
    nv, nf, nu = bpyutil.get_trimesh(obj, return_uv=True)

    n_patch_original = len(component_area_2d)
    n_patch_removed = remove_component_mask.sum()
    n_patch_final = len(np.unique(igl.facet_components(nf)))
    print("#patches original", n_patch_original)
    print("#patches removed due to small area", n_patch_removed)
    print("#patches final", n_patch_final)

    meta_info = dict(n_patch_original=n_patch_original, n_patch_removed=n_patch_removed, n_patch_final=n_patch_final)

    return (ov, of, ou), (nv, nf, nu), meta_info

# objamage autoencoder v2
def repacking_uv2(obj, margin=0.02, min_area=1e-5, max_n_patch=128, rescale_area=False, shape_method='AABB', margin_method='SCALED'):
    """ 
    repacking uv version 2, assuming all parts are joined into a single mesh,
    Different from v1:
        no need to first seperate by loose parts.
        resize with 3d area, detect overlap and re-unwrap the overlaps after repacking.
        weld the obj in 3d (welding in 3D is very useful for obtaining connected uv islands)
    """
    import bpy
    from xgutils import bpyutil
    bpyutil.set_active_exclusive(obj)
    # create if not exist
    if 'Repack' not in obj.data.uv_layers:
        obj.data.uv_layers.new(name='Repack')
    else:
        print('Repack uv map already exists, skip creating.')
        return
    # set the new uv map as the active one
    obj.data.uv_layers['Repack'].active = True

    # Ensure we're in edit mode for UV packing
    bpy.ops.object.mode_set(mode='EDIT')
    # weld the obj in 3d (welding in 3D is very useful for obtaining connected uv islands)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=1e-6, use_unselected=False, use_sharp_edge_from_normals=True) # already selected all verts

    # remove islands with area less than 0.0001 and only keep the largest K islands
    # remove small islands
    _, _, meta_info = remove_small_patches(obj, min_area=min_area, max_n_patch=max_n_patch)

    # select all uv
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.context.scene.tool_settings.use_uv_select_sync = False # It turns out you have to select multiple uv verts if sync is turned on, which may cause problem for uv repacking operations.
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.select_all(action='SELECT')

    if rescale_area == True:
        bpy.ops.uv.average_islands_scale() # scale all islands according to their area in 3D. For .glb with multiple objects, this is necessary as the UV islands are not scaled properly across objects.
    # Pack UV islands with specified parameters
    bpy.ops.uv.pack_islands(rotate=False, scale=True, margin=margin, shape_method=shape_method, margin_method=margin_method)
    """ AABB is fastest (3.5s), Convex is fast (4s) while Concave is slow (15s). test case: formschussel_fur_terra_sigillata-schalen.glb
        Note that also AABB will cause more empty space, but it is easier for generative models.
    """
    bpy.ops.uv.seams_from_islands()

    # # detect overlap #!! It turns out overlap is very common, but most of them are tiny glitches. But if uv-unwrap the pattern may be destroyed.
    # bpy.ops.uv.select_all(action='SELECT')
    # bpy.ops.uv.select_overlap() # select all overlaps
    # bpy.ops.uv.select_linked() # select all islands containing these overlaps
    # bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.1) # 
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    return meta_info


def setup_baking(node_tree, resolution=10, bake_target='position'):
    """ First bake objnormal since it can be used to determine occupancy in 2D uv map.
    """
    import bpy
    # object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.cycles.use_denoising = False # disable denoising to speed up baking

    baking_kwargs = dict(type='EMIT', pass_filter={'EMIT'}, use_clear=True, margin=32, margin_type='EXTEND') # large margin size will make the baking slower. Small margin (e.g. 2) will cause artifacts when downscale the objamage.

    nodes = node_tree.nodes
    links = node_tree.links
    bsdf_node = nodes["Principled BSDF"]
    output_node = nodes["Material Output"]

    if bake_target == 'objnormal' or bake_target == 'position':
        nnname = "Objamage_geometry"
        geo_node = nodes.get(nnname, None)
        if geo_node is None:
            geo_node = nodes.new(type='ShaderNodeNewGeometry')
            geo_node.name  = nnname

        if bake_target == 'objnormal':
            bpy.ops.object.shade_smooth(use_auto_smooth=False) # use smooth shading
            baking_kwargs.update(type='NORMAL', pass_filter=set(), normal_space='OBJECT', margin_type='EXTEND', margin=1)
            links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        elif bake_target == 'position':
            links.new(geo_node.outputs['Position'], output_node.inputs['Surface'])
    else:
        mapper = dict(color='Base Color', metal='Metallic', rough='Roughness') # removed normal. use objnormal instead.
        # First check if there is a link (texture map), if not, use the value directly
        if len(bsdf_node.inputs[mapper[bake_target]].links) == 0:
            # create a new value node whose value is the same as the bsdf node's input
            if bake_target == 'color':
                node_type = 'ShaderNodeRGB'
                node_output = 'Color'
            else:
                node_type = 'ShaderNodeValue'
                node_output = 'Value'
            value_node = nodes.new(type=node_type)
            value_node.name = f"Objamage_{bake_target}_value"
            value_node.outputs[node_output].default_value = bsdf_node.inputs[mapper[bake_target]].default_value
            # if bake_target == 'color':
            #     for i in range(3):
            #         value_node.outputs[node_output].default_value[i] **= 1/2.2 # gamma correction
            input_socket = value_node.outputs[node_output]
            # connect the value node to output node's Surface socket
            links.new(input_socket, output_node.inputs['Surface'])
        else:
            link = bsdf_node.inputs[mapper[bake_target]].links[0]
            if bake_target == 'normal':
                link = link.from_node.inputs['Color'].links[0]
            source_node = link.from_node
            source_socket = link.from_socket
            # connect the source node to output node's Surface socket
            links.new(source_node.outputs[source_socket.name], output_node.inputs['Surface'])

    # remove the old image if exists
    bake_image = bpy.data.images.get(f"objamage_{bake_target}")
    if bake_image is None:
        bake_image = bpy.data.images.new(f"objamage_{bake_target}", width=2**resolution, height=2**resolution, alpha=True, float_buffer=True)
    bake_image.file_format = 'OPEN_EXR'
    bake_image.colorspace_settings.name = 'Non-Color'
    
    ntexname = f"Objamage_{bake_target}"
    bake_node = nodes.get(ntexname, None)
    if bake_node is not None:
        nodes.remove(bake_node)
    bake_node = nodes.new(type='ShaderNodeTexImage')
    bake_node.image = bake_image
    bake_node.name  = ntexname

    nnname = "Objamage_uvmap"
    uv_map_node = nodes.get(nnname, None)
    # remove the old uv map node if exists
    if uv_map_node is None:
        # Create a new UV Map node
        uv_map_node = nodes.new(type='ShaderNodeUVMap')
        uv_map_node.uv_map = "Repack" # Assuming'Repack' is the name of the new UV Map
        uv_map_node.name = nnname
        
    # Connect the UV Map node to the Image Texture node
    links.new(uv_map_node.outputs['UV'], bake_node.inputs['Vector'])

    # set pos_node as the active node
    nodes.active = bake_node

    return baking_kwargs

def baking_obj(obj, resolution=10, bake_target='position'):
    """ 
    setup baking for an object
    """
    import bpy
    from xgutils import bpyutil
    # Iterate all materials of the object
    for mat in obj.data.materials:
        if mat.use_nodes:
            baking_kwargs = setup_baking(mat.node_tree, resolution=resolution, bake_target=bake_target)
    
    bpy.ops.object.bake(**baking_kwargs)
    
    # save all modified images
    bpy.ops.image.save_all_modified()
    baked_img = bpyutil.bpyimg2np(bpy.data.images[f"objamage_{bake_target}"])
    return baked_img

def baking_uv(obj, resolution=10):
    import bpy
    from xgutils import bpyutil
    bpyutil.set_active_exclusive([obj])
    bpy.ops.object.shade_smooth(use_auto_smooth=False) # use smooth shading
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.context.scene.tool_settings.use_uv_select_sync = False # It turns out you have to select multiple     
    obj.data.uv_layers['UVMap'].active_render = True # activate the default uv map
    
    # set up baking settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 1 # baking surface properties does not need high samples
    bpy.context.scene.cycles.use_adaptive_sampling = False
    bpy.context.scene.render.bake.margin_type = 'EXTEND' # or 'ADJACENT_FACES'
    bpy.context.scene.render.bake.margin = 2#**resolution

    # Iterate all materials of the object
    # Difference between objnormal and normal: objnormal is the final normal (combining normal map with geometry normal), while normal is the normal map to the material.
    #objamage = dict(objnormal=None, position=None, color=None, normalmap=None, metal=None, rough=None)
    objamage = dict(objnormal=None, position=None, color=None, metal=None, rough=None)
    for bake_target in objamage:
        objamage[bake_target] = baking_obj(obj, resolution=resolution, bake_target=bake_target)
    # occupancy is objnormal booleaned. if trunormal is 0 vector then it is 0, otherwise 1
    objamage['occupancy']  = (np.linalg.norm(objamage['objnormal'][...,:3], axis=-1) > 0)[..., None]
    objamage['position']   = (objamage['position'][...,:3] + 1) / 2 # normalize to [0, 1]
    objamage['objnormal']  = objamage['objnormal'][...,:3] # remove alpha channel

    return objamage # all images are in [0, 1] range
def objamage_encoderv2(glb_path, resolution=10, min_patch_area=1e-4, max_n_patch=128, repack_margin=0.02, repacking_margin_method='FRACTION', save_path=None, rescale_area=False, use_uint16=False, subdiv=0,  preset_blend=PRESET_BLEND):
    from xgutils import bpyutil
    import bpy
    timer = sysutil.Timer()
    if preset_blend is None:
        preset_blend = bpyutil.preset_blend
    bpyutil.load_blend(preset_blend)
    timer.update("loading preset blend")
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(glb_path) # load glb, join into a single mesh and put it to workbench collection
    if subdiv > 0:
        bpy.ops.object.modifier_add(type='SUBSURF')
        bpy.context.object.modifiers["Subdivision"].levels = subdiv
        bpy.ops.object.modifier_apply(modifier="Subdivision")
        # triangulate
        bpy.ops.object.modifier_add(type='TRIANGULATE')
        bpy.ops.object.modifier_apply(modifier="Triangulate")
    timer.update("loading glb")

    obj = bpy.context.active_object
    bpyutil.set_active_exclusive(obj)
    meta_info = repacking_uv2(obj, margin=repack_margin, min_area=min_patch_area, max_n_patch=max_n_patch, margin_method=repacking_margin_method, rescale_area=rescale_area)
    timer.update("repacking uv")
    #vert, face, uv = bpyutil.get_trimesh_uv(obj)
    vert, face, uv = bpyutil.get_trimesh(obj, return_uv=True)
    #uv = vert[..., :2]
    timer.update("get mesh")
    objamage = baking_uv(obj, resolution=resolution)
    timer.update("baking uv")
    
    timer.rank_interval()
    if save_path is not None:
        if save_path.endswith(".ply"):
            import igl
            igl.write_triangle_mesh(save_path, vert, face, uv, force_ascii=False)
        if save_path.endswith(".blend"):
            # sometimes saving may cause exception when using multi-processing
            # so try again if failed
            bpyutil.save_blend(save_path, over_write=True)

    objamage_tensor = omg2tensor(objamage)
    objamage['combined'] = np.clip(objamage_tensor, 0, 1)
    if use_uint16:
        objamage['combined'] = (objamage['combined'] * 65535).astype(np.uint16)
    return objamage, vert, face, uv, meta_info
