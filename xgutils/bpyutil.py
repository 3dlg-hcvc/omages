import re
import os
import sys
import mcubes
import numpy as np
import contextlib

import bpy
import bmesh

import matplotlib.image as mpimg
from xgutils.miscutil import preset_glb, preset_blend
def readImg(path):
    path = os.path.expanduser(path)
    return mpimg.imread(path)

# data related
def load_blend(path, verbose=False):
    if path[0] != '/' and path[0] != '~':
        print("Warning: path is not absolute, trying to convert to absolute path.")
        path = os.path.join(os.getcwd(), path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if verbose:
        print("Clearing everything and loading blend file: {}".format(path))
    bpy.ops.wm.open_mainfile(filepath=path)
def save_blend(path, over_write=False, verbose=False):
    # make sure path ends with .blend
    if path[-6:] != '.blend':
        raise ValueError("path must end with .blend")
    if verbose==True:
        print("Saving blend file: {}".format(path))
    if os.path.exists(path) and over_write == True:
        os.remove(path)
    # try to save the blend file. If it fails, then raise an error.
    # repeatedly try to save for 10 times. If all of them fail, then raise an error.
    for i in range(10):
        try:
            bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
            break
        except Exception as e:
            if i == 9:
                print(f"Failed to save blend file: {path}")
                raise e
            print(f"Failed to save blend file. Retrying... Retry count: {i+1}/10")

    
def load_glb(glb_path, collection='workbench'):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=glb_path, import_shading='FLAT')
    # join all selected objects
    
    for obj in bpy.context.selected_objects:
        # Check if the object is not a mesh
        if obj.type == 'MESH': # set a mesh object as active (in order to join)
            bpy.context.view_layer.objects.active = obj
    bpy.ops.object.join()
    
    obj = bpy.context.view_layer.objects.active
    normalize_mesh(obj)
    for obj in bpy.context.selected_objects:
        # remove non-mesh objects
        if obj.type != 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    # move all selected objects to the workbench collection
    move_selected_to_collection(collection)
    return obj


def move_selected_to_collection(collection_name):
    # Get the collection
    collection = bpy.data.collections.get(collection_name)
    if collection:
        # Loop through selected objects
        for obj in bpy.context.selected_objects:
            # Unlink the object from other collections
            for col in list(obj.users_collection):
                if col.name != collection_name:
                    col.objects.unlink(obj)            
            # Link the object to the collection
            collection.objects.link(obj)
        # Update the scene
        bpy.context.view_layer.update()
    else:
        print(f"Collection '{collection_name}' not found. Create it now.")
        bpy.data.collections.new(collection_name)
        move_selected_to_collection(collection_name)

def is_structrna_valid(obj):
    try:
        _ = obj.name
        return True
    except ReferenceError:
        return False
def purge_obj(obj):
    # if obj is already removed, then return
    if is_structrna_valid(obj) == False:
        orphan_purge()
        return
        
    bpy.data.objects.remove(obj, do_unlink=True)
    orphan_purge()

def orphan_purge():
    bpy.ops.outliner.orphans_purge(do_recursive=True)

def cleanup_unused_data():
    # Function to remove unused data blocks of a specific type
    def remove_unused_data_block(data_block_collection):
        for item in [block for block in data_block_collection if block.users == 0]:
            data_block_collection.remove(item)

    # List of data types to clean up
    data_types_to_cleanup = [
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.curves,
        bpy.data.metaballs,
        bpy.data.fonts,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.libraries,
        bpy.data.actions,
        bpy.data.particles,
        # Add more data types here as needed
    ]

    # Remove unused data blocks of each type
    for data_type in data_types_to_cleanup:
        remove_unused_data_block(data_type)

    print("Recursive unused data blocks cleanup complete.")

def bpyimg2np(bpy_img):
    if type(bpy_img) is str:
        bpy_img = bpy.data.images.get(bpy_img)
    width, height = bpy_img.size
    channels = bpy_img.channels
    img = np.zeros(shape = (width*height*channels), dtype=np.float32)
    bpy_img.pixels.foreach_get(img)
    img = np.reshape(img, (height, width, channels))
    return img
    
def npimg2bpy(np_img, bpy_img=None, use_sRGB=False):
    """ bpy_img: bpy.data.images or None or a string. If None, a new image will be created. If a string, the image will be created with the name. """
    alpha = True if np_img.shape[2] == 4 else False
    if len(np_img.shape) == 2 or np_img.shape[2] == 1:
        np_img = np.repeat(np_img, 4, axis=2)
    if np_img.shape[2] == 3:
        np_img = np.concatenate([np_img, np.ones((np_img.shape[0], np_img.shape[1], 1))], axis=2)
    if type(bpy_img) is str:
        # delete the old image if exists
        bi_name = bpy_img
        bpy_img = bpy.data.images.get(bpy_img)
        if bpy_img is None:
            #     bpy.data.images.remove(bpy_img)
            bpy_img = bpy.data.images.new(name=bi_name, width=np_img.shape[0], height=np_img.shape[1], alpha=alpha, float_buffer=True, is_data=True)
    elif bpy_img is None:
        bpy_img = bpy.data.images.new(name="temp", width=np_img.shape[0], height=np_img.shape[1], alpha=alpha, float_buffer=True, is_data=True)
    else:
        # rescale if the size is different
        if bpy_img.size[0] != np_img.shape[0] or bpy_img.size[1] != np_img.shape[1]:
            bpy_img.scale(np_img.shape[0], np_img.shape[1])
    #np_img = np_img[:,:,::-1] # convert from rgb to bgr
    #np_img = np.random.rand(256, 256, 4)
    np_img = np_img.flatten().astype(np.float32)
    # clip the value to [0,1]
    np_img = np.clip(np_img, 0, 1)
    if use_sRGB:
        np_img = np_img ** (1/2.2)
    bpy_img.pixels.foreach_set(np_img)
    return bpy_img


def clear_all():
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Select all objects
    bpy.ops.object.select_all(action='SELECT')

    # Delete selected objects
    bpy.ops.object.delete()

def clear_collection(collection_name):
    # Get the collection
    collection = bpy.data.collections.get(collection_name)
    if collection:
        # Duplicate the list of objects since we'll be modifying it during iteration
        objs = list(collection.objects)
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        # Loop through objects and delete them
        for obj in objs:
            # if obj is rootnode, skip
            if 'RootNode' in obj.name:
                continue
            # Select the object
            obj.select_set(True)
            # Remove the object from the collection
            collection.objects.unlink(obj)
            # Delete the object
            bpy.data.objects.remove(obj)
        # Update the scene
        bpy.context.view_layer.update()
    else:
        print(f"Collection '{collection_name}' not found. Create it now.")
        bpy.data.collections.new(collection_name)
    cleanup_unused_data()

# spatial transform
def normalize_mesh(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    # clear the glb's default axes tree structure.
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    # move the object origin (the yellow dot) to world origin.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # move the geometry to object (world) origin, so that the origin is the center of the bounding box.
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

    # scale the object to fit the bounding box into a unit cube.
    max_extent = max(obj.dimensions)
    scale_factor = 1.9999 / max_extent if max_extent != 0 else 1
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    return obj

def set_obj_material(obj, material):
    if type(material) is str:
        material =  bpy.data.materials[material]
    # Assign material to object
    # obj.data.materials = [material]
    obj.data.materials.clear()
    obj.data.materials.append(material)
    # if obj.data.materials:
    #     obj.data.materials[0] = material  # Replace the first material
    # else:
    #     obj.data.materials.append(material)  # Add new material if none exists


# render
def set_camera_pos(location):
    # Get the current scene
    scene = bpy.context.scene
    # Ensure there is an active camera
    if not scene.camera:
        print("No active camera found in the scene.")
        return
    # Get the active camera
    camera = scene.camera
    # Set the camera location
    camera.location = location

def random_temp_filename(suffix='png', temp_dir='~/.tmp_xgutils/'):
    import random
    import string
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    fname = name + '.' + suffix
    
    import os
    temp_dir = os.path.expanduser('~/.temp/xgutils/')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    path = os.path.join(temp_dir, fname)
    return path, name

def render_scene(p=None, obj=None, resolution=(256,)*2, samples=1024, resolution_percentage=100, shadow_catcher=True, render_engine='CYCLES', camera_position=None, output_path=None, view_transform='AgX', view_look=f'AgX - Very High Contrast'): # can not use BLENDER_EEVEE in docker
    if camera_position is not None:
        set_camera_pos(camera_position)

    # Set render settings
    bpy.context.scene.view_layers["ViewLayer"].use_freestyle = False
    bpy.context.scene.render.film_transparent = True
    # Get the 'Floor' object
    floor = bpy.data.objects.get('Floor')
    if obj is not None:
        # get the lowest z value of the object to set the floor 
        min_z = obj.bound_box[0][2]
        # set the object to the ground
        # obj.location.z = min_z + .001
        # set the floor to the lowest z value of the object
        floor.location.z = min_z - .05
    if output_path is None:
        file_format = 'PNG'
        output_path, _ = random_temp_filename(suffix='png')
        print("output_path", output_path)
    else:
        # Ensure the output path exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        file_format = output_path.split('.')[-1].upper()
    if render_engine == 'CYCLES':
        print("default samples", bpy.context.scene.cycles.samples)
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.cycles.use_denoising = True
        # set the floor as shadow catcher
        if floor:
            floor.cycles.is_shadow_catcher = True
        # make it visible in render
        floor.hide_render = not shadow_catcher
        # bpy.context.scene.view_settings.look = 'AgX - Punchy'
        # bpy.context.scene.view_settings.look = 'AgX - Very High Contrast'
        # bpy.context.scene.view_settings.look = 'AgX - High Contrast'
        bpy.context.scene.view_settings.view_transform = view_transform
        bpy.context.scene.view_settings.look = view_look
        
        bpy.context.scene.view_settings.exposure = 0


    else:
        floor.hide_render = True
    bpy.context.scene.view_settings.exposure = 0

    bpy.context.scene.render.engine = render_engine
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = resolution_percentage
    print("File format: {}".format(file_format))
    bpy.context.scene.render.image_settings.file_format = file_format
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.color_depth = '16'
    # Set output path
    bpy.context.scene.render.filepath = output_path

    
    # TODO: support render passes
    # bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
    # bpy.ops.render.render(write_still=True, layer="ViewLayer")
    # print(f"Render complete. Image saved to: {bpy.context.scene.render.filepath}")
    # albedo_layer = bpy.data.images['Render Result'].view_layer[0].passes['DiffCol']
    
    # Render the scene
    bpy.ops.render.render(write_still=True, layer="ViewLayer")
    print(f"Render complete. Image saved to: {bpy.context.scene.render.filepath}")
    img = readImg(output_path)
    return img
def make_PBSDF_vertex_color_mat(baseMatName='MetalMaterial', newMatName=None, color_layer_name='vertex_color'):
    if newMatName is None:
        newMatName = baseMatName + '_vertex_color'
    base_mat = bpy.data.materials.get(baseMatName)
    # Create a new material
    mat = base_mat.copy()
    mat.name = newMatName
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf_node = nodes["Principled BSDF"]
    output_node = nodes["Material Output"]

    vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
    vertex_color_node.name = 'vertex_color'
    # Link the nodes
    links.new(vertex_color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    return mat
def render_mesh(vert, face, vert_color=None, preset_blend=preset_glb, **kwargs):
    load_blend(preset_blend)
    clear_collection('workbench')
    obj = mesh_from_pydata(vert, face, vert_color=vert_color)
    make_PBSDF_vertex_color_mat(
        baseMatName="RoughMaterial",
        newMatName="rough_vertex_color_mat"
    )
    
    set_obj_material(obj, 'rough_vertex_color_mat')
    #normalize_mesh(obj)
    img = render_scene(obj=obj, **kwargs)
    return img

# data related
def set_active_exclusive(objs):
    if type(objs) is not list:
        objs = [objs]
    # if there is an active object
    if bpy.context.active_object is not None:
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
def list_objects():
    for obj in bpy.data.objects:
        print(obj.name)
    
def boolean_union(obj1, obj2):
    mod = obj1.modifiers.new(name='Boolean', type='BOOLEAN')
    mod.operation = 'UNION'
    mod.use_self = False
    mod.object = obj2
    set_active_exclusive(obj1)
    bpy.ops.object.modifier_apply(modifier=mod.name)

def get_face_adj_mat(obj):
    num_verts, num_faces = len(obj.data.vertices), len(obj.data.polygons)
    me = obj.data
    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(me)   # fill it in from a Mesh
    adj_mat = np.zeros((num_faces, num_faces), dtype=bool)
    for edge in bm.edges:
        linked_faces = edge.link_faces
        lfaces = np.array([lf.index for lf in linked_faces])
        # select submatrix by rows and colums
        adj_mat[ np.ix_(lfaces, lfaces)] = True
    bm.free()
    return adj_mat

def get_polygon(obj):
    num_verts, num_faces = len(obj.data.vertices), len(obj.data.polygons)
    vertices = np.array([vertex.co for vertex in obj.data.vertices])
    # Get the polygonal face list
    faces = [[vertex for vertex in polygon.vertices] for polygon in obj.data.polygons]
    face_normals = np.array([polygon.normal for polygon in obj.data.polygons])
    return vertices, faces, face_normals

def split_verts_based_on_uv(vert, face, uv):
    # Flatten the arrays for processing
    face_flat = face.flatten()
    uv_flat = uv.reshape(-1, 2)  # Flatten UV while keeping UV coords together

    # uint64 can represent 19 digits.
    # combine the 1 integer and 2 floats into a single 19 digits uint64
    # first 9 digits for the vertex index, next 5 for the u coord, last 5 for the v coord
    enc0 = face_flat.astype(np.uint64)
    enc1 = (uv_flat[:, 0] * 10**5).astype(np.uint64)
    enc2 = (uv_flat[:, 1] * 10**5).astype(np.uint64)
    combined = enc0 * 10**8 + enc1 * 10**5 + enc2

    unique_combined, uind, indices = np.unique(combined, return_index=True, return_inverse=True)
    nvert = vert[ face_flat[uind] ]
    nuv   = uv_flat[ uind ]
    nface = indices.reshape(-1, 3)

    return nvert, nface, nuv #new_verts, new_F, new_UVs

def get_trimesh(obj, return_uv=False):
    """ triangulate the mesh and return vert and face as numpy array"""
    assert obj.type == 'MESH'
    # switch to edit mode to select all
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT') # this is to make sure uv is also selected
    bpy.ops.object.mode_set(mode='OBJECT')

    vert = np.empty((len(obj.data.vertices), 3))
    face = np.empty((len(obj.data.polygons), 3), dtype=int)
    obj.data.vertices.foreach_get('co', vert.ravel())
    obj.data.loop_triangles.foreach_get('vertices', face.ravel())

    if return_uv:
        uv   = np.empty((len(obj.data.polygons), 3, 2))
        obj.data.uv_layers.active.data.foreach_get('uv', uv.ravel())
        nv, nf, nu = split_verts_based_on_uv(vert, face, uv)
        return nv, nf, nu
    else:
        return vert, face

# camera related
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
def interpolate_camera_poses(pose1, pose2, t):
    """
    Interpolate between two camera poses around (0, 0, 0) in a circular manner.

    Parameters:
    - pose1: np.array, shape (3,), the first camera pose (rotation vector or Euler angles)
    - pose2: np.array, shape (3,), the second camera pose (rotation vector or Euler angles)
    - t: float, interpolation parameter between 0 and 1

    Returns:
    - interpolated_pose: np.array, shape (3,), the interpolated camera pose (rotation vector)
    """
    # Convert poses to rotation objects
    rot1 = R.from_rotvec(pose1)
    rot2 = R.from_rotvec(pose2)
    rots = R.concatenate([rot1, rot2])
    
    # Perform spherical linear interpolation (slerp)
    slerp = Slerp([0, 1], rots)
    interpolated_rot = slerp(t)
    
    # Convert the interpolated rotation back to a rotation vector
    interpolated_pose = interpolated_rot.as_rotvec()
    
    return interpolated_pose

# obsolete, slow implementation
# def get_trimesh_uv(obj):
#     """ triangulate the mesh and return vert and face as numpy array
#         Also, may add new vert to separate uv vertices.
#     """
#     assert obj.type == 'MESH'
#     bm = bmesh.new()
#     bm.from_mesh(obj.data)
#     # Triangulate the mesh (unchanged if the mesh is already triangular)
#     bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY')
#     # Get the vertices and facesx
#     verts = np.array([vert.co for vert in bm.verts])
#     faces = np.array([[vert.index for vert in face.verts] for face in bm.faces], dtype=int)

#     new_verts = []
#     new_uv_verts = []

#     uv_verts = np.zeros((len(verts), 2))
#     uv_filled = np.zeros(len(verts), dtype=bool)
#     uv_layer = bm.loops.layers.uv.active
#     for fi, face in enumerate(bm.faces):
#         for loop in face.loops:
#             vi = loop.vert.index
#             #if uv_filled[vi] == True: # if the two verts are the same, then no need to separate
#             if uv_filled[vi] == True:
#                 if np.allclose(uv_verts[vi], loop[uv_layer].uv, atol=1e-5):
#                     continue
#                 # need to separate the uv verts
#                 new_verts.append(verts[vi])
#                 new_uv_verts.append(loop[uv_layer].uv)
#                 # get loop's face index
#                 lfi = loop.face.index
#                 if vi == faces[lfi][0]:
#                     faces[lfi][0] = len(verts) + len(new_verts) - 1
#                 elif vi == faces[lfi][1]:
#                     faces[lfi][1] = len(verts) + len(new_verts) - 1
#                 elif vi == faces[lfi][2]:
#                     faces[lfi][2] = len(verts) + len(new_verts) - 1
#                 else:
#                     raise ValueError("loop's face index not in face")
#             else:
#                 uv_filled[vi] = True
#                 uv_verts[vi] = loop[uv_layer].uv
#     if len(new_verts) > 0:
#         verts = np.concatenate([verts, new_verts], axis=0)
#         uv_verts = np.concatenate([uv_verts, new_uv_verts], axis=0)
#     # Free the BMesh data
#     bm.free()
#     return verts, faces, uv_verts

def mesh_from_pydata(vert, face, uv=None, vert_color=None, name='mesh'):
    """
        Assuming color is per vertex,
        but the actual color to blender is per face vert. Meaning that the input color is actually of shape (F*3,4)
    """
    # Example data (replace with your actual data)
    # vert = [(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1)]  # Nx3
    # face = [(0, 1, 2), (2, 3, 0)]  # Mx3 assuming triangles
    # uv = [(0, 0), (1, 0), (1, 1), (0, 1)]  # Nx2

    # Create mesh and object
    mesh = bpy.data.meshes.new(name=name)
    obj = bpy.data.objects.new(name, mesh)

    # Link object to the current collection
    bpy.context.collection.objects.link(obj)

    # Set mode to Object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Fill mesh data
    mesh.from_pydata(vert, [], face)
    mesh.update()

    # Create UV map
    if uv is not None:
        mesh.uv_layers.new(name="UVMap")
        uv_layer = mesh.uv_layers.active.data
        # Update the mesh
        mesh.update()

        uv_layer.foreach_set('uv', uv[face].ravel())

    if vert_color is not None:
        vtc = vert_color[face] # (F, 3, 4)
        # add alpha channel if not exists
        if vtc.shape[2] == 3:
            vtc = np.concatenate([vtc, np.ones((*vtc.shape[:2], 1))], axis=-1)
        print(vtc.shape, "@@@@@@@@@@@")
        mesh.vertex_colors.new(name="vertex_color")
        color_layer = mesh.vertex_colors.active.data
        color_layer.foreach_set('color', vtc.ravel())
    # for poly in mesh.polygons:
    #     for loop_index in poly.loop_indices:
    #         vertex_index = mesh.loops[loop_index].vertex_index
    #         uv_layer[loop_index].uv = uv[vertex_index]

    return obj

def if_uv_overlap(obj):
    """Check if the active UV map of the object has overlapping UVs."""
    assert obj.type == 'MESH', "Object must be of type MESH"

    # Switch to Edit Mode
    # deselect all
    set_active_exclusive(obj)
    
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Enable UV Sync Selection
    bpy.context.scene.tool_settings.use_uv_select_sync = True

    # Make sure face selection mode is enabled
    bpy.ops.mesh.select_mode(type="FACE")

    # Deselect all to start fresh
    bpy.ops.mesh.select_all(action='DESELECT')

    # Switch to UV Editor and deselect all
    bpy.ops.uv.select_all(action='DESELECT')

    # Select overlapping UVs
    bpy.ops.uv.select_overlap()

    # Check if any UV face is selected
    overlap_detected = any(face.select for face in bmesh.from_edit_mesh(obj.data).faces)

    # Go back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    return overlap_detected

def smart_uv_project(obj):
    assert obj.type == 'MESH', "Object must be of type MESH"
    set_active_exclusive(obj)
    bpy.ops.object.mode_set(mode='EDIT')

    # Enable UV Sync Selection
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    # select all faces
    bpy.ops.mesh.select_all(action='SELECT')
    # apply smart uv project
    bpy.ops.uv.smart_project(angle_limit=89, island_margin=0.01, use_aspect=False, scale_to_bounds=False)
    # Go back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

def scaling_uv(obj, scale_factor=[1,1,1]):
    assert obj.type == 'MESH', "Object must be of type MESH"
    set_active_exclusive(obj)
    #bpy.ops.object.mode_set(mode='EDIT')

    # # Enable UV Sync Selection
    # bpy.context.scene.tool_settings.use_uv_select_sync = True
    # # select all faces
    # bpy.ops.mesh.select_all(action='SELECT')
    # # select all uv 
    # bpy.ops.uv.select_all(action='SELECT')
    #area_type = bpy.context.area.type
    #print(area_type)
    #bpy.context.area.type = 'IMAGE_EDITOR'
    # apply smart uv project
    #bpy.ops.transform.resize(value=(scale_factor[0], scale_factor[1], 1.))
    #bpy.context.area.type = area_type

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active
    for face in bm.faces:
        for loop in face.loops:
            loop[uv_layer].uv[0] *= scale_factor[0]
            loop[uv_layer].uv[1] *= scale_factor[1]
    bm.to_mesh(obj.data)
    # Free the BMesh data
    bm.free()
    obj.data.update()

    # Go back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

def rescale_uv(obj, min_size=.02):
    vert, face, vert_uv = get_trimesh_uv(obj)
    # get uv scale
    uv_scale = np.max(vert_uv, axis=0) - np.min(vert_uv, axis=0)
    if uv_scale[0] == 0 or uv_scale[1] == 0:
        print("Warning: uv scale is zero, no uv scaling is applied.")
        return
    #new_uv_scale = uv_scale ** exp
    new_uv_scale = np.clip(uv_scale, min_size, None)
    factor = new_uv_scale / uv_scale
    #factor = [2,2]
    #print(factor, new_uv_scale, uv_scale)
    scaling_uv(obj, scale_factor=[factor[0], factor[1]])

def join_objs(objs):
    set_active_exclusive(objs)
    bpy.ops.object.join()
    obj = bpy.context.active_object
    return obj

def get_mesh_data(obj):
    # Apply transformation to get world coordinates
    mesh = obj.to_mesh()
    mesh.transform(obj.matrix_world)

    # Get vertices and faces
    verts = np.array([v.co for v in mesh.vertices])
    faces = np.array([[v for v in p.vertices] for p in mesh.polygons])

    bpy.data.meshes.remove(mesh)
    return verts, faces


@contextlib.contextmanager
def managed_bmesh(mesh): # obj.data or just bpy.data.meshes type 
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    try:
        yield bm
    finally:
        bm.free()

def vfe2mesh(verts, faces, edges=[]):
    new_mesh = bpy.data.meshes.new('new_mesh')
    new_mesh.from_pydata(verts, edges, faces)
    new_mesh.update()
    return new_mesh
def mesh2vfe(mesh):
    # get numpy verts
    verts = np.array([vertex.co for vertex in mesh.vertices])
    edges = [[edge.vertices[0], edge.vertices[1]] for edge in mesh.edges]
    faces = [[vertex for vertex in polygon.vertices] for polygon in mesh.polygons]
    return verts, faces, edges
def triangulate_mesh(mesh):
    with managed_bmesh(mesh) as bm:
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY')
        bm.to_mesh(mesh)
        mesh.update()
    return mesh
def vf2trimesh(vert, face, return_index=True):
    poly_ind = []
    triv, trif = vert, []
    for fi, f in enumerate(face):
        if len(f) > 3:
            tempv = [vert[find] for find in f]
            tempf = [list(range(len(f)))]
            mesh = vfe2mesh(tempv, tempf)
            mesh = triangulate_mesh(mesh)
            tvert, tface, _ = mesh2vfe(mesh)
            otface = [[f[j] for j in tface[i]] for i in range(len(tface))]
            trif += otface
            poly_ind += [fi,] * len(tface)
        elif len(f) == 3:
            trif.append(f)
            poly_ind.append(fi)

    if return_index:
        return np.array(triv), np.array(trif), np.array(poly_ind)

# # Dictionary to store mesh data
# meshes_data = {}

# # Iterate over selected objects
# for obj in bpy.context.selected_objects:
#     if obj.type == 'MESH':
#         verts, faces = get_mesh_data(obj)
#         meshes_data[obj.name] = {"verts": verts, "faces": faces}

# print(meshes_data)
# def export_mesh(obj, outpath=None):
#     bpy.ops.object.select_all(action='DESELECT')
#     obj.select_set(True)
#     num_verts, num_faces = len(obj.data.vertices), len(obj.data.polygons)
    
#     vertices = np.array([vertex.co for vertex in obj.data.vertices])
#     # Get the polygonal face list
#     faces = [[vertex for vertex in polygon.vertices] for polygon in obj.data.polygons]
#     face_normals = np.array([polygon.normal for polygon in obj.data.polygons])
    
#     me = bpy.context.object.data
#     bm = bmesh.new()   # create an empty BMesh
#     bm.from_mesh(me)   # fill it in from a Mesh
    
#     adj_mat = np.zeros((num_faces, num_faces), dtype=bool)
#     for edge in bm.edges:
#         linked_faces = edge.link_faces
#         lfaces = np.array([lf.index for lf in linked_faces])
#         # select submatrix by rows and colums
#         adj_mat[ np.ix_(lfaces, lfaces) ] = True
#     bm.free()
    
#     ditem = dict(verts=vertices, faces=faces, face_normals=face_normals, f_adj = adj_mat)
#     if outpath is not None:
#         np.savez(outpath, **ditem)
#     return ditem

def geometry2world_coordinate(shape):
    set_active_exclusive(shape)
    obj = shape
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def weld_and_decimate(obj, fix_nonmanifold=False, decimate_angle=1):
    bpy.context.view_layer.objects.active = obj
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.edge_split(type='VERT')
    bpy.ops.mesh.delete_loose() # delete loose verts and edges
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # add a solidify modifier on active object
    #weld_modifier = obj.modifiers.new(name="Weld", type='WELD')
    #weld_modifier.merge_threshold = 0.00001
    #bpy.ops.object.modifier_apply(modifier=weld_modifier.name)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles() # remove by distance
    bpy.ops.object.mode_set(mode='OBJECT')
    
#    if fix_nonmanifold==True:
#        # Go into edit mode
#        bpy.ops.object.mode_set(mode='EDIT')
#        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
#        # Deselect everything
#        bpy.ops.mesh.select_all(action='DESELECT')
#        # Select non-manifold edges
#        bpy.ops.mesh.select_non_manifold()
#        # Apply adding face operation (Fill)
#        bpy.ops.mesh.edge_face_add()
#        # Return to object mode
#        bpy.ops.object.mode_set(mode='OBJECT')
#        
#        # apply triangulation to the 
#        planar_decimation_modifier = obj.modifiers.new(name="triangulate", type='TRIANGULATE')
#        planar_decimation_modifier.min_vertices = 4
#        
#        bpy.ops.object.modifier_apply(modifier=planar_decimation_modifier.name)
        
    
    if len(obj.data.polygons) >= 3:
        planar_decimation_modifier = obj.modifiers.new(name="Planar Decimation", type='DECIMATE')
        planar_decimation_modifier.decimate_type = 'DISSOLVE'
        planar_decimation_modifier.angle_limit = (np.pi*2)/360 * decimate_angle
        bpy.ops.object.modifier_apply(modifier=planar_decimation_modifier.name)
    else:
        # enter edit mode
        bpy.ops.object.mode_set(mode='EDIT')
        # select all faces
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.dissolve_limited()
        # return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')


# Function to decimate a mesh to a target number of faces
def decimate_mesh_to_facenum(obj, target_faces=800, if_weld=False):
    # Select the object
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    if if_weld:
        # Weld the vertices
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles() # remove by distance
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add a Decimate modifier to the object
    mod = obj.modifiers.new(name="Decimation", type='DECIMATE')
    
    # Calculate the ratio based on current and target face count
    current_faces = len(obj.data.polygons)
    ratio = target_faces / current_faces
    mod.ratio = ratio

    # Set the modifier to use the 'COLLAPSE' mode and face count limit
    mod.decimate_type = 'COLLAPSE'
    mod.use_collapse_triangulate = True  # Ensures the mesh remains manifold
    
    # Apply the modifier
    bpy.ops.object.modifier_apply(modifier=mod.name)

def decimate_trimesh_to_facenum(vert, face, **kwargs):
    obj = mesh_from_pydata(vert, face)
    decimate_mesh_to_facenum(obj, **kwargs)
    vert, face = get_trimesh(obj)
    # remove the temp mesh
    bpy.data.objects.remove(obj)
    return vert, face

def preprocess_glb(glb_path):
    clear_all()
    load_glb(glb_path)

    # First weld, then fix nonmanifold, triangulate again, then 
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            weld_and_decimate(obj, fix_nonmanifold=False)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()
    
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    # clear the glb's default axes tree structure.
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # The following two recalculate the obj center. the Median one is desired.
    #bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='BOUNDS')
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

    
    max_extent = max(obj.dimensions)
    scale_factor = 1.8 / max_extent if max_extent != 0 else 1

    # Apply the scaling
    obj.scale = (scale_factor, scale_factor, scale_factor)
    obj.location = (0, 0, 0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj

# Now bm contains the mesh defined by vertices and faces
def export_mesh(obj, outpath=None):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    num_verts, num_faces = len(obj.data.vertices), len(obj.data.polygons)
    
    vertices = np.array([vertex.co for vertex in obj.data.vertices])
    # Get the polygonal face list
    faces = [[vertex for vertex in polygon.vertices] for polygon in obj.data.polygons]
    face_normals = np.array([polygon.normal for polygon in obj.data.polygons])
    vertices = vertices[:,[0,2,1]]
    face_normals = face_normals[:,[0,2,1]]
    
    me = bpy.context.object.data
    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(me)   # fill it in from a Mesh
    
    adj_mat = np.zeros((num_faces, num_faces), dtype=bool)
    for edge in bm.edges:
        linked_faces = edge.link_faces
        lfaces = np.array([lf.index for lf in linked_faces])
        # select submatrix by rows and colums
        adj_mat[ np.ix_(lfaces, lfaces)] = True
    bm.free()
    
    ditem = dict(verts=vertices, faces=faces, face_normals=face_normals, f_adj = adj_mat)
    if outpath is not None:
        np.savez(outpath, **ditem)
    return ditem

def glb2polygon(glb_path, outpath):
    #bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    obj = preprocess_glb(glb_path)
    ditem = export_mesh(obj, outpath = outpath)
    return ditem
#glb2polygon


def glb2trimesh(path):
    clear_all()
    load_glb(path)
    obj = bpy.context.view_layer.objects.active
    verts, faces, uvs = get_trimesh(obj, return_uv=True)
    return verts, faces, uvs






















# import bpy

# import re
# import os
# import sys
# import numpy as np

# import bpy
# import bmesh

# def create_bmesh_from_verts_faces(verts, faces):
#     # Create a new BMesh
#     bm = bmesh.new()
#     # Add vertices to the BMesh
#     for v in verts:
#         bm.verts.new(v)
#     # Add faces to the BMesh
#     for f in faces:
#         bm.faces.new([bm.verts[i] for i in f])

#     return bm


# def clear_all():
#     # Deselect all objects
#     bpy.ops.object.select_all(action='DESELECT')

#     # Select all objects
#     bpy.ops.object.select_all(action='SELECT')

#     # Delete selected objects
#     bpy.ops.object.delete()
# def set_active_exclusive(obj):
#     bpy.ops.object.select_all(action='DESELECT')
#     obj.select_set(True)
#     bpy.context.view_layer.objects.active = obj
# def list_objects():
#     for obj in bpy.data.objects:
#         print(obj.name)
    
# def boolean_union(obj1, obj2):
#     mod = obj1.modifiers.new(name='Boolean', type='BOOLEAN')
#     mod.operation = 'UNION'
#     mod.use_self = False
#     mod.object = obj2
#     set_active_exclusive(obj1)
#     bpy.ops.object.modifier_apply(modifier=mod.name)

# def get_face_adj_mat(obj):
#     num_verts, num_faces = len(obj.data.vertices), len(obj.data.polygons)
#     me = obj.data
#     bm = bmesh.new()   # create an empty BMesh
#     bm.from_mesh(me)   # fill it in from a Mesh
#     adj_mat = np.zeros((num_faces, num_faces), dtype=bool)
#     for edge in bm.edges:
#         linked_faces = edge.link_faces
#         lfaces = np.array([lf.index for lf in linked_faces])
#         # select submatrix by rows and colums
#         adj_mat[ np.ix_(lfaces, lfaces)] = True
#     bm.free()
#     return adj_mat

# def get_polygon(obj):
#     num_verts, num_faces = len(obj.data.vertices), len(obj.data.polygons)
#     vertices = np.array([vertex.co for vertex in obj.data.vertices])
#     # Get the polygonal face list
#     faces = [[vertex for vertex in polygon.vertices] for polygon in obj.data.polygons]
#     face_normals = np.array([polygon.normal for polygon in obj.data.polygons])
#     return vertices, faces, face_normals

# def get_trimesh(obj):
#     """ triangulate the mesh and return vert and face as numpy array"""
#     bm = bmesh.new()
#     bm.from_mesh(obj.data)
#     # Triangulate the mesh
#     bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY')
#     # Get the vertices and faces
#     verts = [vert.co for vert in bm.verts]
#     faces = [[vert.index for vert in face.verts] for face in bm.faces]
#     # Free the BMesh data
#     bm.free()
#     return verts, faces

# # def export_mesh(obj, outpath=None):
# #     bpy.ops.object.select_all(action='DESELECT')
# #     obj.select_set(True)
# #     num_verts, num_faces = len(obj.data.vertices), len(obj.data.polygons)
    
# #     vertices = np.array([vertex.co for vertex in obj.data.vertices])
# #     # Get the polygonal face list
# #     faces = [[vertex for vertex in polygon.vertices] for polygon in obj.data.polygons]
# #     face_normals = np.array([polygon.normal for polygon in obj.data.polygons])
    
# #     me = bpy.context.object.data
# #     bm = bmesh.new()   # create an empty BMesh
# #     bm.from_mesh(me)   # fill it in from a Mesh
    
# #     adj_mat = np.zeros((num_faces, num_faces), dtype=bool)
# #     for edge in bm.edges:
# #         linked_faces = edge.link_faces
# #         lfaces = np.array([lf.index for lf in linked_faces])
# #         # select submatrix by rows and colums
# #         adj_mat[ np.ix_(lfaces, lfaces) ] = True
# #     bm.free()
    
# #     ditem = dict(verts=vertices, faces=faces, face_normals=face_normals, f_adj = adj_mat)
# #     if outpath is not None:
# #         np.savez(outpath, **ditem)
# #     return ditem

# def geometry2world_coordinate(shape):
#     set_active_exclusive(shape)
#     obj = shape
#     bpy.context.scene.cursor.location = (0, 0, 0)
#     bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
#     bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
#     return obj


# def load_glb(path):
#     bpy.ops.import_scene.gltf(filepath=path, import_shading='FLAT')

# def weld_and_decimate(obj, fix_nonmanifold=False, decimate_angle=1):
#     bpy.context.view_layer.objects.active = obj
    

    
    
#     bpy.ops.object.mode_set(mode='EDIT')
#     bpy.ops.mesh.select_all(action='SELECT')
#     bpy.ops.mesh.edge_split(type='VERT')
#     bpy.ops.mesh.delete_loose() # delete loose verts and edges
#     bpy.ops.object.mode_set(mode='OBJECT')
    
    
#     # add a solidify modifier on active object
#     #weld_modifier = obj.modifiers.new(name="Weld", type='WELD')
#     #weld_modifier.merge_threshold = 0.00001
#     #bpy.ops.object.modifier_apply(modifier=weld_modifier.name)
#     bpy.ops.object.mode_set(mode='EDIT')a
#     bpy.ops.mesh.select_all(action='SELECT')
#     bpy.ops.mesh.remove_doubles() # remove by distance
#     bpy.ops.object.mode_set(mode='OBJECT')
    
# #    if fix_nonmanifold==True:
# #        # Go into edit mode
# #        bpy.ops.object.mode_set(mode='EDIT')
# #        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
# #        # Deselect everything
# #        bpy.ops.mesh.select_all(action='DESELECT')
# #        # Select non-manifold edges
# #        bpy.ops.mesh.select_non_manifold()
# #        # Apply adding face operation (Fill)
# #        bpy.ops.mesh.edge_face_add()
# #        # Return to object mode
# #        bpy.ops.object.mode_set(mode='OBJECT')
# #        
# #        # apply triangulation to the 
# #        planar_decimation_modifier = obj.modifiers.new(name="triangulate", type='TRIANGULATE')
# #        planar_decimation_modifier.min_vertices = 4
# #        
# #        bpy.ops.object.modifier_apply(modifier=planar_decimation_modifier.name)
        
    
#     planar_decimation_modifier = obj.modifiers.new(name="Planar Decimation", type='DECIMATE')
#     planar_decimation_modifier.decimate_type = 'DISSOLVE'
#     planar_decimation_modifier.angle_limit = (np.pi*2)/360 * decimate_angle
    
#     bpy.ops.object.modifier_apply(modifier=planar_decimation_modifier.name)

# def preprocess_glb(glb_path):
#     clear_all()
#     load_glb(glb_path)

#     # First weld, then fix nonmanifold, triangulate again, then 
#     bpy.ops.object.select_all(action='DESELECT')
#     for obj in bpy.context.scene.objects:
#         if obj.type == 'MESH':
#             obj.select_set(True)
#             weld_and_decimate(obj, fix_nonmanifold=True)
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.join()
    
#     obj = bpy.context.view_layer.objects.active
#     bpy.ops.object.select_all(action='DESELECT')
#     obj.select_set(True)
#     # clear the glb's default axes tree structure.
#     bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

#     bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
#     # The following two recalculate the obj center. the Median one is desired.
#     #bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='BOUNDS')
#     #bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
#     bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

#     #bbox_center = [sum(axis) / 8 for axis in zip(*obj.bound_box)]
#     #obj.location = bbox_center
#     #obj.data.update()
    
#     max_extent = max(obj.dimensions)
#     scale_factor = 1.8 / max_extent if max_extent != 0 else 1

#     # Apply the scaling
#     obj.scale = (scale_factor, scale_factor, scale_factor)
#     obj.location = (0, 0, 0)
#     bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
#     return obj

# # Now bm contains the mesh defined by vertices and faces
# def export_mesh(obj, outpath=None):
#     bpy.ops.object.select_all(action='DESELECT')
#     obj.select_set(True)
#     num_verts, num_faces = len(obj.data.vertices), len(obj.data.polygons)
    
#     vertices = np.array([vertex.co for vertex in obj.data.vertices])
#     # Get the polygonal face list
#     faces = [[vertex for vertex in polygon.vertices] for polygon in obj.data.polygons]
#     face_normals = np.array([polygon.normal for polygon in obj.data.polygons])
    
#     me = bpy.context.object.data
#     bm = bmesh.new()   # create an empty BMesh
#     bm.from_mesh(me)   # fill it in from a Mesh
    
#     adj_mat = np.zeros((num_faces, num_faces), dtype=bool)
#     for edge in bm.edges:
#         linked_faces = edge.link_faces
#         lfaces = np.array([lf.index for lf in linked_faces])
#         # select submatrix by rows and colums
#         adj_mat[ np.ix_(lfaces, lfaces)] = True
#     bm.free()
    
#     ditem = dict(verts=vertices, faces=faces, face_normals=face_normals, f_adj = adj_mat)
#     if outpath is not None:
#         np.savez(outpath, **ditem)
#     return ditem

# def glb2polygon(glb_path, outpath):
#     #bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
#     obj = preprocess_glb(glb_path)
#     ditem = export_mesh(obj, outpath = outpath)
#     return ditem
# glb2polygon(glb_path=r"C:\Users\星光\Downloads\glbs\000-003\88ce158336f246af983a1ec51ebc4271.glb", outpath=None)


