import numpy as np
import fresnel
import matplotlib.pyplot as plt
import math
import copy
from scipy.spatial.transform import Rotation as R
from skimage.color import rgba2rgb, rgb2gray
import igl

from xgutils import nputil, geoutil

with nputil.temp_seed(315):
    unique_colors = np.random.rand( 100000, 3)
unique_colors[:4] = np.array([[1.,0,0],[0,1,0],[0,0,1],[1,1,0]])

dflt_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
    camUp=np.array([0,1,0]),camHeight=2, fit_camera=False, 
    camera_type = "orthographic", \
    light_samples=32, samples=32, resolution=(256, 256))
dflt_camera["camHeight"] = 3.
dflt_camera["resolution"] = (256, 256)
dfltw_camera = copy.deepcopy(dflt_camera)
dfltw_camera["camHeight"] = 3.
HD_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
    camUp=np.array([0,1,0]),camHeight=2, fit_camera=False, 
    camera_type = "orthographic", \
    light_samples=32, samples=256, resolution=(1024,1024))
gold_color = np.array([253, 204, 134])/256
gray_color = np.array([0.9, 0.9, 0.9])
white_color= np.array([1,1,1.])
black_color= np.array([0,0,0.])
red_color  = np.array([1.,  0.,  0.])

voxel_mat = dict(specular=.5, roughness=.5, metal=1., spec_trans=0.)
#default_mat = dict(specular=.5, roughness=.5, metal=1., spec_trans=0.)
default_mesh_mat = dict(color = gray_color, solid=0., roughness=.2, specular=.8, spec_trans=0., metal=0.)
light_preset = ["lightbox", "Cloudy", "Rembrandt", "loop", "butterfly", "ring"]

def render_test_scene():
    camera_kwargs = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
        camUp=np.array([0,1,0]),camHeight=2, fit_camera=False, \
        light_samples=32, samples=32, resolution=(256,256))

    renderer = FresnelRenderer(camera_kwargs=camera_kwargs)
    renderer.add_box(center=[0,0,0])
    img = renderer.render()
    return img
def addAxes(scene, radius=[0.01,0.01,0.01]):
    axs = fresnel.geometry.Cylinder(scene, N=3)
    axs.material = fresnel.material.Material(solid=1.)
    axs.material.primitive_color_mix = 1.0
    axs.points[:] = [[[0,0,0],[1,0,0]],
                        [[0,0,0],[0,1,0]],
                        [[0,0,0],[0,0,1]]]
    axs.radius[:] = radius
    axs.color[:] =  [[[1,0,0],[1,0,0]],
                        [[0,1,0],[0,1,0]],
                        [[0,0,1],[0,0,1]]]
    return axs
def addArrows(scene, starts, ends, radius=0.01, solid=0., values=None):
    N = starts.shape[0]
    if values is None:
        values = np.linspace(0,1,N)

    axs = fresnel.geometry.Cylinder(scene, N=N)
    axs.material = fresnel.material.Material(solid=solid)
    axs.material.primitive_color_mix = 1.0
    points = np.array([starts,ends]).transpose(1,0,2)
    axs.points[:] = points
    axs.radius[:] = radius
    # get cmap from matplotlib and values:
    cmap = plt.get_cmap('viridis')
    colors = cmap(values)
    axs.color[:] = colors[:,None,:3][:,[0,0],:]

    
    heads = fresnel.geometry.Sphere(scene, N=N)
    heads.material = fresnel.material.Material(solid=solid)
    heads.material.primitive_color_mix = 1.0
    points = np.array([starts,ends]).transpose(1,0,2)
    heads.position[:] = points[:,1,:]
    heads.radius[:] = radius*2
    # get cmap from matplotlib and values:
    cmap = plt.get_cmap('viridis')
    colors = cmap(values)
    heads.color[:] = colors[:,:3]
    return axs, heads
def addBBox(scene, bb_min=np.array([-1,-1,-1.]), bb_max=np.array([1,1,1.]), color=red_color, radius=0.003, solid=1.):
    axs = fresnel.geometry.Cylinder(scene, N=12)
    axs.material = fresnel.material.Material(   color = fresnel.color.linear(color),
                                                solid=solid,
                                                spec_trans=.4)
    #axs.material.primitive_color_mix = 1.0
    pts = []
    xi,yi,zi = bb_min
    xa,ya,za = bb_max
    axs.points[:] = [   [[xi,yi,zi],[xa,yi,zi]],
                        [[xi,yi,zi],[xi,ya,zi]],
                        [[xi,yi,zi],[xi,yi,za]], # 
                        [[xi,ya,za],[xa,ya,za]],
                        [[xi,ya,za],[xi,yi,za]],
                        [[xi,ya,za],[xi,ya,zi]], #
                        [[xa,ya,zi],[xi,ya,zi]],
                        [[xa,ya,zi],[xa,yi,zi]],
                        [[xa,ya,zi],[xa,ya,za]], #
                        [[xa,yi,za],[xi,yi,za]], 
                        [[xa,yi,za],[xa,ya,za]],
                        [[xa,yi,za],[xa,yi,zi]], #
                    ]
    axs.radius[:] = radius
    axs.color[:] =  [ [[.5,0,0],[.5,0,0]] ] * 12
    return axs
def addBox(scene, center, spec=(1,1,1), color=gray_color, solid=0., 
            outline_width=0., metal=0., specular=0.0, roughness=1.0, **kwargs):
    X, Y, Z = spec[0], spec[1], spec[2]
    poly_info = fresnel.util.convex_polyhedron_from_vertices([
        [-X, -Y, -Z],
        [-X, -Y, Z],
        [-X, Y, -Z],
        [-X, Y, Z],
        [X, -Y, -Z],
        [X, -Y, Z],
        [X, Y, -Z],
        [X, Y, Z],
    ])
    geometry = fresnel.geometry.ConvexPolyhedron(scene,
                                             poly_info,
                                             position=center,
                                             outline_width=outline_width)#0.015)
    geometry.material = fresnel.material.Material( \
                                roughness=roughness,
                                solid = solid,
                                specular=specular,
                                metal = metal, **kwargs)
    geometry.material.primitive_color_mix = 1.0
    geometry.material.color   = fresnel.color.linear([1,1,1])
    geometry.color[:]         = color
    geometry.outline_material = fresnel.material.Material( \
                                color     = fresnel.color.linear([0,0,0]),
                                roughness = 0.3,
                                metal     =  .0)
    #geometry.color[:] = color
    geometry.outline_material.primitive_color_mix = .7
    geometry.outline_material.solid  = 0.0
    return geometry
def addPlane(scene, center, up=(0,1,0), spec=(1,1), color=white_color, solid=0., **kwargs):
    X, Z = spec[0], spec[1]
    poly_info = np.array([
        [-X, 0, -Z],
        [ X, 0, -Z],
        [ X, 0,  Z],
        [-X, 0,  Z]      ])
    vertices = poly_info[[0,1,3, 3,1,2]]
    geometry = fresnel.geometry.Mesh(scene, N=1, vertices = vertices,
                                     position=center,
                                     outline_width=0)
    geometry.material = fresnel.material.Material( \
                                roughness=1.0,
                                specular=0.,
                                color=color,
                                solid=solid)
    geometry.material.primitive_color_mix = 0.0 #Set 0 to use the color specified in the Material, 
    return geometry
def add_error_cloud(scene, cloud, radius=0.006, color=None, solid=0., name=None):
    cloud = fresnel.geometry.Sphere(scene, position = cloud, radius=radius)
    cloud_flat_color = gold_color
    if color is not None and len(np.array(color).shape)==1:
        cloud_flat_color = color
    cloud.material = fresnel.material.Material(solid=solid, \
                                                color=fresnel.color.linear(cloud_flat_color),\
                                                roughness=1.0,
                                                specular=0.0)
    if color is not None and len(np.array(color).shape)>1:
        cloud.material.primitive_color_mix = 1.0
        cloud.color[:] = fresnel.color.linear(plt.cm.plasma(color)[:,:3])
    return cloud
def add_cloud(scene, cloud, radius=0.006, color=None, solid=0., primitive_color_mix=1., cloud_flat_color = gold_color, 
        roughness=.2, specular=.8, spec_trans=0., metal=0., name=None):
    cloud = fresnel.geometry.Sphere(scene, position = cloud, radius=radius)
    
    if color is not None and len(np.array(color).shape)==1:
        cloud_flat_color = color
    cloud.material = fresnel.material.Material(solid=solid, \
                                                color=fresnel.color.linear(cloud_flat_color),\
                                                roughness=roughness,
                                                specular=specular,
                                                metal=metal,
                                                spec_trans=spec_trans)
    if color is not None and len(np.array(color).shape)>1:
        cloud.material.primitive_color_mix = primitive_color_mix
        cloud.color[:] = fresnel.color.linear(color)
    return cloud
def add_lines(scene, lines, radius=0.006, color=None, solid=0., primitive_color_mix=1., cloud_flat_color = gold_color, 
        roughness=.2, specular=.8, spec_trans=0., metal=0., name=None):
    lines = fresnel.geometry.Cylinder(scene, points = lines, radius=radius)

    if color is not None and len(np.array(color).shape)==1:
        lines_flat_color = color
    lines.material = fresnel.material.Material(solid=solid, \
                                                color=fresnel.color.linear(lines_flat_color),\
                                                roughness=roughness,
                                                specular=specular,
                                                metal=metal,
                                                spec_trans=spec_trans)
    if color is not None and len(np.array(color).shape)>1:
        lines.material.primitive_color_mix = primitive_color_mix
        lines.color[:] = fresnel.color.linear(color)
    return lines

def add_mesh(scene, vert, face, outline_width=None, name=None,
        color = gray_color, vert_color=None, vert_color_scheme=None, solid=0., roughness=.2, specular=.8, spec_trans=0., metal=0.) :
    """ vert_color: (Vn, 4) """
    mesh = fresnel.geometry.Mesh(scene,vertices=vert[face].reshape(-1,3) ,N=1)
    mesh.material = fresnel.material.Material(color=fresnel.color.linear(color),
                                                solid=solid,
                                                roughness=roughness,
                                                specular=specular,
                                                spec_trans=spec_trans,
                                                metal=metal)
    if vert_color is not None:
        mesh.color[:] = fresnel.color.linear(vert_color)
        mesh.material.primitive_color_mix = 1.0
    elif vert_color_scheme is not None:
        if vert_color_scheme == "normal":
            normals = igl.per_vertex_normals(vert, face)
            vert_color = (normals[face].reshape(-1,3)+1)/2.
        mesh.color[:] = fresnel.color.linear(vert_color)
        mesh.material.primitive_color_mix = 1.0
    if outline_width is not None:
        mesh.outline_width = outline_width
    return mesh

def add_trajectory(scene, translations, rotations=None):
    add_cloud(scene, translations, radius=0.01, color=red_color, solid=1., primitive_color_mix=0.)
    traj = fresnel.geometry.Cylinder(scene, N=len(translations))
    traj.material = fresnel.material.Material(solid=1.)
    traj.material.primitive_color_mix = 1.0
    traj.points[:] = [[translations[i], translations[i+1]] for i in range(len(translations)-1)]
    traj.radius[:] = 0.01
    traj.color[:] =  [[[1,0,0],[1,0,0]]] * (len(translations)-1)
    return traj
def get_cam2world(camera, lookat=np.array([0,0,0]), up=np.array([0,1,0])):
    """ get camera to world transformation """
    shift = -camera
    z_axis = -lookat + camera  # +z
    x_axis = np.cross(up, z_axis)
    y_axis = np.cross(z_axis, x_axis)
    x_axis = x_axis / np.sqrt(np.sum(x_axis**2))
    y_axis = y_axis / np.sqrt(np.sum(y_axis**2))
    z_axis = z_axis / np.sqrt(np.sum(z_axis**2))
    rot = np.array([x_axis, y_axis, z_axis]).transpose()
    return shift, rot
def get_world2cam(camera, lookat=np.array([0,0,0]), up=np.array([0,1,0])):
    shift, rot = get_cam2world(camera, lookat, up)
    return -shift, rot.transpose()

def world2camera(point, camera):
    """ lookat transform """
    point, camera = np.array(point), np.array(camera)
    shift, rot = get_cam2world(camera)
    rot = R.from_matrix( rot.transpose() )
    return rot.apply(point)
def add_world_light(scene, direction, camera_pos, color, theta=1.0):
    world_dir = direction
    cam_dir = world2camera( world_dir, camera_pos )
    new_light = fresnel.light.Light(direction= cam_dir, color=color, theta=theta)
    scene.lights.append( new_light )
    return new_light
def get_world_lights(directions, colors, thetas, camera_pos):
    lights = []
    for i, direction in enumerate(directions):
        world_dir = direction
        cam_dir = world2camera( world_dir, camera_pos )
        new_light = fresnel.light.Light(direction= cam_dir, color=colors[i], theta=thetas[i])
        lights.append( new_light )
    return lights

def old_renderMeshCloud(    mesh=None, meshC=gray_color, mesh_outline_width=None, meshflat=False,  # mesh settings
                        cloud=None, cloudR=0.006, cloudC=None,  # pc settings
                        camPos=None, camLookat=None, camUp=np.array([0,0,1]), camHeight=1.,  # camera settings
                        samples=32, axes=False, bbox=False, resolution=(1024,1024),  # render settings
                        lights="rembrandt", **kwargs):
    device = fresnel.Device()
    scene = fresnel.Scene(device)
    if mesh is not None and mesh['vert'].shape[0]>0:
        mesh = fresnel.geometry.Mesh(scene,vertices=mesh['vert'][mesh['face']].reshape(-1,3) ,N=1)
        mesh.material = fresnel.material.Material(color=fresnel.color.linear(meshC),
                                                    roughness=0.3,
                                                    specular=1.,
                                                    spec_trans=0.)
        if mesh_outline_width is not None:
            mesh.outline_width = mesh_outline_width
    if cloud is not None and cloud.shape[0]>0:
        cloud = fresnel.geometry.Sphere(scene, position = cloud, radius=cloudR)
        solid = .7 if mesh is not None else 0.
        cloud_flat_color = gold_color
        if cloudC is not None and len(np.array(cloudC).shape)==1:
            cloud_flat_color = cloudC
        cloud.material = fresnel.material.Material(solid=solid, \
                                                    color=fresnel.color.linear(cloud_flat_color),\
                                                    roughness=1.0,
                                                    specular=1.0)
        if cloudC is not None and len(np.array(color).shape)>1:
            cloud.material.primitive_color_mix = 1.0
            cloud.color[:] = fresnel.color.linear(plt.cm.plasma(cloudC)[:,:3])
    if axes == True:
        addAxes(scene)
    if bbox == True:
        addBBox(scene)
    if camPos is None or camLookat is None:
        print("Fitting")
        scene.camera = fresnel.camera.fit(scene,margin=0)
    else:
        scene.camera = fresnel.camera.Orthographic(camPos, camLookat, camUp, camHeight) # fresnel==0.13.4
    if lights == "cloudy":
        scene.lights = fresnel.light.cloudy()
    if lights == "rembrandt":
        scene.lights = fresnel.light.rembrandt()
    if lights == "lightbox":
        scene.lights = fresnel.light.lightbox()
    if lights == "loop":
        scene.lights = fresnel.light.loop()
    if lights == "butterfly":
        scene.lights = fresnel.light.butterfly()
    #scene.lights[0].theta = 3

    tracer = fresnel.tracer.Path(device=device, w=resolution[0], h=resolution[1])
    tracer.sample(scene, samples=samples, light_samples=32)
    #tracer.resize(w=450, h=450)
    #tracer.aa_level = 3
    image = tracer.render(scene)[:]
    return image
def renderMeshCloud(mesh=None, meshC=gray_color, mesh_outline_width=None, meshflat=False,  # mesh settings
                        cloud=None, cloudR=0.006, cloudC=None,  # pc settings
                        camPos=None, camLookat=None, camUp=np.array([0,0,1]), camHeight=1.,  # camera settings
                        samples=32, axes=False, bbox=False, resolution=(1024,1024),  # render settings
                        lights="rembrandt", **kwargs):
    camera_opt = dict(resolution=resolution, samples=samples, \
        camPos=camPos, camLookat=camLookat, camUp=camUp, camHeight=camHeight)
    
    renderer = FresnelRenderer(lights=lights, camera_kwargs=camera_opt)
    if axes == True:
        renderer.addAxes()
    if bbox == True:
        renderer.addBBox()
    if mesh is not None and mesh['vert'].shape[0]>0:
        renderer.add_mesh(mesh["vert"], mesh["face"], color=meshC, outline_width=mesh_outline_width)
    if cloud is not None and cloud.shape[0]>0:
        renderer.add_cloud(cloud, radius=cloudR, color = cloudC)
    image = renderer.render()
    return image
def renderMeshCloud2(mesh=None, meshC=gray_color, mesh_outline_width=None, meshflat=False,  # mesh settings
                        cloud=None, cloudR=0.006, cloudC=None,  # pc settings
                        camHeight=1.,  # camera settings
                        axes=False, bbox=False,  # render settings
                        camera_kwargs={},
                        **kwargs):
    camera_opt = dflt_camera 
    camera_opt.update(camera_kwargs)
    
    renderer = FresnelRenderer(camera_kwargs=camera_opt)
    if axes == True:
        renderer.addAxes()
    if bbox == True:
        renderer.addBBox()
    if mesh is not None and mesh['vert'].shape[0]>0:
        renderer.add_mesh(mesh, color=meshC, outline_width=mesh_outline_width)
    if cloud is not None and cloud.shape[0]>0:
        renderer.add_cloud(cloud, radius=cloudR, color = cloudC)
    image = renderer.render()
    return image

def render_mesh(vert, face, fvert=None, camera_kwargs={}, render_kwargs={}, shadow_catcher=False, show_axes=False, show_bbox=False, lights="rembrandt", **kwargs):
    renderer = FresnelRenderer(camera_kwargs=camera_kwargs, show_axes=show_axes, show_bbox=show_bbox, lights=lights)
    renderer.add_mesh(vert, face, fvert, **kwargs)

    if shadow_catcher==True:
        min_y = vert.min(axis=0)[1] if fvert is None else fvert.min(axis=0)[1]
        img = renderer.render(shadow_catcher=True, min_y=min_y, **render_kwargs)
    else:
        img = renderer.render(**render_kwargs)
    return img
def render_fvert(fvert, face, preview=True, camera_kwargs=dflt_camera, **kwargs):
    """ vert: (4, 3)
        face: (4, 3)
    """
    assert len(fvert) % 3 == 0
    fnum = len(fvert) // 3
    vert_color = unique_colors[ np.repeat(np.arange(fnum), 3) ]
    img = render_mesh(render_kwargs=dict(preview=preview), vert=None, face=None, fvert=fvert, vert_color=vert_color, shadow_catcher=True, camera_kwargs=camera_kwargs, show_axes=True)

    return img
def render_poly_mesh(vert, face, preview=True, camera_kwargs=dflt_camera, triangulation="naive", **kwargs):
    """ vert: (4, 3)
        face: (4, 3)
    """
    if triangulation == "naive":
        tri_face, poly_ind = geoutil.triangulate_poly_face(face, return_index=True)
    elif triangulation == "blender":
        from xgutils import bpyutil
        tri_vert, tri_face, poly_ind = bpyutil.vf2trimesh(vert, face, return_index=True)
    fvert = vert[tri_face].reshape(-1,3)
    fnum = len(face)
    with nputil.temp_seed(315):
        unique_colors = np.random.rand( max(10000, fnum), 3)
    unique_colors[:4] = np.array([[1.,0,0],[0,1,0],[0,0,1],[1,1,0]])

   # indices_tensor = torch.arange(poly_ind.max()+1).to(face)
    #color_ind = indices_tensor.repeat_interleave(poly_ind)

    #print(fvert.shape, poly_ind.shape, tri_face.shape, len(face), vert.shape)
    c_ind = np.repeat(poly_ind, 3)
    vert_color = unique_colors[ c_ind ]
    img = render_mesh(render_kwargs=dict(preview=preview), vert=None, face=None, fvert=fvert, vert_color=vert_color, shadow_catcher=True, camera_kwargs=camera_kwargs, show_axes=True)#, outline_width=0.002)

    return img
def render_polyloop(vert, face, modes=["vert","edge"], centroid=None, preview=True, camera_kwargs=dflt_camera, **kwargs):
    lines = []
    pind = []
    for fi in range(len(face)):
        st_ind  = face[fi]
        end_ind = np.roll(face[fi], -1)
        start_point = vert[st_ind]
        end_point = vert[end_ind]
        ls = np.stack([start_point, end_point], axis=1)
        lines.append(ls)
        pind.append( np.zeros(len(face[fi]), dtype=np.int32) + fi )
    #print(lines[0])
    lines = np.concatenate(lines, axis=0)
    pind = np.concatenate(pind, axis=0)
    pcolors = unique_colors[pind]

    renderer = FresnelRenderer(camera_kwargs, **kwargs)
    if "vert" in modes:
        renderer.add_cloud(vert, radius=.009, color=pcolors*1.1)

    if centroid is not None:
        color = unique_colors[ np.arange(len(centroid), dtype=int) ]
        renderer.add_cloud(centroid, radius=.02, color=color)
    
    if "edge" in modes:
        pcolors = pcolors[:,None,:].repeat(2, axis=1)
        renderer.add_lines(lines, radius=.006, color=pcolors*.9, roughness=.1, specular=1., metal=.1)
    if "face" in modes:
        tri_face, poly_ind = geoutil.triangulate_poly_face(face, return_index=True)
        try:
            from xgutils import bpyutil
            lface=face
            # for f in face:
            #     f = np.array(f)
            #     lface.append( np.r_[f, f[0]] )
            tri_vert, tri_face, poly_ind = bpyutil.vf2trimesh(vert, lface, return_index=True)
        except:
            pass

        fvert = vert[tri_face].reshape(-1,3)
        c_ind = np.repeat(poly_ind, 3)
        vert_color = unique_colors[ c_ind ]
        renderer.add_mesh(vert=None, face=None, fvert=fvert, vert_color=vert_color)
    img = renderer.render(preview=preview)
    return img
    
def render_cloud(cloud, camera_kwargs={}, render_kwargs={}, **kwargs):
    renderer = FresnelRenderer(camera_kwargs=camera_kwargs, **kwargs)
    renderer.add_cloud(cloud=cloud, **kwargs)
    img = renderer.render(**render_kwargs)
    return img
def render_lines(lines, camera_kwargs={}, render_kwargs={}, **kwargs):
    renderer = FresnelRenderer(camera_kwargs=camera_kwargs)
    renderer.add_lines(lines=lines, **kwargs)
    img = renderer.render(**render_kwargs)
    return img

class FresnelRenderer():
    def __init__(self, camera_kwargs={}, lights="rembrandt", show_axes=True, show_bbox=True, **kwargs):
        self.setup_scene(camera_kwargs=camera_kwargs, lights=lights, show_axes=show_axes, show_bbox=show_bbox)
    def setup_camera(self, camera_kwargs={}):
        scene = self.scene
        self.camera_opt = camera_opt = copy.deepcopy(dflt_camera)
        camera_opt.update(camera_kwargs)
        self.camera_kwargs = camera_opt

        if camera_opt["fit_camera"]==True:
            print("Camera is not setup, now auto-fit camera")
            scene.camera = fresnel.camera.fit(scene,margin=0)
        else:
            if camera_opt["camera_type"] == "perspective":
                camPos    = camera_opt["camPos"]
                camLookat = camera_opt["camLookat"]
                camUp     = camera_opt["camUp"]
                camHeight = camera_opt["camHeight"]
                focal_length = camera_opt.get("focal_length", 4)
                scene.camera = fresnel.camera.Perspective(camPos, camLookat, camUp, 
                                focal_length=focal_length,
                                focus_distance = 200,
                                height=camHeight)
                # TODO
            elif camera_opt["camera_type"] == "orthographic":
                camPos    = camera_opt["camPos"]
                camLookat = camera_opt["camLookat"]
                camUp     = camera_opt["camUp"]
                camHeight = camera_opt["camHeight"]
                if hasattr(fresnel.camera, "Orthographic"): # fresnel==0.13.4
                    scene.camera = fresnel.camera.Orthographic(camPos, camLookat, camUp, camHeight)
                elif hasattr(fresnel.camera, "orthographic"):
                    scene.camera = fresnel.camera.orthographic(camPos, camLookat, camUp, camHeight)
                else:
                    raise NotImplementedError

    def setup_scene(self, camera_kwargs={}, lights="rembrandt", show_axes=False, show_bbox=False):
        device = fresnel.Device()
        self.scene = scene = fresnel.Scene(device)
        self.scene.background_color = fresnel.color.linear([1.,1.,1.])
        self.setup_camera(camera_kwargs=camera_kwargs)
        # setup lightings
        if "lights" in camera_kwargs:
            lights = camera_kwargs["lights"]
        if type(lights) is not str:
            scene.lights = camera_kwargs["lights"]
        elif lights == "cloudy":
            scene.lights = fresnel.light.cloudy()
        elif lights == "rembrandt":
            scene.lights = fresnel.light.rembrandt()
        elif lights == "lightbox":
            scene.lights = fresnel.light.lightbox()
        elif lights == "loop":
            scene.lights = fresnel.light.loop()
        elif lights == "butterfly":
            scene.lights = fresnel.light.butterfly()
        elif lights == "up":
            scene.lights = get_world_lights([np.array([0,1,0])], colors=[np.array([1,1,1])], thetas=[1.], camera_pos=camPos)
        if show_axes == True:
            addAxes(scene)
        if show_bbox == True:
            addBBox(scene)
        self.scene, self.device = scene, device
    def add_error_cloud(self, cloud, radius=0.006, color=None, solid=0., name=None):
        scene = self.scene
        cloud = fresnel.geometry.Sphere(scene, position = cloud, radius=radius)
        cloud_flat_color = gold_color
        if color is not None and len(np.array(color).shape)==1:
            cloud_flat_color = color
        cloud.material = fresnel.material.Material(solid=solid, \
                                                    color=fresnel.color.linear(cloud_flat_color),\
                                                    roughness=1.0,
                                                    specular=0.0)
        if color is not None and len(np.array(color).shape)>1:
            cloud.material.primitive_color_mix = 1.0
            cloud.color[:] = fresnel.color.linear(plt.cm.plasma(color)[:,:3])
        return cloud
    def add_cloud(self, cloud, radius=0.006, color=None, solid=0., primitive_color_mix=1., cloud_flat_color = gold_color, 
            roughness=.2, specular=.8, spec_trans=0., metal=0., name=None, **kwargs):
        scene = self.scene
        cloud = fresnel.geometry.Sphere(scene, position = cloud, radius=radius)
        
        if color is not None and len(np.array(color).shape)==1:
            cloud_flat_color = color
        cloud.material = fresnel.material.Material(solid=solid, \
                                                    color=fresnel.color.linear(cloud_flat_color),\
                                                    roughness=roughness,
                                                    specular=specular,
                                                    metal=metal,
                                                    spec_trans=spec_trans)
        if color is not None and len(np.array(color).shape)>1:
            cloud.material.primitive_color_mix = primitive_color_mix
            cloud.color[:] = fresnel.color.linear(color)
        return cloud
    def add_lines(self, lines, radius=0.006, color=None, solid=0., primitive_color_mix=1., flat_color = gold_color, 
            roughness=.2, specular=.8, spec_trans=0., metal=0., name=None):
        scene = self.scene
        lines = fresnel.geometry.Cylinder(scene, points=lines, radius=radius)
        lines.material = fresnel.material.Material(solid=solid, \
                                                    color=fresnel.color.linear(flat_color),\
                                                    roughness=roughness,
                                                    specular=specular,
                                                    metal=metal,
                                                    spec_trans=spec_trans)
        if color is not None and len(np.array(color).shape)>1:
            lines.material.primitive_color_mix = primitive_color_mix
            lines.color[:] = color
        return lines
    def add_mesh(self, vert, face, fvert=None, outline_width=None, name=None,
            color = gray_color, vert_color=None, vert_color_scheme=None, solid=0., roughness=.2, specular=.8, spec_trans=0., metal=0.) :
        """ vert_color: (Vn, 4) 
            if triangle faces are given (fvert), directly use it
        """
        scene = self.scene
        fvert = vert[face].reshape(-1,3) if fvert is None else fvert
        mesh = fresnel.geometry.Mesh(scene,vertices=fvert ,N=1)
        mesh.material = fresnel.material.Material(color=fresnel.color.linear(color),
                                                    solid=solid,
                                                    roughness=roughness,
                                                    specular=specular,
                                                    spec_trans=.5,
                                                    metal=metal)
        if vert_color is not None:
            mesh.color[:] = fresnel.color.linear(vert_color)
            mesh.material.primitive_color_mix = 1.0
        elif vert_color_scheme is not None:
            if vert_color_scheme == "normal":
                normals = igl.per_vertex_normals(vert, face)
                vert_color = (normals[face].reshape(-1,3)+1)/2.
            mesh.color[:] = fresnel.color.linear(vert_color)
            mesh.material.primitive_color_mix = 1.0
        if outline_width is not None:
            mesh.outline_width = outline_width
        return mesh

    def add_light(self, direction=(0,1,0), color=(1,1,1), theta=3.14):
        self.scene.lights.append( 
            fresnel.light.Light(    direction=direction, 
                                    color=color, 
                                    theta=theta) 
            )
    def add_bbox(self, *args, **kwargs):
        addBBox(self.scene, *args, **kwargs)
        return self
    def add_box(self, *args, **kwargs):
        addBox(self.scene, *args, **kwargs)
        return self
    def add_plane(self, *args, **kwargs):
        addPlane(self.scene, *args, **kwargs)
        return self
    def compute_mask(self, min_y=None):
        scene = self.scene
        if min_y is None:
            min_y = scene.get_extents()[0,1]
        #self.add_box(center=np.array([0,min_y-0.04,0]), spec=(100, 0.01, 100), color=black_color*0, solid=1.)
        #temp_lights = [light for light in scene.lights]
        #scene.lights.append( fresnel.light.Light(direction= np.array([0,1,0]), color=np.array([1,1,1])*10, theta=3.14) )

        preview_tracer = fresnel.tracer.Preview(device=self.device, w=self.camera_kwargs["resolution"][0], h=self.camera_kwargs["resolution"][1])
        preview_img = np.array(preview_tracer.render(scene)[:])
        mask = preview_img[..., 3] / 255
        #del scene.geometry[-1] #.material.color = white_color
        #scene.lights = temp_lights
        #mask = (preview_img[...,:3].sum(axis=-1) != preview_img.min())
        #mask = rgb2gray( rgba2rgb(preview_img) )
        return mask

    def render(self, preview=False, shadow_catcher=False, invisible_catcher=False, min_y=None, shadow_percentile=80, shadow_strength=1., 
                    lights=None ):
        scene = self.scene
        resolution = self.camera_opt["resolution"]
        samples = self.camera_opt["samples"]
        light_samples = self.camera_opt["light_samples"]
        #scene.lights[0].direction = np.array([.2,1,0.2])
        if lights is not None:
            scene.lights = lights
        tracer = fresnel.tracer.Path(device=self.device, w=resolution[0], h=resolution[1])

        if preview == True:
            preview_tracer = fresnel.tracer.Preview(device=self.device, w=self.camera_kwargs["resolution"][0], h=self.camera_kwargs["resolution"][1], anti_alias=True)
            image      = np.array(preview_tracer.render(scene)[:])
        else:
            
            if shadow_catcher == True:
                mask = self.compute_mask(min_y)
                self.add_plane(center=np.array([0,min_y-0.04,0]), spec=(400, 400), color=white_color*1., solid=0.0)

                #from xgutils.vis import visutil
                #geos = scene.geometry
                #scene.geometry = [scene.geometry[-1]]
                #preview_tracer = fresnel.tracer.Preview(device=self.device, w=self.camera_kwargs["resolution"][0], h=self.camera_kwargs["resolution"][1])
                #plane_img = np.array(preview_tracer.render(scene)[:])
                #visutil.showImg(plane_img)
                #scene.geometry = geos

            tracer.sample(scene, samples=samples, light_samples=light_samples)
            image = tracer.render(scene)[:]

            if shadow_catcher==True:
                if invisible_catcher == True:
                    del scene.geometry[-1] #.material.color = white_color
                    self.add_box(center=np.array([0,min_y-0.04,0]), spec=(100, 0.01, 100), color=black_color*0, solid=1.)
                    true_img = tracer.render(scene)[:]
                    image[mask] = true_img[mask]
                grayscale = rgb2gray( rgba2rgb(image) )
                shadow_map = (1-grayscale)*255 # 255: opaque
                all_mask = image[..., 3]/255
                catcher_mask = np.maximum( mask, all_mask ) - np.minimum( mask, all_mask )
                shadow_map = shadow_map/255 * catcher_mask
                thresh = np.percentile(shadow_map.reshape(-1), shadow_percentile)
                shadow_map[shadow_map <  thresh] = 0.
                shadow_map[shadow_map >= thresh] = ((shadow_map[shadow_map >= thresh]-thresh) * 1 / (1-thresh) )**shadow_strength
                image[..., 3] = image[..., 3]*(1-catcher_mask) + shadow_map*255*catcher_mask
        image = image / 255. # convert to [0,1]
        return image

# legacy code


def render_polyloop_legacy(vert, face, modes=["vert","edge","face"], centroid=None, preview=True, camera_kwargs=dflt_camera):
    lines = []
    pind = []
    for fi in range(len(face)):
        st_ind  = face[fi]
        end_ind = np.roll(face[fi], -1)
        start_point = vert[st_ind]
        end_point = vert[end_ind]
        ls = np.stack([start_point, end_point], axis=1)
        lines.append(ls)
        pind.append( np.zeros(len(face[fi]), dtype=np.int32) + fi )
    lines = np.concatenate(lines, axis=0)
    pind = np.concatenate(pind, axis=0)

    pcolors = unique_colors[pind]
    renderer = FresnelRenderer(camera_kwargs)
    if "vert" in modes:
        renderer.add_cloud(vert, radius=.009, color=pcolors*1.1)

    if centroid is not None:
        color = unique_colors[ np.arange(len(centroid), dtype=int) ]
        renderer.add_cloud(centroid, radius=.02, color=color)
    
    pcolors = pcolors[:,None,:].repeat(2, axis=1)
    if "edge" in modes:
        renderer.add_lines(lines, radius=.006, color=pcolors*.9, roughness=.1, specular=1., metal=.1)

    img_wire = renderer.render(preview=preview)

    tri_face, poly_ind = geoutil.triangulate_poly_face(face, return_index=True)
    try:
        from xgutils import bpyutil
        lface=face
        # for f in face:
        #     f = np.array(f)
        #     lface.append( np.r_[f, f[0]] )
        tri_vert, tri_face, poly_ind = bpyutil.vf2trimesh(vert, lface, return_index=True)
    except:
        pass

    fvert = vert[tri_face].reshape(-1,3)
    c_ind = np.repeat(poly_ind, 3)
    vert_color = unique_colors[ c_ind ]
    renderer.add_mesh(vert=None, face=None, fvert=fvert, vert_color=vert_color)
    if "face" in modes:
        img_face = renderer.render(preview=preview)
    return img_wire, img_face

def mat_test():
    import fresnel
    import ipywidgets
    import math
    device = fresnel.Device()
    scene = fresnel.Scene(device)
    position = []
    for k in range(5):
        for i in range(5):
            for j in range(5):
                position.append([2*i, 2*j, 2*k])
    geometry = fresnel.geometry.Sphere(scene, position = position, radius=1.0)
    scene.camera = fresnel.camera.Orthographic.fit(scene)

    tracer = fresnel.tracer.Path(device=device, w=1000, h=75)

    tracer.resize(450,450)

    @ipywidgets.interact(color=ipywidgets.ColorPicker(value='#1c1c7f'),
                        primitive_color_mix=ipywidgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.1, continuous_update=False),
                        roughness=ipywidgets.FloatSlider(value=0.3, min=0.1, max=1.0, step=0.1, continuous_update=False),
                        specular=ipywidgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1, continuous_update=False),
                        spec_trans=ipywidgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.1, continuous_update=False),
                        metal=ipywidgets.FloatSlider(value=0, min=0.0, max=1.0, step=1.0, continuous_update=False),
                        light_theta=ipywidgets.FloatSlider(value=5.5, min=0.0, max=2*math.pi, step=0.1, continuous_update=False),
                        light_phi=ipywidgets.FloatSlider(value=0.8, min=0.0, max=math.pi, step=0.1, continuous_update=False))
    def test(color, primitive_color_mix, roughness, specular, spec_trans, metal, light_theta, light_phi):
        r = int(color[1:3], 16)/255;
        g = int(color[3:5], 16)/255;
        b = int(color[5:7], 16)/255;
        scene.lights[0].direction = (math.sin(light_phi)*math.cos(-light_theta),
                                    math.cos(light_phi),
                                    math.sin(light_phi)*math.sin(-light_theta))

        scene.lights[1].theta = math.pi
        geometry.material = fresnel.material.Material(color=fresnel.color.linear([r,g,b]),
                                                    primitive_color_mix=primitive_color_mix,
                                                    roughness=roughness,
                                                    metal=metal,
                                                    specular=specular,
                                                    spec_trans=spec_trans
                                                    )
        return tracer.sample(scene, samples=64, light_samples=1)
