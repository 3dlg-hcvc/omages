import re
import os
import sys
import mcubes
import numpy as np
#import igl # some function requires igl
from scipy.spatial import ConvexHull
from skimage.measure import find_contours
from skimage import morphology
from xgutils import nputil, sysutil
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix

# creating geometries
# def create_box(lb=[-.5,]*3, ub=[.5]*3)


def rotation_v1tov2(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = (v1+v2) / np.linalg.norm(v1+v2)
    rot = Rotation.from_rotvec(v3*180, degrees=True)
    return rot


def length(x):
    return np.linalg.norm(x)


def point2lineDistance(q, p1, p2):
    d = np.linalg.norm(np.cross(p2-p1, p1-q))/np.linalg.norm(p2-p1)
    return d

def pointSegDistance(q, p1, p2):
    """ distance from point q to line segment p1p2, dimension agnostic
    Args:
        q (np.ndarray): query point of shape (d)
        p1 (np.ndarray): point 1 of the line segment of shape (d)
        p2 (np.ndarray): point 2 of the line segment of shape (d)
    Returns:
        float: distance
        np.ndarray: nearest point on line segment
    """
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
# def sphere_picking()
def point_to_line_segment_distance_torch(q, p):
    """ batch version of pointSegDistance
    Args:
        q (np.ndarray): query point of shape (N, d)
        p (np.ndarray): line segments of shape (M, 2, d)
    Returns:
        np.ndarray: distance
        np.ndarray: index of the nearest line segment
    """
    # Ensure input is in torch tensor format
    # q = torch.tensor(q, dtype=torch.float32)
    # p = torch.tensor(p, dtype=torch.float32)

    # Calculate vectors
    p1, p2 = p[:, 0, :], p[:, 1, :] # (M, d)
    p1_p2 = p2 - p1 # (M, d)
    p1_q = q[:, None, :] - p1[None, :, :] # (N, M, d)

    # Calculate squared lengths of p1_p2 and p1_q, and dot product of p1_p2 and p1_q
    p1_p2_squared_length = torch.sum(p1_p2 ** 2, dim=1) # (M,)
    p1_q_dot_p1_p2 = torch.sum(p1_q * p1_p2, dim=2) # (N, M)

    # Project p1_q onto p1_p2, clamping between 0 and 1
    denominator = p1_p2_squared_length[None, :] # (1, M)
    denominator = denominator.repeat(len(q), 1) # (N, M)
    t = torch.where(denominator != 0, p1_q_dot_p1_p2 / denominator, torch.zeros_like(denominator)) # (N, M)
    t = torch.clamp(t, 0, 1) # (N, M)

    # Find the nearest point on the line segment
    nearest = p1[None, :, :] + t[:, :, None] * p1_p2[None, :, :]  # (N, M, d)

    # Calculate the distance from q to nearest
    distance = torch.sqrt(torch.sum((q[:, None, :] - nearest) ** 2, dim=2))

    # Return minimum distances and corresponding nearest line segment index
    min_distance_idx = torch.argmin(distance, dim=1)
    bid = torch.arange(len(q))
    return distance[bid, min_distance_idx], nearest[bid, min_distance_idx], min_distance_idx

def hausdorff(va, fa, vb, fb):
    import igl
    dist1, _, _ = igl.point_mesh_squared_distance(va, vb, fb)
    dist2, _, _ = igl.point_mesh_squared_distance(vb, va, fa)
    return np.sqrt((dist1)), np.sqrt((dist2))

def get2DRotMat(theta=90, mode='degree'):
    if mode == 'degree':
        theta = np.radians(theta)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])



def sample_line_sequence(points, features=None, spacing=1.):
    """
    Resample points with a fixed spacing.

    Args:
        points (np.ndarray): N x C array of points.
        features (np.ndarray, optional): N x D array of features corresponding to points.
        spacing (float, optional): The spacing distance between points.

    Returns:
        np.ndarray: Resampled points.
        np.ndarray (optional): Resampled features if features are provided.
    """
    if points.size == 0:
        raise ValueError("Input 'points' should not be empty.")
    cdim = points.shape[1]
    points = np.array(points)
    N = len(points)
    lines = points[1:]-points[:-1]
    dists = np.linalg.norm(lines, axis=1)
    dists = np.r_[0, dists]
    cumdists = np.cumsum(dists)
    sample_dists = spacing * np.arange(int(cumdists[-1] / spacing)+1)

    nearest_ind = np.searchsorted(cumdists, sample_dists, side="left")
    nearest_ind = nearest_ind[1:]  # remove the starting point
    delta = (sample_dists[1:] - cumdists[nearest_ind-1])
    portion = delta / dists[nearest_ind]
    sample_pos = points[nearest_ind-1] + \
        portion[:, None] * lines[nearest_ind-1]
    sample_pos = np.r_[points[0][None, :], sample_pos]
    if features is not None:
        sample_features = (1-portion[:, None]) * features[nearest_ind-1] + \
            portion[:, None] * features[nearest_ind]
        sample_features = np.r_[features[0][None, :], sample_features]
        return sample_pos, sample_features
    return sample_pos
def sample_line_sequence_N(points, sample_N=10, features=None):
    """
    points: N x C
    The main difference between this function and sample_line_sequence is that this function samples N points uniformly along the line *with start&end points* , while the other function samples points with a fixed spacing.
    """
    points = np.array(points)
    cdim = points.shape[1]
    N = len(points)
    lines = points[1:]-points[:-1] # (N-1, C)
    dists = np.linalg.norm(lines, axis=1) # (N-1, C)
    cumdists = np.cumsum(dists)
    sample_dists = np.linspace(0, cumdists[-1], sample_N)

    nearest_ind = np.searchsorted(cumdists, sample_dists, side="left")
    delta = (sample_dists - np.r_[0,cumdists][nearest_ind])
    portion = np.divide(delta, dists[nearest_ind], out=np.zeros_like(delta), where=dists[nearest_ind]!=0)

    sample_pos = points[nearest_ind] + \
        portion[:, None] * lines[nearest_ind]
    if features is not None:
        sample_features = (1-portion[:, None]) * features[nearest_ind-1] + \
            portion[:, None] * features[nearest_ind]
        sample_features = np.r_[features[0][None, :], sample_features]
        return sample_pos, sample_features
    return sample_pos

def resample_curve(points, num_samples, features=None):
    """
    Resample a sequence of N-dimensional points to a specified number of samples,
    maintaining the first and last points.

    Args:
        points (np.ndarray): An array of shape (N, M) with N points in M dimensions.
        num_samples (int): The number of samples to resample to.
        features (np.ndarray, optional): An array of shape (N, D) corresponding to features for the points.

    Returns:
        np.ndarray: An array of shape (num_samples, M) with the resampled points.
        np.ndarray: An array of shape (num_samples, D) with the resampled features if features are provided.
    """
    points = np.array(points)
    if points.size == 0 or num_samples <= 1:
        raise ValueError("The points array must not be empty and num_samples must be greater than 1.")

    # Calculate cumulative distances between each consecutive point
    distances = np.cumsum([0] + list(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    new_distances = np.linspace(0, distances[-1], num_samples)

    # Find the indices in the original curve that bracket each sample point
    idx = np.searchsorted(distances, new_distances, side='right') - 1
    idx[idx == len(points) - 1] = len(points) - 2  # Ensure no index is out of bounds
    weights = ((new_distances - distances[idx]) / (distances[idx + 1] - distances[idx])).reshape(-1, 1)

    # Perform linear interpolation for resampled points
    resampled_points = (1 - weights) * points[idx] + weights * points[idx + 1]

    if features is not None:
        if len(features) != len(points):
            raise ValueError("The number of features must match the number of points.")
        # Perform linear interpolation for resampled features
        resampled_features = (1 - weights) * features[idx] + weights * features[idx + 1]
        return resampled_points, resampled_features

    return resampled_points
    # Example usage:
    # num_points = 100
    # num_dimensions = 3
    # num_samples = 10
    # points = np.random.rand(num_points, num_dimensions)
    # resampled = resample_curve(points, num_samples)

def sample_inside_unit_circle(N=1000):
    """
    Sample N points inside a unit circle
    """
    R = 1
    r = R * np.sqrt(np.random.rand(N))
    theta = np.random.rand(N) * 2 * np.pi
    pts = np.c_[r * np.cos(theta), r * np.sin(theta)]
    return pts
def sample_on_unit_circle(N=1000):
    """
    Sample N points on a unit circle
    """
    theta = np.linspace(0, 2*np.pi, N)
    pts = np.c_[np.cos(theta), np.sin(theta)]
    return pts


def sample_sphere(point_N, dim=3):
    """
        uniformly sample points on a sphere
        https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
    """
    vec = np.random.randn(point_N, dim)
    vec /= np.linalg.norm(vec, axis=1)[..., None]
    return vec

'''
Function used to Perform Spherical Flip on the Original Point Cloud
'''


def sphericalFlip(points, center, param):
    n = len(points)  # total n points
    points = points - np.repeat(center, n, axis=0)  # Move C to the origin
    # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
    normPoints = np.linalg.norm(points, axis=1)
    R = np.repeat(max(normPoints) * np.power(10.0, param),
                  n, axis=0)  # Radius of Sphere
    flippedPointsTemp = 2 * \
        np.multiply(np.repeat((R - normPoints).reshape(n, 1),
                    len(points[0]), axis=1), points)
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(
        n, 1), len(points[0]), axis=1))  # Apply Equation to get Flipped Points
    flippedPoints += points
    return flippedPoints


def hidden_point_removal(cloud, campos):
    # View Point, which is well above the point cloud in z direction
    C = np.array([campos])
    # Reflect the point cloud about a sphere centered at C
    flippedCloud = sphericalFlip(cloud, C, np.pi)
    # All points plus origin
    points = np.append(flippedCloud, [[0, 0, 0]], axis=0)
    # Visibal points plus possible origin. Use its vertices property.
    hull = ConvexHull(points)
    visible = hull.vertices[:-1]  # remove origin
    return cloud[visible]


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
        rd_a = np.r_[0, 1, 0, rd_a]
        rd_b = np.r_[0, 0, 1, rd_b]
    samples = np.array([vertices[0] + rd_a[i]*(vertices[1]-vertices[0]) + rd_b[i]*(vertices[2]-vertices[0])
                        for i in range(sampleNum)])
    return samples


def randQuat(N=1):
    # Generates uniform random quaternions
    # James J. Kuffner 2004
    # A random array 3xN
    s = np.random.rand(3, N)
    sigma1 = np.sqrt(1.0 - s[0])
    sigma2 = np.sqrt(s[0])
    theta1 = 2*np.pi*s[1]
    theta2 = 2*np.pi*s[2]
    w = np.cos(theta2)*sigma2
    x = np.sin(theta1)*sigma1
    y = np.cos(theta1)*sigma1
    z = np.sin(theta2)*sigma2
    return np.array([w, x, y, z])


def multQuat(Q1, Q2):
    # https://stackoverflow.com/a/38982314/5079705
    w0, x0, y0, z0 = Q1   # unpack
    w1, x1, y1, z1 = Q2
    return([-x1*x0 - y1*y0 - z1*z0 + w1*w0, x1*w0 + y1*z0 - z1*y0 +
            w1*x0, -x1*z0 + y1*w0 + z1*x0 + w1*y0, x1*y0 - y1*x0 + z1*w0 +
            w1*z0])


def conjugateQuat(Q):
    return np.array([Q[0], -Q[1], -Q[2], -Q[3]])


def applyQuat(V, Q):
    P = np.array([0., V[0], V[1], V[2]])
    nP = multQuat(Q, multQuat(P, conjugateQuat(Q)))
    return nP[1:4]


def fibonacci_sphere(samples=1000):
    rnd = 1.

    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - np.power(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        points.append([x, y, z])

    return points

###### Mesh operations ######
def prune_unused_vertices(vert, face):
    # Identify all unique vertex indices used in face
    unique_vert_ind = np.unique(face)
    mapper = np.zeros(vert.shape[0], dtype=int)
    mapper[unique_vert_ind] = np.arange(unique_vert_ind.shape[0])
    new_face = mapper[face]
    # Create the new vertices array using only the used vertices
    new_vert = vert[unique_vert_ind]
    
    return new_vert, new_face, unique_vert_ind
    # # Example usage:
    # verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 2, 2]])  # 5 vertices
    # faces = np.array([[0, 2, 3], [0, 3, 4]])  # 2 faces, vertex 4 is unused
    # new_verts, new_faces = prune_unused_vertices(verts, faces)
    # print("New Vertices:\n", new_verts)
    # print("New Faces:\n", new_faces)

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

def mergeMeshes(meshes):
    """ merge a list of meshes into one mesh
    Args:
        meshes (list): list of meshes, each mesh is a dict with keys "vert" (V, Vk), "face" (F,Fk) and other vertex attributes
        This function will automatically handle arbitrary number of vertex attributes of shape (V, D)
    Returns:
        dict: merged mesh
    """
    nm = {key: [] for key in meshes[0].keys()}
    verts = []
    faces = []
    total_vert_num = 0
    for mesh in meshes:
        for key in mesh:
            nm[key].append(mesh[key])
        nm["face"][-1] += total_vert_num
        if len(nm["face"][-1].shape) == 1: # if there is only one face, reshape it to (1,Fk)
            nm["face"][-1] = nm["face"][-1].reshape(1, -1)
        total_vert_num += len(nm["vert"][-1])
        assert len(
            nm["vert"][-1].shape) == 2 and len(nm["face"][-1].shape) == 2, "Invalid mesh!"
    for key in nm:
        nm[key] = np.concatenate(nm[key])
    return nm


def filterMesh(vert, face, filterV):
    v_keep_mask = filterV
    v_del_mask = 1-filterV

    newIndV = np.zeros(vert.shape[0], dtype=int)-1
    newIndV[v_keep_mask] = np.arange(v_keep_mask.sum()).astype(int)
    nv = vert[v_keep_mask]
    f_keep_mask = (v_keep_mask[face].sum(axis=-1) == 3)
    nf = face[f_keep_mask]
    nf = newIndV[nf]
    assert (nf >= 0).all(), "New face list contains removed vertices"
    return nv, nf
    # TODO filterF


def normalizePointSet(vert, no_scale=False):
    # center = vert.mean(axis=0)
    bbmax = vert.max(axis=0)
    bbmin = vert.min(axis=0)
    bbcenter = (bbmax + bbmin) / 2.
    bbscale = (bbmax - bbmin).max() / 2.
    vert = vert - bbcenter
    if no_scale == False:
        vert = vert/bbscale
    return vert

# A more general point set normalization

def normalize_pts(pts, uniform_scaling=True):
    """ normalize point set to [-1, 1]^3
    Args:
        pts (np.ndarray): point set of shape (*, num_points, 3)
        uniform_scale (bool, optional): whether to use uniform scaling. Defaults to True.
        scale (float, optional): scale factor. Defaults to .9.
    Returns:
        np.ndarray: normalized point set
        bbox_center (np.ndarray): bbox center (*, 1, 3)
        bbox_scale (np.ndarray): bbox scale (*, 1, 3)
    """

    bbox_center = (pts.max(axis=-2, keepdims=True) + pts.min(axis=-2, keepdims=True))/2
    pts = pts - bbox_center
    bbox_scale = pts.max(axis=-2, keepdims=True) 
    if uniform_scaling:
        bbox_scale = bbox_scale.max(axis=-1, keepdims=True)
        bbox_scale = bbox_scale.repeat(3, axis=-1)
    pts = np.divide(pts, bbox_scale, out=np.zeros_like(pts), where=bbox_scale!=0) 
    return pts, bbox_center, bbox_scale
def denormalize_pts(pts, bbox_center, bbox_scale):
    """ denormalize point set to [-1, 1]^3
    Args:
        pts (np.ndarray): point set of shape (*, num_points, 3)
        bbox_center (np.ndarray): bbox center (*, 1, 3)
        bbox_scale (np.ndarray): bbox scale (*, 1, 3)
    Returns:
        np.ndarray: denormalized point set
    """
    pts = pts * bbox_scale
    pts = pts + bbox_center
    return pts

# reconstruction & sampling


def array2mesh(array, thresh=0., dim=3, coords=None, bbox=np.array([[-1, -1, -1], [1, 1, 1]]), return_coords=False,
               if_decimate=False, decimate_face=4096, cart_coord=True, gaussian_sigma=None):
    """from 1-D array to 3D mesh

    Args:
        array (np.ndarray): 1-D array
        thresh (float, optional): threshold. Defaults to 0..
        dim (int, optional): 2 or 3, curve or mesh. Defaults to 3.
        coords (np.ndarray, optional): array's coordinates (num_points, x_dim). Defaults to None.
        bbox (np.ndarray, optional): bounding box of coords. Defaults to np.array([[-1,-1],[1,1]]).
        return_coords (bool, optional): whether return the coords. Defaults to False.
        decimate_face (int, optional): whether to simplify the mesh. Defaults to 4096.
        cart_coord (bool, optional): cartesian coordinate in array form, x->i, y->j,... and all varibles increases monotonically. Defaults to True.
        gaussian_sigma (float, optional): sigma value for gaussian filter (set None if there is no filter)
    Returns:
        tuple: `verts`, `faces`, `coords` or `verts`, `faces` according to `return_coords`
    """
    grid = nputil.array2NDCube(array, N=dim)

    if gaussian_sigma is not None:
        # from scipy.ndimage import gaussian_filter
        # grid = gaussian_filter(grid.astype(float), sigma=gaussian_sigma)
        grid = mcubes.smooth(grid)
    if dim == 3:
        verts, faces = mcubes.marching_cubes(grid, thresh)
        if cart_coord == False:
            verts = verts[:, [1, 0, 2]]
        verts = verts/(grid.shape[0]-1)  # rearrange order and rescale
    elif dim == 2:
        contours = find_contours(grid, thresh)
        vcount, points, edges = 0, [], []
        for contour in contours:
            ind = np.arange(len(contour))
            points.append(contour)
            edges.append(np.c_[vcount+ind, vcount+(ind+1) % len(contour)])
            vcount += len(contour)
        if len(contours) == 0:
            return None, None
        verts = np.concatenate(points, axis=0)[:, [1, 0]] / (grid.shape[0]-1)
        # verts = verts[:,[1,0]]
        faces = np.concatenate(edges,  axis=0)
        # levelset_samples = igl.sample_edges(points, edges, 10)
    if coords is not None:
        bbmin, bbmax = nputil.arrayBBox(coords)
    else:
        bbmin, bbmax = bbox
        coords = nputil.makeGrid(bb_min=bbmin, bb_max=bbmax, shape=grid.shape)
    verts = verts*(bbmax-bbmin) + bbmin
    verts, faces = verts, faces.astype(int)
    if if_decimate == True:
        if dim != 3:
            print("Warning! decimation only works for 3D")
        elif faces.shape[0] > decimate_face:  # Only decimate when appropriate
            import igl
            reached, verts, faces, _, _ = igl.decimate(
                verts, faces, decimate_face)
            faces = faces.astype(int)
    if return_coords == True:
        return verts, faces, coords
    else:
        return verts, faces


def array2curve(array, thresh=0., coords=None):
    pass


def sampleMesh(vert, face, sampleN):
    import igl
    sampled = None
    if vert.shape[-1] == 3:
        resample = True
        while resample:
            try:
                result = igl.random_points_on_mesh(sampleN, vert, face)
                if len(result) == 2:
                    sampled = B[:, 0:1]*vert[face[FI, 0]] + \
                        B[:, 1:2]*vert[face[FI, 1]] + \
                        B[:, 2:3]*vert[face[FI, 2]]
                elif len(result) == 3: # 2.5.0 has 3 return values
                    B, FI, sampled = result
                else:
                    raise ValueError("Unknown result from igl.random_points_on_mesh, possibly from a version > 2.5.0")
                resample = False
                if sampled.shape[0] != sampleN:
                    print(
                        'Failed to sample "sampleN" points, now resampling...', file=sys.__stdout__)
                    resample = True
            except Exception as exc:
                print('Error encountered during mesh sampling:',
                      exc, file=sys.__stdout__)
                import traceback, pdb
                # Print exception details
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("Exception occurred:", exc_type)  # Print the exception type
                traceback.print_tb(exc_traceback)  # Print the stack trace
                print(getattr(exc, 'message', repr(exc)))  # Print the exception message if available

                print('Now resampling...', file=sys.__stdout__)
                resample = True
    elif vert.shape[-1] == 2:
        edge = face
        fac = 2 * np.ceil(sampleN / vert.shape[0]).astype(int)
        sampled = igl.sample_edges(vert, edge, fac)
        choice = np.random.choice(sampled.shape[0], sampleN, replace=False)
        sampled = sampled[choice]

    return sampled


sampleShape = sampleMesh

# geometry
def query_triangle_mesh_2d(queries, vert, face):
    """ find the face index of the query point in a triangle mesh
    Args:
        queries (np.ndarray): N x 2
        vert (np.ndarray): V x 2
        face (np.ndarray): F x 3
    Returns:
        np.ndarray: face index the query point is in, -1 if not in any face
        np.ndarray: barycentric coordinates of the query point in the face
    """
    pass

def point2lineDistance(q, p1, p2, ):
    pass
def signed_distance(queries, vert, face):  # remove NAN's
    import igl
    S, I, C = igl.signed_distance(queries, vert, face)
    if len(S.shape) == 0:
        S = S.reshape(1)
    return np.nan_to_num(S), I, C

def signed_distance_2d(queries, vert, face): 
    """ 2D version of signed_distance
    Args:
        queries (np.ndarray): N x 2
        vert (np.ndarray): V x 2
        face (np.ndarray): F x 3
    Returns:
        np.ndarray: signed distance
        np.ndarray: face index the query point is in
        np.ndarray: closest point on boundary
    """
    bd_edges = igl.boundary_facets(face)
    # calculate query point to boundary edge distance



def shape2sdf(shapePath, shapeInd, gridDim=256, disturb=False):
    vert = H5Var(shapePath, 'vert')[shapeInd]
    face = H5Var(shapePath, 'face')[shapeInd]
    x = y = z = np.linspace(0, 1, gridDim)
    grid = np.stack(np.meshgrid(x, y, z, sparse=False), axis=-1)
    all_samples = grid.reshape(-1, 3)
    if disturb == True:
        disturbation = np.random.rand(all_samples.shape[0], 3)/gridDim
        all_samples += disturbation
    S, I, C = signed_distance(all_samples, vert, face)
    sdfPairs = np.concatenate([all_samples, S[:, None]], axis=-1)
    return sdfPairs


def mesh2sdf(vert, face, gridDim=64, disturb=False):
    # x = y = z = np.linspace(0,1,gridDim)
    # grid = np.stack(np.meshgrid(x,y,z,sparse=False), axis=-1)
    all_samples = nputil.makeGrid(
        [-1, -1, -1.], [1., 1, 1], [gridDim, ]*3, indexing="ij")
    if disturb == True:
        disturbation = np.random.rand(all_samples.shape[0], 3)/gridDim
        all_samples += disturbation
    S, I, C = signed_distance(all_samples, vert, face)
    sdfPairs = np.concatenate([all_samples, S[:, None]], axis=-1)
    return sdfPairs


def pc2sdf():
    # TODO
    pass


# Triangle mesh related
def boundary_loops(vert, face, undirected=True): ######## @UNTESTED!
    """ find boundary loops of a triangle mesh, similar to igl.boundary_loop (return the loop with largest #vertices)), 
    but this function returns all of the boundary loops.
    Args:
        vert (np.ndarray): V x 3
        face (np.ndarray): F x 3
    Returns:
        list: list of boundary loops, each loop is a list of vertex indices
    """
    # Create an adjacency matrix from the edges
    # num_points is the total number of unique points in the graph
    bd_edges = igl.boundary_facets(face)
    rows, cols = zip(*bd_edges)
    data = np.ones(len(bd_edges))
    adj_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_points, num_points))

    if undirected:
        # The graph is undirected, so we mirror the edges
        adj_matrix = adj_matrix + adj_matrix.T

    # Find connected components
    n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

    # Store loops
    loops = []

    for i in range(n_components):
        # Get vertices in the current loop
        loop_vertices = np.where(labels == i)[0]

        if len(loop_vertices) > 1:  # Check for non-trivial loop
            # Perform depth-first search to order vertices
            start_vertex = loop_vertices[0]
            ordered_vertices, _ = scipy.sparse.csgraph.depth_first_order(adj_matrix, start_vertex, directed=False)
            # Filter only the vertices that belong to the current loop
            ordered_loop = [v for v in ordered_vertices if v in loop_vertices]
            loops.append(ordered_loop)
    return loops

def harmonic_parametrization(patch_vert, patch_face):
    """ harmonic parametrization of a triangle patch (part of a mesh)
    Args:
        patch_vert (np.ndarray): V x 3
        patch_face (np.ndarray): F x 3
        subdiv (int, optional): subdivision level. Defaults to 0.
    Returns:
        np.ndarray: parametrization of shape (V, 2)
    """
    vert, face = patch_vert, patch_face
    bnd = igl.boundary_loop(face) # find boundary loop with the largest #vertices
    # Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(vert, bnd)
    # Harmonic parametrization for the internal vertices
    uv = igl.harmonic(vert, face, bnd, bnd_uv, 1) # in [-1, 1]
    uv = (uv + 1) / 2 # in [0, 1]
    return uv

def force_decimate(vert, face, max_face=800): # guarantee reached is true
    if face.shape[0] > max_face:  # Only decimate when appropriate
        import igl
        reached, vert, face, _, _ = igl.decimate(
            vert, face, max_face)
        vert = vert.astype(int)
        if reached == False:
            # force remove some faces
            # face = face[:max_face]
            # # remove unused vertices
            # vert, face, _ = prune_unused_vertices(vert, face)
            pass
        return vert, face, True # indicate if forced
    else:
        return vert, face, False

# open3d related


class Open3D_Toolbox():
    def __init__(self):
        import open3d as o3d
        self.o3d = o3d

    def poisson_recon(self, cloud, estimate_normals=True, depth=6, quantile=.3, knn=10):
        o3d = self.o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.normals = o3d.utility.Vector3dVector(
            np.zeros((1, 3)))  # invalidate existing normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        # with o3d.utility.VerbosityContextManager(
        #    o3d.utility.VerbosityLevel.Debug) as cm:
        poi_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth)
        vertices_to_remove = densities < (np.quantile(densities, quantile))
        poi_mesh.remove_vertices_by_mask(vertices_to_remove)
        return np.asarray(poi_mesh.vertices), np.asarray(poi_mesh.triangles)

    def ball_pivoting(self, cloud, radii=[0.01, 0.02, 0.04], knn=30):
        o3d = self.o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.normals = o3d.utility.Vector3dVector(
            np.zeros((1, 3)))  # invalidate existing normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))

        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        return np.asarray(rec_mesh.vertices), np.asarray(rec_mesh.triangles)


class Meshlab_Toolbox():
    def __init__(self):
        import pymeshlab
        self.mlab = pymeshlab

    def poisson_recon(self, cloud, estimate_normals=True, depth=6, fulldepth=4, knn=10):
        pymeshlab = self.mlab
        temp_dir = os.path.expanduser('~/.temp/meshlab/')
        sysutil.mkdirs(temp_dir)
        cloud_path = os.path.join(temp_dir, "cloud.pts")
        recon_path = os.path.join(temp_dir, "recon.ply")

        np.savetxt(cloud_path, cloud)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(cloud_path)
        ms.compute_normals_for_point_sets(k=20)
        ms.surface_reconstruction_screened_poisson(
            depth=depth, fulldepth=fulldepth)
        ms.save_current_mesh(recon_path)
        vert, face = igl.read_triangle_mesh(recon_path)
        return vert, face

# grid sampling (inefficient)


def shapes2sdfs(shapePath, sdfPath, indices=np.arange(10), gridDim=256, disturb=False):
    # shapeDict = readh5(shapePath)
    # verts, faces = shapeDict['vert'], shapeDict['face']
    if os.path.exists(sdfPath):
        if '.h5' == sdfPath[-3:]:
            os.remove(sdfPath)
        else:
            raise ValueError('sdfPath must ended with .h5')
    args = [indices]

    def func(index): return shape2sdf(
        shapePath, index, gridDim=gridDim, disturb=disturb)
    def batchFunc(batchOut): return [H5Var(
        sdfPath, 'SDF').append(np.array(batchOut))]
    ret = np.array(parallelMap(
        func, args, zippedIn=False, batchFunc=batchFunc))[0]
    # print(ret.shape)
    # writeh5(sdfPath, {'SDF':sdf})
    return ret

def points_dist(p1, p2, k=1, return_ind=False):
    from scipy.spatial import cKDTree
    # from chamfer_distance import ChamferDistance
    # chamfer_dist = ChamferDistance()
    '''distance from p1 to p2'''
    # print(p1.shape, p2.shape)
    # d1, d2  = chamfer_dist( ptutil.np2th(p1), ptutil.np2th(p2) )
    tree = cKDTree(p2)
    dist, ind = tree.query(p1, k=k)
    if return_ind == True:
        return dist, ind
    else:
        return dist

def chamfer_dist(p1, p2, nopool=False):
    d1 = points_dist(p1, p2) ** 2
    d2 = points_dist(p2, p1) ** 2
    if nopool==True:
        return d1, d2
    dist1 = d1.mean()
    dist2 = d2.mean()
    cd = dist1 + dist2
    return dict(dist=cd, )
def chamfer_dist_mesh(v1, f1, v2, f2):
    import igl
    # sample 10^6 points for each mesh
    p1 = sampleMesh(v1, f1, 10**6)
    p2 = sampleMesh(v2, f2, 10**6)
    sqrD1, _, _ = igl.point_mesh_squared_distance(p1, v2, f2)
    sqrD2, _, _ = igl.point_mesh_squared_distance(p2, v1, f1)
    return sqrD1.mean() + sqrD2.mean(), sqrD1, sqrD2

def points_sdf(targetx, sign, ref):
    dist = points_dist(targetx, ref)
    dist = np.sign(sign)*dist
    return dist
# voxelization
def chamfer_dist_cuda(p1, p2):
    from xgutils.external.chamfer_distance import ChamferDistance
    chamfer_dist = ChamferDistance()


def sinkhorn_EMD(A, B):
    from xgutils.external.sinkhorn import sinkhorn
    return sinkhorn(A, B)

def morph_voxelization(vert, face, sampleN=1000000, grid_dim=128, selem_size=6):
    """ Morphological voxelization. Given arbitrary triangle soup, return the watertight voxelization of it.
        First sample cloud from mesh, voxelize the cloud, dilate, floodfill, erose. Note that dilate+erose=closing
    """
    from xgutil import ptutil
    vmin, vmax = np.abs(vert).min(), np.abs(vert).max()
    if vmax > 1.:
        print(
            f"Warning: Mesh should be fallen into [-1,1]^3 bounding box! vmin:{vmin} vmax:{vmax}")
    samples = sampleMesh(vert, face, sampleN)
    voxel, coords = ptutil.ths2nps(ptutil.point2voxel(
        samples[None, ...], grid_dim=grid_dim, ret_coords=True))
    voxel, coords = voxel[0], coords[0]
    if selem_size == 0:
        water_tight_voxel = 1 - morphology.flood(voxel, (0, 0, 0))
    else:
        selem = morphology.ball(selem_size)
        dilated = morphology.binary_dilation(voxel, selem)
        mask = 1-morphology.flood(dilated, (0, 0, 0))
        erosed = morphology.binary_erosion(mask, selem)
        water_tight_voxel = erosed
    return water_tight_voxel, coords
# coordinate transforms


def shapenetv1_to_shapenetv2(voxel):
    return np.flip(np.transpose(voxel, (2, 1, 0)), 2).copy()


def shapenetv2_to_nnrecon(voxel):
    return np.flip(np.transpose(voxel, (1, 0, 2)), 2).copy()


def shapenetv2_to_cart(voxel):
    return np.flip(voxel, 2).copy()


def nnrecon_to_cart(voxel):
    return np.flip(np.transpose(voxel, (2, 1, 0)), 0).copy()


def cart_to_nnrecon(voxel):
    return np.flip(np.transpose(voxel, (1, 0, 2)), 1).copy()


def convonet_to_nnrecon(array, dim=3, flatten=True):
    grid = nputil.array2NDCube(array.reshape(-1), N=3)
    swaped = np.swapaxes(grid, 0, -1)
    if flatten == True:
        return swaped.reshape(-1)
    else:
        return swaped
# SDF functions


def boxSDF(queries, spec, center=None):
    ''' queries: NxD array
        spec:    D array
        center:  D array
    '''
    if center is None:
        center = np.zeros(spec.shape)
    b = spec[None, ...]
    c = center[None, ...]
    queries -= c
    q = np.abs(queries) - b
    sd = q.max(axis=-1)
    # sd = sd*(sd>0)
    sd = np.linalg.norm(q*(q > 0), axis=-1) + sd*(sd < 0)
    return sd


def batchBoxSDF(queries, spec, center=None):
    ''' queries: NxD array
        spec:    MxD array
        center:  MxD array
        return:
            MxN array
    '''
    if center is None:
        center = np.zeros(spec.shape)
    b = spec[:, None, :]
    c = center[:, None, :]
    queries = queries[None, ...] - c
    q = np.abs(queries) - b
    sd = q.max(axis=-1)
    # sd = sd*(sd>0)
    sd = np.linalg.norm(q*(q > 0), axis=-1) + sd*(sd < 0)
    return sd


def sphereSDF(queries, radius=0.5, center=np.array([0., 0., 0.])):
    deltaV = queries - center[None, :]
    delta = np.sqrt((deltaV * deltaV).sum(axis=-1))
    return delta - radius


def SDF_sampling(vert, face, sample_N=64**3, near_std=0.015, far_std=0.2):
    """
        SDF sampling from mesh as in IF-Net
    """
    if np.abs(vert).max() > 1.:
        print("Warning(During SDF sampling), data exceeds bbox 1.",
              shape_path, np.abs(vert).max())
    Xbd = sampleMesh(vert, face, sample_N)

    near_num = sample_N // 2
    far_num = sample_N - near_num

    near_pts = Xbd[:near_num].copy()
    far_pts = Xbd[near_num:].copy()

    near_pts += near_std * np.random.randn(*near_pts.shape)
    far_pts += far_std * np.random.randn(*far_pts.shape)

    Xtg = np.concatenate([near_pts, far_pts], axis=0)
    mask = np.logical_or(Xtg > .99, Xtg < -.99)
    Xtg[mask] = np.random.rand(mask.sum())*2 - 1
    Xtg = Xtg.clip(-.99, .99)
    assert Xtg.min() >= -1.00001 and Xtg.max() <= 1.00001
    Ytg, _, _ = signed_distance(Xtg, vert, face)

    Xtg = Xtg.astype(np.float16)
    Ytg = Ytg.astype(np.float16)
    Xbd = Xbd.astype(np.float16)
    return Xbd, Xtg, Ytg
# simple geometries


cube = {"vert": np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                          [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1.0]]),
        "face": np.array([[0, 1, 2], [0, 2, 3],  [0, 3, 4], [0, 4, 5],  [0, 5, 6], [0, 6, 1],
                          [1, 6, 7], [1, 7, 2],  [7, 4, 3], [7, 3, 2],  [4, 7, 6], [4, 6, 5]])
        }

# Obsolete


def extract_levelset(target_x=None, target_y=None, sampleN=256, **kwargs):
    print("Warning, extract_levelset is obsolete now! Use array2mesh & sampleMesh instead.")
    dim = target_x.shape[-1]
    if dim == 3:
        shape = LevelsetVisual(opt=None).visualize(
            target_y=target_y,
            target_x=target_x,
            name='levelset')['shape']['levelset']
        vert, face = shape['vert'], shape['face']
        levelset_samples = sampleMesh(vert, face, sampleN)
    elif dim == 2:
        # TODO
        vert, edge = array2mesh(target_y, thresh=0., dim=2, coords=target_x)
        fac = 2 * np.ceil(sampleN / vert.shape[0]).astype(int)
        levelset_samples = igl.sample_edges(vert, edge, fac)
        levelset_samples = np.random.choice(
            levelset_samples.shape[0], sampleN, replace=False)
    return levelset_samples


def write_mesh(data_dir, vert, face, input_name):
    mesh_dir = os.path.join(data_dir, "meshes/")
    sysutil.mkdirs(mesh_dir)
    if vert.shape[0] < 10:
        vert, face = np.array([[0, 0, 0.]]), np.array([[0, 0, 0]])
    igl.write_triangle_mesh(os.path.join(
        mesh_dir, input_name+".ply"), vert, face,  force_ascii=False)
def write_ply(out_path, verts, faces):
    sysutil.mkdirs(os.path.dirname(out_path))
    vertices, polygons = verts, faces
    with open(out_path, 'w') as fout:
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(len(vertices))+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("element face "+str(len(polygons))+"\n")
        fout.write("property list uchar int vertex_index\n")
        fout.write("end_header\n")

        for i in range(len(vertices)):
            fout.write(str(vertices[i][0])+" "+str(vertices[i][1])+" "+str(vertices[i][2])+"\n")

        for i in range(len(polygons)):
            fout.write(str(len(polygons[i])))
            for j in range(len(polygons[i])):
                fout.write(" "+str(polygons[i][j]))
            fout.write("\n")
def adjacency_matrix(vert, face, zero_diag=False):
    row = np.concatenate([face[:,0], face[:,1], face[:,2]])
    col = np.concatenate([face[:,1], face[:,2], face[:,0]])
    data = np.ones(len(row), dtype=int)
    n = len(vert)
    adj_matrix = coo_matrix((data, (row, col)), shape=(n, n), dtype=int)
    if zero_diag:
        adj_matrix.setdiag(0)
    else:
        adj_matrix.setdiag(1)
    return np.asarray(adj_matrix.todense()).astype(bool)  # Convert to a dense numpy array
def polymesh_adjacency_matrix(vert, face, zero_diag=False):
    row, col = [], []
    for f in face:
        row.append(np.array(f))
        col.append(np.roll(row[-1], -1))
    row, col = np.concatenate(row), np.concatenate(col)
    data = np.ones(len(row), dtype=int)
    n = len(vert)
    adj_matrix = coo_matrix((data, (row, col)), shape=(n, n), dtype=int)
    if zero_diag:
        adj_matrix.setdiag(0)
    else:
        adj_matrix.setdiag(1)
    return np.asarray(adj_matrix.todense()).astype(bool)  # Convert to a dense numpy array

import six
def write_poly_ply_file(outdir, vertices, polygons, loops=None, use_color=False):
    if loops is not None:
        llen = [len(l) for l in loops]
        llen = np.array(llen)
        lacc = np.cumsum(llen) - llen
        vertices = np.concatenate(loops, axis=0)
        polygons = [np.arange(llen[i],dtype=int)+lacc[i] for i in range(len(llen))]
        print(vertices.shape)
        #print(polygons)
    with open(outdir, 'w') as fout:
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(len(vertices))+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        if use_color==True:
            fout.write("property uchar red\n")
            fout.write("property uchar green\n")
            fout.write("property uchar blue\n")
        fout.write("element face "+str(len(polygons))+"\n")
        fout.write("property list uchar int vertex_index\n")
        fout.write("end_header\n")

        for i in range(len(vertices)):
            vertex_str = str(vertices[i][0])+" "+str(vertices[i][1])+" "+str(vertices[i][2])
            if use_color==True:
                r,g,b = vertices[i][3:].astype(int)
                color_str = " {} {} {}".format(r,g,b)
                vertex_str += color_str
            fout.write(vertex_str + "\n")
        for i in range(len(polygons)):
            fout.write(str(len(polygons[i])))
            for j in range(len(polygons[i])):
                fout.write(" "+str(polygons[i][j]))
            fout.write("\n")
# This function is modified from https://github.com/deepmind/deepmind-research/blob/master/polygen/data_utils.py
def read_obj_file(obj_file):
    """Read vertices and faces from already opened file."""
    verts = []
    faces = []
    reordered_verts = []
    reordered_faces = []

    for line in obj_file:
        tokens = line.split()
        if not tokens:
            continue
        line_type = tokens[0]
        # We skip lines not starting with v or f.
        if line_type == 'v':
            verts.append([float(x) for x in tokens[1:]])
        elif line_type == 'f':
            face = []
            for i in range(len(tokens) - 1):
                vertex_name = tokens[i + 1]
                index = vertex_name = six.ensure_str(vertex_name).split('/')[0]
                face.append(int(index) - 1)
            faces.append(face)
            
    verts = np.array(verts, dtype=np.float32)
    reordered_faces, reorder_mapping = reindex_poly_face(faces)
    reordered_verts = verts[reorder_mapping[:, 0]]
    return verts, faces, reordered_verts, reordered_faces, reorder_mapping
def reindex_poly_face(faces):
    # face: list of list of int
    vert_ind_dict = {}
    reind_faces = []
    for i in range(len(faces)):
        face = faces[i]
        reind_faces.append([])
        for j in range(len(face)):
            if face[j] not in vert_ind_dict:
                new_ind = len(vert_ind_dict)
                vert_ind_dict[face[j]] = new_ind
            else:
                new_ind = vert_ind_dict[face[j]]
            reind_faces[-1].append( new_ind )
    #list(vert_ind_dict.items())
    mapping = np.array( list(vert_ind_dict.items()) )
    return reind_faces, mapping
def triangulate_poly_face(faces, return_index=False):
    tri_faces = []
    inds = []
    for fi, face in enumerate(faces):
        for i in range(len(face)-2):
            tri_faces.append([face[0], face[i+1], face[i+2]])
            inds.append(fi)
    if return_index:
        return np.array(tri_faces), np.array(inds)
    else:
        return np.array(tri_faces)

def polymesh2soup(verts, faces):
    s_verts = []
    s_faces = []
    s_num_vert = 0
    for fi, face in enumerate(faces):
        s_faces.append( list(s_num_vert + np.arange(len(face)) ) )
        s_verts.append(verts[face])
        s_num_vert = s_num_vert + len(face)
    s_verts = np.concatenate(s_verts, axis=0)
    return s_verts, s_faces
def soup_grouping_mat(soup_faces):
    faces = soup_faces
    face_vert_count = np.zeros(len(faces), dtype=int)
    for i, face in enumerate(faces):
        face_vert_count[i] = len(face)
    mat = np.eye(len(faces))
    mat = mat.repeat(face_vert_count, axis=0)
    mat = mat.repeat(face_vert_count, axis=1)
    mat = mat.astype(bool)
    sv_faceind = np.repeat( np.arange(len(faces)), face_vert_count )
    return mat, sv_faceind
def read_obj_polymesh(obj_path):
  """Open .obj file from the path provided and read vertices and faces."""

  with open(obj_path) as obj_file:
    return read_obj_file(obj_file)

def polymesh2face_adjmat(vert, face):
    """ compute face adjacency matrix from vertex and face of a polymesh
    Two faces are adjacent if they share an edge
    """
    # row = []
    # col = []
    # edge_dict = {}
    # adj_f = np.zeros((len(face), len(face)))
    # for i, f in enumerate(face):
    #     for j in range(len(f)):
    #         e = (f[j], f[(j+1)%len(f)])
    #         #print(e)
    #         if e not in edge_dict:
    #             edge_dict[e] = i
    #             edge_dict[(e[1],e[0])] = i
    #         else:
    #             edge_owner = edge_dict[e]
    #             adj_f[ edge_owner, i ] = 1
    #             adj_f[ i, edge_owner ] = 1
    row = []
    col = []
    vert_dict = {}
    adj_f = np.zeros((len(face), len(face)))
    for i, f in enumerate(face):
        for j in range(len(f)):
            e = f[j]
            #print(e)
            if e not in vert_dict:
                vert_dict[e] = [i]
                #vert_dict[(e[1],e[0])] = i
            else:
                for owner in vert_dict[e]:
                    adj_f[ owner, i ] = 1
                    adj_f[ i, owner ] = 1
                vert_dict[e].append(i)
                # owner = vert_dict[e]
                # adj_f[ owner, i ] = 1
                # adj_f[ i, owner ] = 1
    return adj_f

def get_foind_from_grouping_mat(mat):
    """ mat: 
        [  [[1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1]],

           [[1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]]]
        vert_fid:
            [[0, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 2]]
        vert_oid:
            [[0, 0, 1, 0, 1, 2],
            [0, 1, 2, 0, 1, 0]]
    """
    import torch
    A = (mat.cumsum(axis=1)*mat)
    if type(mat) == torch.Tensor:
        vert_oid, vert_fid = A.max(axis=-1)
        vert_oid = vert_oid.long()
        vert_fid = vert_fid - torch.roll(vert_fid, 1, dims=1)
    elif type(mat) == np.ndarray:
        vert_oid, vert_fid = A.max(axis=-1), A.argmax(axis=-1)
        vert_oid = vert_oid.astype(int)
        vert_fid = vert_fid - np.roll(vert_fid, 1, axis=1)
    vert_fid = (vert_fid != 0).cumsum(axis=1)
    vert_fid, vert_oid = vert_fid - 1, vert_oid - 1
    return vert_fid, vert_oid

def grouping_mat2face(mat, loop=False):
    fid, oid = get_foind_from_grouping_mat(mat[None,...])
    fid, oid = fid[0], oid[0]
    vid = np.arange(len(fid))
    faces = []
    for i in range(fid.max()+1):
        face = vid[fid==i]
        if loop:
            face = np.concatenate([face, face[:1]])
        faces.append(face)
    return faces

def estimate_pointcloud_normals2(points, k=5):
    """
    Args:
    points: 2D point cloud, shape: (N, c)
    Returns:
    normals: the estimated normals, shape: (N, c)
    warning: use 5 nearest neighbors (or test will fail)
    """
    tree = cKDTree(points)
    I = tree.query(points, k=k)[1]
    p = points[I]
    m = p.mean(axis=1, keepdims=True)
    p -= m
    C = np.einsum('ijk,ijl->ikl', p, p) # equivalent to C = p.transpose(0,2,1) @ p
    W, V = np.linalg.eig(C)
    ## Warning: find the eigenvector that has lowest eigen value (numpy.argmin)
    normals = V[np.arange(W.shape[0]),:,W.argmin(axis=-1)]
    return normals
def estimate_points_normal(points):
    """
    Args:
    points: 2D point cloud, shape: (N, c)
    Returns:
    pd_vec: the (unit norm) principal directions, shape: (c,)
    """
    mean = points.mean(axis=0, keepdims=True)
    points -= mean
    C = points.transpose(1,0) @ points
    pd_len, pd_vec = np.linalg.eig(C) #compute_principal_directions
    argmin = pd_len.argmin(axis=-1)
    normal = pd_vec[:,argmin] # normal direction
    normal = normal / np.linalg.norm(normal)
    return normal, mean[0]
def estimate_plane(points):
    """
    Args:
    points: 2D point cloud, shape: (N, c)
    Returns:
    normal: the (unit norm) normal, shape: (c,)
    mean: the mean of the points, shape: (c,)
    """
    normal, mean = estimate_points_normal(points)
    d = np.dot(normal, mean)
    print( ((points - mean[None,:]) * normal[None,:]).sum(axis=-1) )
    # if d < 0:
    #     normal = -normal
    #     d = -d
    plane_param = np.array([normal[0], normal[1], normal[2], -d])

    P_aug = np.hstack((points, np.ones((points.shape[0],1))))
    dist = np.abs(P_aug @ plane_param[:,None])

    return plane_param, dist
def polymesh2planes(verts, faces, face_normals=None, quiet=True, outlier_eps=0.01):
    """
    Args:
    verts: 2D point cloud, shape: (N, c)
    faces: 2D point cloud, shape: (N, c)
    Returns:
    planes: the estimated planes, shape: (N, c)
    """
    planes = np.zeros((len(faces), 4))
    for fi, face in enumerate(faces):
        points = verts[face]
        if face_normals is not None:
            normal = face_normals[fi]
            normal = normal / np.linalg.norm(normal)
            d = np.dot(normal, points.mean(axis=0))
            # if d < 0:
            #     normal = -normal
            #     d = -d
            plane_param = np.array([-normal[0], -normal[1], -normal[2], d])
        else:
            plane_param, dist = estimate_plane(points)
        planes[fi] = plane_param

        if not quiet:
            P_aug = np.hstack((points, np.ones((points.shape[0],1))))
            # print(P_aug)
            # print(P_aug @ plane_param)
            dist = np.abs(P_aug @ plane_param)
            if dist.max() > outlier_eps:
                print(f'Polynomal face {fi} has outliers, max dist:', dist.max())
                print(dist)
        #exit(0)
    return planes

def polyloops2vf(polyloops):
    face = []
    vert = []
    vnum = 0
    for loop in polyloops:
        face.append( np.arange(len(loop), dtype=int) + vnum )
        vert.append( loop )
        vnum += len(loop)
    vert = np.concatenate(vert, axis=0)
    return vert, face
def fwe2vf(fwe_vert, fwe_wid):
    # (wire_num, n_per_wire, 3)
    wires = np.unique(fwe_wid)
    face = []
    vert = []
    vnum = 0
    for wi in wires:
        mask = fwe_wid == wi
        vert.append( fwe_vert[mask].reshape(-1, 3) )
        face.append( np.arange(len(vert[-1]), dtype=int) + vnum )
        vnum += len(vert[-1])
    vert = np.concatenate(vert, axis=0)
    return vert, face
def fw2vf(fw_vert):
    num_loop, points_per_loop = fw_vert.shape[:2]
    soup_vert = fw_vert.reshape(-1, 3)
    soup_face = np.arange(len(soup_vert)).reshape(-1, points_per_loop)

def fw2fwe(fw_vert, fw_wid):
    wid = np.unique(fw_wid)
    fwe_vert = []
    fwe_wid = fw_wid
    for wi in wid:
        mask = fw_wid == wi
        wire_vert = fw_vert[mask]
        wire_vert_next = np.roll(wire_vert, -1, axis=0)
        wev = np.stack([wire_vert, wire_vert_next], axis=1)
        fwe_vert.append(wev)
    fwe_vert = np.concatenate(fwe_vert, axis=0)
    return fwe_vert, fwe_wid

def sample_wire_from_edge_loop(edge_list, points_per_wire=64):
    """ Sample `points_per_wire` points for a wire (edge loop), from a list of edges (edge loop)
        First, evenly sample same amount of points for each edge, then randomly sample some points from each edge to make up the total number of points.

        Args:
            edge_list: list of edges (of shape (n, 3)), assuming the edges are strictly connected (last point of edge i is the same as first point of edge i+1), and form a loop
            points_per_wire: int
        Returns:
            wire_samples: (num_wire, points_per_wire, 3) face-wire point samples
            sample_eid: (num_wire, points_per_wire) edge id of each wire point sample.
    """
    edge_list = [np.array(e) for e in edge_list]
    num_edge = len(edge_list)
    s_num = points_per_wire #- num_edge * 2
    n_per_edge = s_num // num_edge
    residue = s_num % num_edge
    extra_choice = np.random.choice(num_edge, residue, replace=False)
    edge_samples = []
    sample_edge_id = [] 
    for i in range(num_edge):
        edge = edge_list[i]
        sampled = sample_line_sequence_N(edge, n_per_edge)
        #print(sampled.shape, n_per_edge, s_num, num_edge)
        edge_samples.append(sampled)
        sample_edge_id.append(np.zeros(len(sampled), dtype=int)+i)
    for choice in extra_choice:
        edge_samples[choice] = np.r_[edge_samples[choice], edge_list[choice][-1:]]
        sample_edge_id[choice] = np.r_[sample_edge_id[choice], np.array([choice])]
    wire_samples = np.concatenate(edge_samples, axis=0)
    sample_edge_id = np.concatenate(sample_edge_id, axis=0) # face-wire-edge-vert wire-edge id, local edge id in each wire
    return wire_samples, sample_edge_id
    # Unit testing
    # geoutil.sample_wire_from_edge_loop([[[0,0,0],[1,0,0]],[[1,0,0],[1,1,0],[0,1,0],[0,0,0]]], points_per_wire=5)


### OBSOLETE ###
# def sample_fw(fwe_vert, fwe_wid, wires=None, points_per_loop=64):
#     """ Sample same number of points for each wire loop 
#         Args:
#             fwe_vert: (num_edge, points_per_edge, 3)
#             fwe_wid: (num_edge,), face-wire-edge wire id.
#             points_per_loop: int
#         Returns:
#             all_samples: (num_wire, points_per_loop, 3) face-wire point samples
#             all_sample_eid: (num_wire, points_per_loop) face-wire-edge-vert wire-edge id, local edge id in each wire
#     """
#     wires = np.unique(fwe_wid)
#     num_wire = len(wires)
#     num_edge = 0
#     samples = []
#     #all_samples = np.zeros((num_wire, points_per_loop, 3))
#     all_samples=[]
#     all_fw_weid = []
#     avg_edge_num = 0
#     for wi in wires:
#         wire_edge = fwe_vert[fwe_wid==wi]
#         num_edge = len(wire_edge)
#         avg_edge_num += num_edge
#         if num_edge * 2 > points_per_loop:
#             #raise ValueError("Too many edges in #loop", wi)
#             #print("Too many edges in #loop", wi)
#             pass
#         s_num = points_per_loop #- num_edge * 2
#         n_per_edge = s_num // num_edge
#         residue = s_num % num_edge
#         #print("n_per_edge", n_per_edge, "residue", residue)
#         wire_samples = []
#         extra_choice = np.random.choice(num_edge, residue, replace=False)
#         edge_samples = []
#         sample_eid = [] # face-wire-edge-vert wire-edge id, local edge id in each wire
#         for i in range(num_edge):
#             edge = wire_edge[i]
#             sampled = sample_line_sequence_N(edge, n_per_edge)
#             edge_samples.append(sampled)
#             sample_eid.append(np.zeros(len(sampled), dtype=int)+i)
#         #print("extra_choice", extra_choice)
#         for choice in extra_choice:
#             edge_samples[choice] = np.r_[edge_samples[choice], edge_samples[choice][-1:]]
#         # for i,es in enumerate(edge_samples):
#         #     print(i, es.shape)
#         wire_samples = np.concatenate(edge_samples, axis=0)
#         sample_eid = np.concatenate(sample_eid, axis=0)
#         #all_samples[wi] = wire_samples
#         all_samples.append(wire_samples)
#         all_fw_weid.append(sample_eid)
#     all_samples = np.stack(all_samples, axis=0)
#     all_fw_weid = np.concatenate(all_fw_weid, axis=0)
#     avg_edge_num = avg_edge_num / num_wire
#     #print("avg_edge_num", avg_edge_num)
#     return all_samples, all_fw_weid


def explode_polyloops(polyloops, ext, jittered_speed=False, **kwargs):
    epolyloops = []
    for polyloop in polyloops:
        center = polyloop.mean(axis=0)
        mag = np.linalg.norm(center) 
        direc = center/mag if mag>0 else np.array([0,0,0])
        direc = direc[None, :]
        spd = ext if not jittered_speed else ext * (1+np.random.rand()/1)
        e_polyloop = polyloop + direc * spd
        epolyloops.append(e_polyloop)
    return epolyloops



# Pytorch functions

def barycentric_coordinates_2d_torch(queries, vert, face):
    """
    Calculate barycentric coordinates for multiple query points and face.

    Args:
        queries (np.ndarray): N x 2 array of query points.
        vert (np.ndarray): V x 2 array of vertex positions.
        face (np.ndarray): F x 3 array of indices into vert, defining the face.

    Returns:
        np.ndarray: N x F x 3 array of barycentric coordinates for each query point with respect to each triangle.
    """
    import torch
    # Extract the vert for each triangle (F x 3 x 2)
    A = vert[face[:, 0], :]  # (F, 2)
    B = vert[face[:, 1], :]  # (F, 2)
    C = vert[face[:, 2], :]  # (F, 2)

    # Compute vectors from point A to point B and C (F x 2)
    AB = B - A  # (F, 2)
    AC = C - A  # (F, 2)

    # Compute area of the triangle (using cross product) (F,)
    area_ABC = 0.5 * (AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0])

    # Vectors from point A to each query point (N x F x 2)
    queries = queries[:, None, :]  # (N, 1, 2)
    PA = A[None, :, :] - queries  # (N, F, 2)
    PB = B[None, :, :] - queries  # (N, F, 2)
    PC = C[None, :, :] - queries  # (N, F, 2)

    # Compute area of sub-triangles (using cross product) (N x F)
    area_PAB = 0.5 * (PA[:, :, 0] * PB[:, :, 1] - PA[:, :, 1] * PB[:, :, 0])
    area_PBC = 0.5 * (PB[:, :, 0] * PC[:, :, 1] - PB[:, :, 1] * PC[:, :, 0])
    area_PCA = 0.5 * (PC[:, :, 0] * PA[:, :, 1] - PC[:, :, 1] * PA[:, :, 0])

    # Calculate barycentric coordinates (N x F x 3)
    denominator = area_ABC[None, :].repeat(len(queries), 1)  # (N, F
    # if denominator == 0, then the triangle is degenerate, set barycentric coordinates to (-1, -1, -1)
    alpha = torch.where(denominator != 0, area_PBC / denominator, -1.+torch.zeros_like(denominator))  # (N, F)
    beta = torch.where(denominator != 0, area_PCA / denominator, -1.+torch.zeros_like(denominator))  # (N, F)
    gamma = torch.where(denominator != 0, area_PAB / denominator, -1.+torch.zeros_like(denominator))  # (N, F)

    return torch.stack((alpha, beta, gamma), axis=-1)

def interpolate_trimesh2d_torch(queries, vert, face, features):
    """
    Rasterize a triangle mesh to an interpolated image.
    Args:
        queries (np.ndarray): (N, 2) ,N is typically reso**2
        vert (np.ndarray): (V, 2)
        face (np.ndarray): (F, 3)
        features (np.ndarray): (V, C)
    Returns:
        np.ndarray: (H, W, C)
    """
    import torch
    if type(vert) is np.ndarray:
        vert = torch.from_numpy(vert).float()
        face = torch.from_numpy(face).long()
        queries = torch.from_numpy(queries).float()

    bc_coords = barycentric_coordinates_2d_torch(queries, vert, face) # (N, F, 3)
    inside_a_triangle = (bc_coords >= 0).all(axis=-1)
    inside = inside_a_triangle.any(axis=-1)
    outside = ~inside
    occupancy = inside
    # getting the first triangle that the point is inside (use pytorch nonzero)
    tri_ind = torch.argmax(inside_a_triangle.long(), dim=-1) # (H*W,)
    tri_coord = bc_coords[torch.arange(len(queries)), tri_ind]
    # interpolate 3d coordinates according to barycentric coordinates
    interpolated = tri_coord[:,:,None] * features[face[tri_ind]]
    interpolated = interpolated.sum(axis=1)

    interpolated[outside] = 0
    occupancy = occupancy.reshape(queries.shape[0])
    pixel_3d = interpolated.reshape(queries.shape[0], 3)

    return occupancy, pixel_3d

def rasterize_triangle_torch(vert, face, vert_uv, vert_values=None, resolution=(512,512), max_cuda_memory=16, device='cuda'):
    """
    Rasterize a triangle mesh to an interpolated image.
    Args:
        vert (torch.tensor): (V, 3)
        face (torch.tensor): (F, 3)
        vert_uv (torch.tensor): (V, 2) in [0,1]
        vert_values (torch.tensor, optional): (V, C). Defaults to None.
        resolution (tuple, optional): (H, W). Defaults to (512,512).
        max_cuda_memory (int, optional): Defaults to 16 (GB).
        device (str, optional): Defaults to 'cuda'.
    Returns:
        torch.tensor: (H, W, C)
    """
    import torch
    from xgutils import ptutil
    if type(vert) is np.ndarray:
        vert = torch.from_numpy(vert).float()
        face = torch.from_numpy(face).long()
        vert_uv = torch.from_numpy(vert_uv).float()
    if device == 'cuda':
        vert = vert.cuda()
        face = face.cuda()
        vert_uv = vert_uv.cuda()
    #print("face num", face.shape[0])

    grid_pts = nputil.makeGrid([0,0], [1,1], resolution, flatten=True)
    grid_pts = torch.from_numpy(grid_pts).float().to(vert)
    grid_pts = grid_pts.reshape(-1, 2)

    # calculate sub_batch_size to avoid cuda out of memory
    # mem = N * F * 3 * 3 * 4, 4 is the size of float32
    sub_batch_size = max_cuda_memory * 1024 * 1024 * 1024 // (face.shape[0] * 3 * 3 * 4) 
    # only use 1/4 to be safer
    sub_batch_size = int(sub_batch_size * 1/4.)
    # floor to the nearest power of 2
    #sub_batch_size = 2 ** int(np.log2(sub_batch_size))
    #print('sub_batch_size', sub_batch_size, 64**2, 128**2, 256**2)
    #print(grid_pts.dtype, vert.dtype, face.dtype, vert_uv.dtype)
    if sub_batch_size < 1:
        raise ValueError('Too many faces, even query 1 point at a time will still out of memory')

    occ, pixel3d = ptutil.subbatch_run_func(interpolate_trimesh2d_torch, 
        batch_input=dict(queries=grid_pts), 
        other_input=dict(vert=vert_uv, face=face, features=vert),
        sub_batch_size=sub_batch_size)
    occupancy = occ.reshape(resolution)
    pixel_3d = pixel3d.reshape(resolution[0], resolution[1], 3)
    return occupancy, pixel_3d
def image_to_mesh_torch(occ, pixel3d): # obsolete, use glbutil.meshing_objamage instead
    import torch
    pixel_index = torch.arange(occ.numel()).reshape(occ.shape).to(occ.device)
    #print(pixel_index)
    occ = occ.bool()
    is_tri_vert = occ & torch.roll(occ, -1, dims =0) & torch.roll(occ, -1, dims =1)
    verta = pixel_index
    vertb = torch.roll(pixel_index, -1, dims=1)
    vertc = torch.roll(pixel_index, -1, dims=0)
    new_face = torch.stack([verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], dim=1)
    #print(occ.shape, is_tri_vert.shape, pixel_index.shape, new_face.shape, verta.shape, vertb.shape, vertc.shape)
    #print(is_tri_vert.dtype)
    is_tri_vert = occ & torch.roll(occ, 1, dims =0) & torch.roll(occ, 1, dims =1)
    verta = pixel_index
    vertb = torch.roll(pixel_index, 1, dims=1)
    vertc = torch.roll(pixel_index, 1, dims=0)
    new_face2 = torch.stack([verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], dim=1)
    new_face = torch.cat([new_face, new_face2], dim=0)

    return pixel3d.reshape(-1, 3), new_face.reshape(-1, 3)

def rasterize_triangle_torch2(vert, face, vert_uv, vert_values=None, resolution=(512,512)):
    """
    Rasterize a triangle mesh to an interpolated image.
    Args:
        vert (torch.tensor): (V, 3)
        face (torch.tensor): (F, 3)
        vert_uv (torch.tensor): (V, 2) in [0,1]
        vert_values (torch.tensor, optional): (V, C). Defaults to None.
        resolution (tuple, optional): (H, W). Defaults to (512,512).
    Returns:
        torch.tensor: (H, W, C)
    """
    import torch
    if type(vert) is np.ndarray:
        vert = torch.from_numpy(vert).float()
        face = torch.from_numpy(face).long()
        vert_uv = torch.from_numpy(vert_uv).float()
    #print(type(vert), type(face), type(vert_uv  ))

    grid_pts = nputil.makeGrid([0,0], [1,1], resolution, flatten=True)
    grid_pts = torch.from_numpy(grid_pts).float().to(vert)
    bc_coords = barycentric_coordinates_2d_torch(grid_pts, vert_uv, face) # (H*W, F, 3)
    inside_a_triangle = (bc_coords >= 0).all(axis=-1)
    inside = inside_a_triangle.any(axis=-1)
    outside = ~inside
    occupancy = inside
    # getting the first triangle that the point is inside (use pytorch nonzero)
    tri_ind = torch.argmax(inside_a_triangle.long(), dim=-1) # (H*W,)
    tri_coord = bc_coords[torch.arange(len(grid_pts)), tri_ind]
    # interpolate 3d coordinates according to barycentric coordinates
    interpolated = tri_coord[:,:,None] * vert[face[tri_ind]]
    interpolated = interpolated.sum(axis=1)
    interpolated[outside] = 0
    occupancy = occupancy.reshape(resolution)
    pixel_3d = interpolated.reshape(resolution[0], resolution[1], 3)

    return occupancy, pixel_3d

