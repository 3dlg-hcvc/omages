import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from mpl_toolkits.mplot3d import Axes3D, proj3d, art3d
import matplotlib.patheffects as path_effects
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def mesh_color_by_normal(vert, face):
    V = vert[face]
    T =  V[:,:,:3]
    v1, v2 = V[:,1]-V[:,0], V[:,2]-V[:,0]
    # manual cross product
    N = np.c_[v1[:,1]*v2[:,2]-v1[:,2]*v2[:,1], v1[:,2]*v2[:,0]-v1[:,0]*v2[:,2], v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0]]
    #N /= np.linalg.norm(N, axis=1).reshape(-1,1)
    # avoid divid by zero
    N = np.divide(N, np.linalg.norm(N, axis=1).reshape(-1,1), out=np.zeros_like(N), where=np.linalg.norm(N, axis=1).reshape(-1,1)!=0)
    C = (N+1)/2
    C = np.c_[C, np.ones(len(C))] # add alpha
    # duplicate 3 times
    C = np.repeat(C, 3, axis=0)
    return C

def add_axes(ax, bmin, bmax):
    for d in np.arange(0, 11):
        color, linewidth, zorder = "0.75", 0.5, -100
        d = (d - 5) / 5.0
        if d in [bmin, 0, bmax]:
            color, linewidth, zorder = "0.5", 0.75, -50
        ax.plot([bmin, bmin], [d, d], [bmin, bmax], linewidth=linewidth, color=color, zorder=zorder)
        ax.plot([bmin, bmin], [bmax, bmin], [d, d], linewidth=linewidth, color=color, zorder=zorder)
        ax.plot([bmin, bmax], [bmax, bmax], [d, d], linewidth=linewidth, color=color, zorder=zorder)
        ax.plot([d, d], [bmax, bmin], [bmin, bmin], linewidth=linewidth, color=color, zorder=zorder)
        ax.plot([bmin, bmax], [d, d], [bmin, bmin], linewidth=linewidth, color=color, zorder=zorder)
        ax.plot([d, d], [bmax, bmax], [bmin, bmax], linewidth=linewidth, color=color, zorder=zorder)

    size, color = "small", "0.75"
    for d in np.arange(1, 11):
        d = (d - 5) / 5.0
        ax.text(d, bmin-.1, bmin, "%.1f" % d, color=color, size=size, ha="center", va="center")
        ax.text(bmax+.1, d, bmin, "%.1f" % d, color=color, size=size, ha="center", va="center")
        ax.text(bmin, bmin-.1, d, "%.1f" % d, color=color, size=size, ha="center", va="center")
        ax.text(bmax+.1, bmax, d, "%.1f" % d, color=color, size=size, ha="center", va="center")
        ax.text(d, bmax, bmax+.1, "%.1f" % d, color=color, size=size, ha="center", va="center")
        ax.text(bmin, d, bmax+.1, "%.1f" % d, color=color, size=size, ha="center", va="center")


    plt.plot([bmin, bmax], [bmax, bmax], [bmin, bmin], linewidth=1.5, color="red")

    plt.plot([bmin, bmin], [bmax, bmin], [bmin, bmin], linewidth=1.5, color="green")

    plt.plot([bmin, bmin], [bmax, bmax], [bmin, bmax], linewidth=1.5, color="blue")

    #text = ax.text(bmin, bmin, bmin, "O", color="black", size="large", ha="center", va="center")
    #text.set_bbox(dict(facecolor='white', alpha=1., edgecolor='white'))
    #text.set_path_effects(
    #    [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
    #)
    return ax

def add_cloud(vert, fig=None, ax=None, show_axes=False, show_proj=False):
    if fig is None:
        # Style
        # -----------------------------------------------------------------------------
        # plt.rc('font', family="Roboto Condensed")
        bmin, bmax = -1, 1
        plt.rc("font", family="Ubuntu")
        plt.rc("xtick", labelsize="small")
        plt.rc("ytick", labelsize="small")
        plt.rc("axes", labelsize="medium", titlesize="medium")
        fig = plt.figure(figsize=(6, 6))
    if ax is None:
        ax = plt.subplot(1, 1, 1, projection="3d")
        ax.set_xlim(bmin, bmax)
        ax.set_ylim(bmax, bmin)
        ax.set_ylim(bmin, bmax)
        ax.set_zlim(bmin, bmax)
        ax.set_axis_off()
        ax.set_aspect("equal")
    if show_axes:
        add_axes(ax, bmin, bmax)
    if show_proj:
        vo = np.argsort(vert[:,2])#[::-1]
        ax.scatter(vert[vo,0], vert[vo,1], -vert[vo,2]*.01+bmin, s=6.5, c=vert[vo,2], alpha=0.5, zorder=100, cmap='bone')
        vo = np.argsort(vert[:,1])#[::-1]
        ax.scatter(vert[vo,0], -vert[vo,1]*.01+bmax, vert[vo,2], s=6.5, c=vert[vo,1], alpha=0.5, zorder=100, cmap='bone')
        vo = np.argsort(vert[:,0])#[::-1]
        ax.scatter(-vert[vo,0]*0.01+bmin, vert[vo,1], vert[vo,2], s=6.5, c=vert[vo,0], alpha=0.5, zorder=100, cmap='bone')
    ax.scatter(vert[:,0], vert[:,1], vert[:,2], s=.1, c=vert[:,0], alpha=0.99, zorder=100)
    plt.tight_layout()
    return fig, ax
def add_mesh(vert, face, vert_color=None, fig=None, ax=None, show_axes=False, show_proj=False):
    if fig is None:
        # Style
        # -----------------------------------------------------------------------------
        # plt.rc('font', family="Roboto Condensed")
        bmin, bmax = -1, 1
        plt.rc("font", family="Ubuntu")
        plt.rc("xtick", labelsize="small")
        plt.rc("ytick", labelsize="small")
        plt.rc("axes", labelsize="medium", titlesize="medium")
        fig = plt.figure(figsize=(6, 6))
    if ax is None:
        ax = plt.subplot(1, 1, 1, projection="3d")
        ax.set_xlim(bmin, bmax)
        ax.set_ylim(bmax, bmin)
        ax.set_ylim(bmin, bmax)
        ax.set_zlim(bmin, bmax)
        ax.set_axis_off()
        ax.set_aspect("equal")
    if show_axes:
        add_axes(ax, bmin, bmax)
    if show_proj:
        vo = np.argsort(vert[:,2])#[::-1]
        ax.scatter(vert[vo,0], vert[vo,1], -vert[vo,2]*.01+bmin, s=6.5, c=vert[vo,2], alpha=0.5, zorder=100, cmap='bone')
        vo = np.argsort(vert[:,1])#[::-1]
        ax.scatter(vert[vo,0], -vert[vo,1]*.01+bmax, vert[vo,2], s=6.5, c=vert[vo,1], alpha=0.5, zorder=100, cmap='bone')
        vo = np.argsort(vert[:,0])#[::-1]
        ax.scatter(-vert[vo,0]*0.01+bmin, vert[vo,1], vert[vo,2], s=6.5, c=vert[vo,0], alpha=0.5, zorder=100, cmap='bone')
    V = vert[face]
    T =  V[:,:,:3]
    Z = V[:,:,0].mean(axis=1)
    #zmin, zmax = Z.min(), Z.max()
    #Z = (Z-zmin)/(zmax-zmin)
    #C = plt.get_cmap("magma")(Z)
    v1, v2 = V[:,1]-V[:,0], V[:,2]-V[:,0] # (F, 3)
    # manual cross product
    N = np.c_[v1[:,1]*v2[:,2]-v1[:,2]*v2[:,1], v1[:,2]*v2[:,0]-v1[:,0]*v2[:,2], v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0]]
    N /= np.linalg.norm(N, axis=1).reshape(-1,1) # (F, 3)
    C = (N+1)/2 
    C = np.c_[C, np.ones(len(C))] # (F, 4)
    #C = C[I, :]
    #I = np.argsort(Z)
    I = np.arange(len(Z))
    T, C = T[I,:], C[I,:]
    if vert_color is not None:
        C = vert_color
        C = np.c_[C, np.ones(len(C))]
    #collection = Poly3DCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor="black")
    collection = Poly3DCollection(T, closed=True, linewidth=0.05, facecolor=C, edgecolor="black")
    ax.add_collection(collection)
    plt.tight_layout()
    return fig, ax
