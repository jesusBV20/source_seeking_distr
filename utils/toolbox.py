"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
from numpy import linalg as la
import random

# Graphic tools
import matplotlib.pylab as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# ----------------------------------------------------------------------
# Mathematical tools
# ----------------------------------------------------------------------

# Affine transformation
M_scale = lambda s, n=2: np.eye(n)*s

# Affine transformation
M_scale = lambda s, n=2: np.eye(n)*s

# Rotation matrix
M_rot = lambda psi: np.array([[np.cos(psi), -np.sin(psi)], \
                              [np.sin(psi),  np.cos(psi)]])

# Angle between two vectors (matrix computation)
def angle_of_vectors(A,B):
    cosTh = np.sum(A*B, axis=1)
    sinTh = np.cross(A,B, axis=1)
    theta = np.arctan2(sinTh,cosTh)
    return theta

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def XY_distrib(N, rc0, lims, n=2):
    """
    Function to generate uniform rectangular distributions.

    Attributes
    ----------
    N: int
        number of points
    rc0: numpy array
        position [x,y,...] in the real space of the centroid
    lims: numpy array
        distance limits [lim_x,lim_y,...] in each dimension of the real space
    n: int
        dimension of the real space
    """
    if len(rc0) + len(lims) != n*2:
        raise Exception("The length of rc0 and lims should be equal to n")
        
    X0 = (np.random.rand(N,n) - 0.5)*2
    for i in range(n):
        X0[:,i] = X0[:,i] * lims[i]
    return rc0 + X0

def gen_random_graph(N):
    non_visited_nd = set(range(10))
    non_visited_nd.remove(0)
    visited_nd = {0}

    Z = []
    while len(non_visited_nd) != 0:
        i = random.choice(list(visited_nd))
        j = random.choice(list(non_visited_nd))
        visited_nd.add(j)
        non_visited_nd.remove(j)
        Z.append((i,j))
        
    return Z

# ----------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------

def kw_def_arrow(scale):
    return {"lw":2*scale**(1/5), "hw":0.05*scale, "hl":0.1*scale}

def kw_def_patch(scale):
    return {"size":0.15*scale, "lw":0.2*scale**(1/2)}

def unicycle_patch(XY, yaw, color, size=1, lw=0.5):
    """
    A function to draw the unicycle robot patch.

    Attributes
    ----------
    XY: list
        position [X, Y] of the unicycle patch
    yaw: float
        unicycle heading
    size: float
        unicycle patch scale
    lw: float
        unicycle patch linewidth
    """
    Rot = np.array([[np.cos(yaw), np.sin(yaw)],[-np.sin(yaw), np.cos(yaw)]])

    apex = 45*np.pi/180 # 30 degrees apex angle
    b = np.sqrt(1) / np.sin(apex)
    a = b*np.sin(apex/2)
    h = b*np.cos(apex/2)

    z1 = size*np.array([a/2, -h*0.3])
    z2 = size*np.array([-a/2, -h*0.3])
    z3 = size*np.array([0, h*0.6])

    z1 = Rot.dot(z1)
    z2 = Rot.dot(z2)
    z3 = Rot.dot(z3)

    verts = [(XY[0]+z1[1], XY[1]+z1[0]), \
             (XY[0]+z2[1], XY[1]+z2[0]), \
             (XY[0]+z3[1], XY[1]+z3[0]), \
             (0, 0)]

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)

    return patches.PathPatch(path, fc=color, lw=lw)