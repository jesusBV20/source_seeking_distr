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
from matplotlib.colors import ListedColormap

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

def gen_random_graph(N, rounds=1):
    """
    Function that generates a random graph using a simple heuristic
    """
    Z = []
    for round in range(rounds):
        non_visited_nd = set(range(N))
        non_visited_nd.remove(0)
        visited_nd = {0}

        while len(non_visited_nd) != 0:
            i = random.choice(list(visited_nd))
            j = random.choice(list(non_visited_nd))
            visited_nd.add(j)
            non_visited_nd.remove(j)

            if (i,j) not in Z:
                Z.append((i,j))
        
    return Z

def L_sigma(X, sigma, denom=None):
    """
    Funtion to compute L_sigma.

    Attributes
    ----------
        X: numpy array
            (N x 2) matrix of agents position from the centroid
        sigma: numpy array
            (N) vector of simgma_values on each row of X
    """
    N = X.shape[0]
    l_sigma_hat = sigma[:,None].T @ X
    if denom == None:
        x_norms = np.zeros((N))
        for i in range(N):
            x_norms[i] = X[i,:] @ X[i,:].T
            D_sqr = np.max(x_norms)
        l_sigma_hat = l_sigma_hat / (N * D_sqr)
    else:
        l_sigma_hat = l_sigma_hat/denom
    return l_sigma_hat.flatten()

def unit_vec(v):
    if la.norm(v) > 0:
        return v/la.norm(v)
    else:
        return v
    
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

def alpha_cmap(cmap, alpha):
  # Get the colormap colors
  my_cmap = cmap(np.arange(cmap.N))

  # Define the alphas in the range from 0 to 1
  alphas = np.linspace(alpha, alpha, cmap.N)

  # Define the background as white
  BG = np.asarray([1., 1., 1.,])

  # Mix the colors with the background
  for i in range(cmap.N):
      my_cmap[i,:-1] = my_cmap[i,:-1] * alphas[i] + BG * (1.-alphas[i])

  # Create new colormap which mimics the alpha values
  my_cmap = ListedColormap(my_cmap)
  return my_cmap


def vector2d(axis, P0, Pf, c="k", ls="-", lw = 0.7, hw=0.1, hl=0.2, alpha=1, zorder=1):
    """
    Function to easy plot a 2D vector
    """
    axis.arrow(P0[0], P0[1], Pf[0], Pf[1],
                lw=lw, color=c, ls=ls,
                head_width=hw, head_length=hl, 
                length_includes_head=True, alpha=alpha, zorder=zorder)