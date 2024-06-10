"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Simulator class utilities -
"""

import numpy as np

# ----------------------------------------------------------------------
# Mathematical utility functions
# ----------------------------------------------------------------------

def build_B(list_edges, n):
    """
    Generate the incidence matrix.
    """
    B = np.zeros((n,len(list_edges)))
    for i in range(len(list_edges)):
        B[list_edges[i][0]-1, i] = 1
        B[list_edges[i][1]-1, i] = -1
    return B

def build_L_from_B(B):
    """
    Generate the Laplacian matrix by using the incidence matrix (unit weights).
    """
    L = B@B.T
    return L

def angle_of_vectors(A,B):
    """
    Calculate the angle between two vectors (matrix computing).
    """
    cosTh = np.sum(A*B, axis=1)
    sinTh = np.cross(A,B, axis=1)
    theta = np.arctan2(sinTh,cosTh)
    return theta
    
# ----------------------------------------------------------------------
# Centroid and ascending direction estimation dynamics
# ----------------------------------------------------------------------

def dyn_centroid_estimation(xhat, t, Lb, p, k=1):
    xhat_dt = - k*(Lb.dot(xhat) - Lb.dot(p))
    return xhat_dt

def dyn_mu_estimation(mu, t, Lb, k=1):
    mu_dt = - k*(Lb.dot(mu))
    return mu_dt