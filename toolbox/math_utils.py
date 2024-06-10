"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Common mathematical utilities -
"""

import numpy as np
import random

from numpy.linalg import norm

# ----------------------------------------------------------------------

# Affine transformation
M_scale = lambda s, n=2: np.eye(n)*s

# Affine transformation
M_scale = lambda s, n=2: np.eye(n)*s

# Rotation matrix
M_rot = lambda psi: np.array([[np.cos(psi), -np.sin(psi)], \
                              [np.sin(psi),  np.cos(psi)]])

def two_dim(X):
    """
    Check if the dimensions are correct and adapt the input to 2D
    """
    if isinstance(X, list):
        X = np.array(X)
    
    if len(X.shape) < 2:
        return np.array([X])
    if len(X.shape) > 2:
        raise Exception("The dimensionality of X is greater than 2!")
    return X

def unit_vec(V):
    """
    Normalise a bundle of 2D vectors
    """
    if len(V.shape) == 1:
        if norm(V) > 0:
            V = V/norm(V)
    else:
        for i,v in enumerate(V):
            if norm(v) > 0:
                V[i,...] = v/norm(v)
    return V

def L_sigma(X, sigma):
    """
    Cetralised L_sigma calculation function (only for numerical validation).

    Attributes
    ----------
        X: numpy array
            (N x 2) matrix of agents position from the centroid
        sigma: numpy array
            (N) vector of simgma_values on each row of X
    """
    N = X.shape[0]
    l_sigma_hat = sigma[:,None].T @ X

    x_norms = np.zeros((N))
    for i in range(N):
        x_norms[i] = X[i,:] @ X[i,:].T
        D_sqr = np.max(x_norms)
    l_sigma_hat = l_sigma_hat / (N * D_sqr)

    return l_sigma_hat.flatten()

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