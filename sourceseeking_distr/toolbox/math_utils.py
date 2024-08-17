"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Common mathematical utilities -
"""

import numpy as np
import random

from numpy.linalg import norm

# --------------------------------------------------------------------------------------
# COMMON MATH UTILS

# Affine transformation
M_scale = lambda s, n=2: np.eye(n) * s

# Affine transformation
M_scale = lambda s, n=2: np.eye(n) * s

# Rotation matrix
M_rot = lambda psi: np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])


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


def unit_vec(V, delta=0):
    """
    Normalise a bundle of 2D vectors
    """
    if len(V.shape) == 1:
        if norm(V) > delta:
            V = V / norm(V)
    else:
        for i, v in enumerate(V):
            if norm(v) > delta:
                V[i, ...] = v / norm(v)
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
    l_sigma_hat = sigma[:, None].T @ X

    x_norms = np.zeros((N))
    for i in range(N):
        x_norms[i] = X[i, :] @ X[i, :].T
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
    if len(rc0) + len(lims) != n * 2:
        raise Exception("The length of rc0 and lims should be equal to n")

    X0 = (np.random.rand(N, n) - 0.5) * 2
    for i in range(n):
        X0[:, i] = X0[:, i] * lims[i]
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

            if (i, j) not in Z:
                Z.append((i, j))

    return Z


def gen_Z_distance(P, dist_thr):
    y2 = np.sum(P**2, axis=1)
    x2 = y2.reshape(-1, 1)
    dist = np.sqrt(x2 - 2 * P @ P.T + y2)

    mask = dist + 2 * np.eye(dist.shape[0]) * dist_thr <= dist_thr
    Z = [(i, j) for i, j in zip(*np.where(mask))]
    return Z


def gen_Z_split(N, order):
    X = np.ones((N, 2))
    for i in range(order):
        if i != order - 1:
            X[i * int(N / order) : (i + 1) * int(N / order), :] = [i, i]
        else:
            X[i * int(N / order) :, :] = [i, i]

    y2 = np.sum(X**2, axis=1)
    x2 = y2.reshape(-1, 1)
    dist = np.sqrt(x2 - 2 * X @ X.T + y2)

    dist_thr = 0.1
    mask = dist + 2 * np.eye(dist.shape[0]) * dist_thr <= dist_thr

    Z = [(i, j) for i, j in zip(*np.where(mask))]
    return Z


def step_repeat(array, order):
    n = order - 1
    rep_mask = (np.arange(array.shape[0]) % n == (n - 1)) + 1
    rep_mask = [1, *rep_mask[:-2], 1]
    return np.repeat(array, rep_mask)


# --------------------------------------------------------------------------------------
# SCALAR_FIELD MATH UTILS


def Q_prod_xi(Q, X):
    """
    Matrix Q dot each row of X
    """
    return (Q @ X.T).T


def exp(X, Q, mu):
    """
    Exponential function with quadratic form:
                            exp(r) = e^((r - mu)^t @ Q @ (r - mu))
    """
    return np.exp(np.sum((X - mu) * Q_prod_xi(Q, X - mu), axis=1))


# -------------------------------------------------------------------------------------
# SIMULATOR MATH UTILS

# ----------------------------------------------------------------------
# Mathematical utility functions
# ----------------------------------------------------------------------


def build_B(list_edges, N):
    """
    Generate the incidence matrix
    """
    B = np.zeros((N, len(list_edges)))
    for i in range(len(list_edges)):
        B[list_edges[i][0], i] = 1
        B[list_edges[i][1], i] = -1
    return B


def build_L_from_B(B):
    """
    Generate the Laplacian matrix by using the incidence matrix (unit weights)
    """
    L = B @ B.T
    return L


def angle_of_vectors(A, B):
    """
    Calculate the angle between two vectors (matrix computing)
    """
    cosTh = np.sum(A * B, axis=1)
    sinTh = np.cross(A, B, axis=1)
    theta = np.arctan2(sinTh, cosTh)
    return theta


# ----------------------------------------------------------------------
# Centroid and ascending direction estimation dynamics
# ----------------------------------------------------------------------


def dyn_centroid_estimation(xhat, t, Lb, p, k=1):
    xhat_dt = -k * (Lb.dot(xhat) - Lb.dot(p))
    return xhat_dt


def dyn_mu_estimation(mu, t, Lb, k=1):
    mu_dt = -k * (Lb.dot(mu))
    return mu_dt


# -------------------------------------------------------------------------------------
