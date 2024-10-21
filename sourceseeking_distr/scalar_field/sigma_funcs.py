"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Scalar field functions -
"""

import numpy as np
from numpy import linalg as la

from .sigma_common import SigmaField

# Our utils
from ..toolbox.math_utils import Q_prod_xi, exp, two_dim

# --------------------------------------------------------------------------------------

# Default parameters
S1 = 0.9 * np.array([[1 / np.sqrt(30), 0], [0, 1]])
S2 = 0.9 * np.array([[1, 0], [0, 1 / np.sqrt(15)]])
A = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
a = [1, 0]
b = [0, -1.5]

# ----------------------------------------------------------------------
# Gaussian function
# ----------------------------------------------------------------------


class SigmaGauss(SigmaField):
    """
    Gaussian scalar field function

    Attributes
    ----------
        mu: list
            center of the Gaussian.
        max_intensity: float
            scalar field value at the source
        dev: float
            models the width of the Gaussian
        S: numpy array
            2x2 matrix, rotation matrix applied to the scalar field
        R: numpy array
            2x2 matrix, scaling matrix applied to the scalar field
    """

    def __init__(self, mu=[0, 0], max_intensity=100, dev=10):
        self.x0 = mu
        self.max_intensity = max_intensity
        self.dev = dev
        self.Q = -np.eye(2) / (2 * self.dev**2)

        self._mu = mu

    @property
    def mu(self):
        return self._mu

    def eval_value(self, X):
        X = two_dim(X)
        sigma = (
            self.max_intensity
            * exp(X, self.Q, self.x0)
            / np.sqrt(2 * np.pi * self.dev**2)
        )
        return sigma

    # sigma(p) \prop exp(g(p)), where g(p) = (X-x0)^T @ Q @ (X-x0)

    def eval_grad(self, X):
        X = two_dim(X)
        x, y = X[0,0], X[0,1]
        x0, y0 = self.x0[0], self.x0[1] 
        q11,q12 = self.Q[0,0], self.Q[0,1]
        q21,q22 = self.Q[1,0], self.Q[1,1]

        # Compute the gradient
        g_dx = 2 * q11 * (x-x0) + (q12 + q21) * (y-y0)
        g_dy = 2 * q22 * (y-y0) + (q12 + q21) * (x-x0)

        return np.array([[g_dx, g_dy]]) * self.value(X)
    
    def eval_hessian(self, X):
        X = two_dim(X)
        x, y = X[0,0], X[0,1]
        x0, y0 = self.x0[0], self.x0[1]
        q11,q12 = self.Q[0,0], self.Q[0,1]
        q21,q22 = self.Q[1,0], self.Q[1,1]

        # Compute the hessian
        g_dx = 2 * q11 * (x-x0) + (q12 + q21) * (y-y0)
        g_dy = 2 * q22 * (y-y0) + (q12 + q21) * (x-x0)

        g_dxx = 2 * q11
        g_dxy = (q12 + q21)
        g_dyx = (q12 + q21)
        g_dyy = 2 * q22

        H = np.zeros((2,2))
        H[0,0] = self.value(X) * (g_dx * g_dx + g_dxx) 
        H[0,1] = self.value(X) * (g_dx * g_dy + g_dyx)
        H[1,0] = self.value(X) * (g_dy * g_dx + g_dxy)
        H[1,1] = self.value(X) * (g_dy * g_dy + g_dyy)

        return H

# ----------------------------------------------------------------------
# Non-convex function (two Gaussians + contant * norm)
# ----------------------------------------------------------------------


class SigmaNonconvex(SigmaField):
    """
    Non-convex scalar field function (two Gaussians + "norm factor" * norm)

    Attributes
    ----------
        k: float
            norm factor
        mu: list
            center of the Gaussian.
        dev: float
            models the scale of the distribution while maintaining its properties
        a: np.ndarray
            center of the first Gaussian
        b: np.ndarray
            center of the second Gaussian
        Qa: numpy array
            2x2 matrix, quadratic transformation of the first Gaussian input
        Qb: numpy array
            2x2 matrix, quadratic transformation of the second Gaussian input
    """

    def __init__(self, k, mu=[0, 0], dev=1, a=a, b=b, Qa=-S1, Qb=-A.T @ S2 @ A):
        self.x0 = mu
        self.k = k
        self.dev = dev

        if isinstance(a, list):
            a = np.array(a)
        if isinstance(b, list):
            b = np.array(b)
        self.a = a
        self.b = b
        self.Qa = Qa
        self.Qb = Qb

        self._mu = mu
        self._mu = self.find_max(mu)

    @property
    def mu(self):
        return self._mu

    def eval_value(self, X):
        X = two_dim(X)
        X = (X - self.x0) / self.dev
        sigma = (
            -2
            - exp(X, self.Qa, self.a)
            - exp(X, self.Qb, self.b)
            + self.k * la.norm(X, axis=1)
        )
        return -sigma

    def eval_grad(self, X):
        X = two_dim(X)
        X = (X - self.x0) / self.dev
        alfa = 0.0001  # Trick to avoid x/0
        sigma_grad = (
            -Q_prod_xi(self.Qa, X - self.a) * exp(X, self.Qa, self.a)[:, None]
            - Q_prod_xi(self.Qb, X - self.b) * exp(X, self.Qb, self.b)[:, None]
            + self.k * X / (la.norm(X, axis=1)[:, None] + alfa)
        )
        return -sigma_grad

    def eval_hessian(self, X):
        return None

# ----------------------------------------------------------------------
# Fractal function (two Gaussians + contant * norm)
# ----------------------------------------------------------------------


# Analyzing the previous cases, we propose a function that allows us to play much more
# with the generation of scalar fields
class SigmaFract(SigmaField):
    """
    Fractal scalar field

    Attributes
    ----------
        k: float
            norm factor
        mu: list
            center of the Gaussian.
        dev: float
            models the scale of the distribution while maintaining its properties
        a: np.ndarray
            center of the first Gaussian
        b: np.ndarray
            center of the second Gaussian
        Qa: numpy array
            2x2 matrix, quadratic transformation of the first Gaussian input
        Qb: numpy array
            2x2 matrix, quadratic transformation of the second Gaussian input
    """

    def __init__(self, k, mu=[0, 0], dev=[1, 1], a=a, b=b, Qa=-S1, Qb=-A.T @ S2 @ A):
        self.x0 = mu
        self.k = k
        self.dev = dev

        if type(a) == list:
            a = np.array(a)
        if type(b) == list:
            b = np.array(b)
        self.a = a
        self.b = b
        self.Qa = Qa
        self.Qb = Qb

        self._mu = mu
        self._mu = self.find_max(mu)

    @property
    def mu(self):
        return self._mu

    def eval_value(self, X):
        X = two_dim(X)
        X = X - self.x0
        c1 = -exp(X / self.dev[0], self.Qa, self.a) - exp(
            X / self.dev[0], self.Qb, self.b
        )
        c2 = -exp(X / self.dev[1], self.Qa, self.a) - exp(
            X / self.dev[1], self.Qb, self.b
        )
        x_dist = la.norm(X, axis=1)
        sigma = -2 + 2 * c1 + c2 + self.k * x_dist
        return -sigma

    def eval_grad(self, X):
        X = two_dim(X)
        X = (X - self.x0) / self.dev
        alfa = 0.0001  # Trick to avoid x/0
        sigma_grad = (
            -Q_prod_xi(self.Qa, X - self.a) * exp(X, self.Qa, self.a)[:, None]
            - Q_prod_xi(self.Qb, X - self.b) * exp(X, self.Qb, self.b)[:, None]
            + self.k * X / (la.norm(X, axis=1)[:, None] + alfa)
        )
        return -sigma_grad

    def eval_hessian(self, X):
        return None