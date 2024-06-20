"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Scalar field functions -
"""

import numpy as np
from numpy import linalg as la

from scalar_field.utils import Q_prod_xi, exp
from toolbox.math_utils import two_dim
    
# (all of the following classes need "eval" and "grad" functions, 
#  and the variable "x0")

# ----------------------------------------------------------------------
# Gaussian function
# ----------------------------------------------------------------------

class sigma_gauss:
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
    """
    def __init__(self, mu=[0,0], max_intensity=100, dev=10, S=None, R=None):
        n = 2
        self.max_intensity = max_intensity
        self.dev = dev

        # Variables needed for the sigma (common) class
        self.x0  = mu
        # ---

        if S is None:
            S = -np.eye(n)
        if R is None:
            R = np.eye(n)
        self.Q = R.T@S@R/(2*self.dev**2)

    def eval(self, X):
        X = two_dim(X)
        sigma = self.max_intensity * exp(X,self.Q,self.x0) / np.sqrt(2*np.pi*self.dev**2)
        return sigma

    def grad(self, X):
        X = two_dim(X)
        return Q_prod_xi(self.Q,X-self.x0) * self.eval(X)

# ----------------------------------------------------------------------
# Non-convex function (two Gaussians + contant * norm)
# ----------------------------------------------------------------------

# Default parameters
S1 = 0.9*np.array([[1/np.sqrt(30),0], [0,1]])
S2 = 0.9*np.array([[1,0], [0,1/np.sqrt(15)]])
A = (1/np.sqrt(2))*np.array([[1,-1], [1,1]])
a = [1, 0]
b = [0,-2]

class sigma_nonconvex:
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
        a: list
            center of the first Gaussian
        b: list
            center of the second Gaussian
        Qa: numpy array
            2x2 matrix, quadratic transformation of the first Gaussian input
        Qb: numpy array
            2x2 matrix, quadratic transformation of the second Gaussian input
    """
    def __init__(self, k, mu=[0,0], dev=1, a=a, b=b, Qa=-S1, Qb=-A.T@S2@A):
        self.k = k
        self.dev = dev

        # Variables needed for the sigma (common) class
        self.x0 = mu
        # ---

        if isinstance(a, list):
            a = np.array(a)
        if isinstance(b, list):
            b = np.array(b)
        self.a = a
        self.b = b
        self.Qa = Qa
        self.Qb = Qb

    def eval(self, X):
        X = two_dim(X)
        X = (X - self.x0)/self.dev
        sigma = - 2 - exp(X,self.Qa,self.a) - exp(X,self.Qb,self.b) + self.k*la.norm(X, axis=1)
        return -sigma

    def grad(self, X):
        X = two_dim(X)
        X = (X - self.x0)/self.dev
        alfa = 0.0001 # Trick to avoid x/0
        sigma_grad = - Q_prod_xi(self.Qa,X-self.a) * exp(X,self.Qa,self.a)[:,None] \
                            - Q_prod_xi(self.Qb,X-self.b) * exp(X,self.Qb,self.b)[:,None] \
                            + self.k * X / (la.norm(X, axis=1)[:,None] + alfa)
        return -sigma_grad

# Analyzing the previous cases, we propose a function that allows us to play much more with the 
# generation of scalar fields.
class sigma_fract:
    """
    Fractal scalar field

    Attributes
    ----------
    * k: norm factor.
    * dev: models the scale of the distribution while maintaining its properties.
    """
    def __init__(self, k, mu=[0,0], dev=[1,1], a=a, b=b, Qa=-S1, Qb=-A.T@S2@A):
        self.k = k
        self.dev = dev

        # Variables needed for the sigma (common) class
        self.x0 = mu
        # ---

        if type(a) == list:
            a = np.array(a)
        if type(b) == list:
            b = np.array(b)
        self.a = a
        self.b = b
        self.Qa = Qa
        self.Qb = Qb

    def eval(self, X):
        X = two_dim(X)
        X = (X - self.x0)
        c1 = - exp(X/self.dev[0],self.Qa,self.a) - exp(X/self.dev[0],self.Qb,self.b)
        c2 = - exp(X/self.dev[1],self.Qa,self.a) - exp(X/self.dev[1],self.Qb,self.b)
        x_dist = la.norm(X, axis=1)
        sigma = - 2 + 2*c1 + c2 + self.k*x_dist
        return -sigma

    def grad(self, X):
        X = two_dim(X)
        X = (X - self.x0)/self.dev
        alfa = 0.0001 # Trick to avoid x/0
        sigma_grad = - Q_prod_xi(self.Qa,X-self.a) * exp(X,self.Qa,self.a)[:,None] \
                        - Q_prod_xi(self.Qb,X-self.b) * exp(X,self.Qb,self.b)[:,None] \
                        + self.k * X / (la.norm(X, axis=1)[:,None] + alfa)
        return -sigma_grad