"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
- Scalar field functions -
"""

import numpy as np
from numpy import linalg as la

# Matrix Q dot each row of X
Q_prod_xi = lambda Q,X: (Q @ X.T).T

# Exponential function with quadratic form: exp(r) = e^((r - mu)^t @ Q @ (r - mu)).
exp = lambda X,Q,mu: np.exp(np.sum((X - mu) * Q_prod_xi(Q,X - mu), axis=1))

# Check if the dimensions are correct and adapt the input to 2D
def two_dim(X):
    if type(X) == list:
        return np.array([[X]])
    elif len(X.shape) < 2:
        return np.array([X])
    else:
        return X
    
# ----------------------------------------------------------------------
# Scalar fields used in simulations
# (all these classes need eval and grad functions)
# ----------------------------------------------------------------------

"""
Gaussian function.
  * a: center of the Gaussian.
  * dev: models the width of the Gaussian.
"""
class sigma_gauss:
  def __init__(self, mu=[0,0], n=2, max_intensity=100, dev=10, S=None, R=None):
    self.n = n
    self.max_intensity = max_intensity
    self.dev = dev

    # Variables needed for the sigma class
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

"""
Non-convex function: two Gaussians plus factor * norm.
  * k: norm factor.
  * dev: models the scale of the distribution while maintaining its properties.
"""
# Default parameters
S1 = 0.9*np.array([[1/np.sqrt(30),0], [0,1]])
S2 = 0.9*np.array([[1,0], [0,1/np.sqrt(15)]])
A = (1/np.sqrt(2))*np.array([[1,-1], [1,1]])
a = np.array([1, 0])
b = np.array([0,-2])

class sigma_nonconvex:
  def __init__(self, k, mu=[0,0], dev=1, a=a, b=b, Qa=-S1, Qb=-A.T@S2@A):
    self.k = k
    self.dev = dev

    # Variables needed for the sigma class
    self.x0 = mu
    self.rot = np.eye(2)
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

"""
Analyzing the previous case, we propose a function that allows us to play much more with the 
generation of scalar fields.
  * k: norm factor.
  * dev: models the scale of the distribution while maintaining its properties.
"""
class sigma_fract:
  def __init__(self, k, mu=[0,0], dev=[1,1], a=a, b=b, Qa=-S1, Qb=-A.T@S2@A):
    self.k = k
    self.dev = dev

    # Variables necesarias para la clase sigma
    self.x0 = mu
    self.rot = np.eye(2)
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