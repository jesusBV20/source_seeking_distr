"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Scalar field utilities -
"""

import numpy as np

# ----------------------------------------------------------------------
# Mathematical utility functions
# ----------------------------------------------------------------------

def Q_prod_xi(Q,X):
    """
    Matrix Q dot each row of X
    """
    return (Q @ X.T).T

def exp(X,Q,mu):
    """
    Exponential function with quadratic form: 
                            exp(r) = e^((r - mu)^t @ Q @ (r - mu))
    """
    return np.exp(np.sum((X - mu) * Q_prod_xi(Q,X - mu), axis=1))