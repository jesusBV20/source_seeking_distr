"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Scalar field common class -
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize

# Graphic tools
import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Our utils
from ..toolbox.math_utils import Q_prod_xi, unit_vec
from ..toolbox.plot_utils import vector2d, alpha_cmap

MY_CMAP = alpha_cmap(plt.cm.jet, 0.3)

# --------------------------------------------------------------------------------------


class SigmaField(ABC):
    A = np.eye(2)

    # ------------------------------------------------------------------------
    # These operations have to be implemented in the subclass ################

    @property
    @abstractmethod
    def mu(self) -> np.ndarray:
        pass

    @abstractmethod
    def eval_value(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluation of the scalar field for a vector of values
        """
        pass

    @abstractmethod
    def eval_grad(self, X: np.ndarray) -> np.ndarray:
        """
        Gradient vector of the scalar field for a vector of values
        """
        pass

    # ------------------------------------------------------------------------

    def value(self, X: np.ndarray) -> np.ndarray:
        X = Q_prod_xi(self.A, X - self.mu) + self.mu
        return self.eval_value(X)

    def grad(self, X: np.ndarray) -> np.ndarray:
        X = Q_prod_xi(self.A, X - self.mu) + self.mu
        grad = self.eval_grad(X)
        return Q_prod_xi(self.A.T, grad)

    def find_max(self, x0: np.ndarray) -> np.ndarray:
        return minimize(lambda x: -self.value(np.array([x])), x0).x

    def draw(
        self,
        fig: matplotlib.figure.Figure = None,
        ax: plt.Axes = None,
        xlim: float = 30,
        ylim: float = 30,
        cmap: matplotlib.colors.ListedColormap = MY_CMAP,
        n: int = 256,
        contour_levels: int = 0,
        contour_lw: float = 0.3,
        cbar_sw: bool = True,
    ):
        """
        Scalar field drawing function
        """
        if fig == None:
            fig = plt.figure(figsize=(16, 9), dpi=100)
            ax = fig.subplots()
        elif ax == None:
            ax = fig.subplots()

        # Calculate
        x = np.linspace(self.mu[0] - xlim, self.mu[0] + xlim, n)
        y = np.linspace(self.mu[1] - ylim, self.mu[1] + ylim, n)
        X, Y = np.meshgrid(x, y)

        P = np.array([list(X.flatten()), list(Y.flatten())]).T
        Z = self.value(P).reshape(n, n)

        # Draw
        ax.plot(self.mu[0], self.mu[1], "+k")
        color_map = ax.pcolormesh(X, Y, Z, cmap=cmap)

        if cbar_sw:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)

            cbar = fig.colorbar(color_map, cax=cax)
            cbar.set_label(label="$\sigma$ [u]", labelpad=10)

        if contour_levels != 0:
            kwargs = {
                "colors": "k",
                "linewidths": contour_lw,
                "linestyles": "-",
                "alpha": 0.2,
            }
            contr_map = ax.contour(X, Y, Z, **kwargs)
            return color_map, contr_map
        else:
            return (color_map,)

    def draw_grad(
        self, x: np.ndarray, axis: plt.Axes, kw_arr: dict = None, ret_arr: bool = True
    ):
        """
        Function for drawing the gradient at a given point in space
        """
        # kw_arr: c="k", ls="-", lw = 0.7, hw=0.1, hl=0.2, s=1, alpha=1
        if isinstance(x, list):
            x = np.array(x)

        grad_x = self.grad(x)[0]
        grad_x_unit = unit_vec(grad_x)

        if kw_arr is not None:
            quiv = vector2d(axis, x, grad_x_unit, **kw_arr)
        else:
            quiv = vector2d(axis, x, grad_x_unit)

        if ret_arr:
            return quiv
        else:
            return grad_x_unit

    def draw_L1(self, pc: np.ndarray, P: np.ndarray):
        """
        Funtion for calculating and drawing L^1

        Attributes
        ----------
        pc: numpy array
            [x,y] position of the centroid
        P: numpy array
            (N x 2) matrix of agents position from the centroid
        """
        if isinstance(pc, list):
            pc = np.array(pc)

        N = P.shape[0]
        X = P - pc

        grad_pc = self.grad(pc)[0]
        l1_sigma_hat = (grad_pc[:, None].T @ X.T) @ X

        x_norms = np.zeros((N))
        for i in range(N):
            x_norms[i] = (X[i, :]) @ X[i, :].T
            D_sqr = np.max(x_norms)

        l1_sigma_hat = l1_sigma_hat / (N * D_sqr)
        return l1_sigma_hat.flatten()


# --------------------------------------------------------------------------------------
