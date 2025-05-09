"""
"""

__all__ = ["PlotProp2"]

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.optimize import brentq

from matplotlib.legend import Legend # legend artist
from matplotlib import ticker

# -----------------------------------------

from ssl_simulator.math import L_sigma
from ssl_simulator.visualization import vector2d
from ssl_simulator.components.scalar_fields import ScalarField, SigmaGauss, PlotterScalarField

# -----------------------------------------

#######################################################################################

def calculate_a(N, alpha):
    """
    Funcion to distribute the agents following a geometric series
    """

    a = np.zeros(N)
    a[0] = 1

    for i in range(1,int(N/2)-1):
        a[i] = a[i-1]*alpha

    a[int(N/2)-1] = -(np.sum(a) - 1 - N/4)
    a = np.sqrt(a)

    j = 1
    for i in range(int(N/2), N):
        a[i] = -a[int(N/2)-j]
        j = j + 1

    return a

def find_alpha(n, d, k=1):
    # Function to solve
    def f(alpha):
        if alpha == 1.0:
            return n - d  # special case: sum becomes a * n
        return (1 - alpha**(k*n)) / (1 - alpha**k) - d

    # Find alpha in (0, 1]
    return brentq(f, 1e-10, 1.0)  # use a small epsilon to avoid div-by-zero

class PlotProp2:
    def __init__(self, p0, nx, ny, lx, ly, scalar_field=None):
        self.nx, self.ny = nx, ny

        # Generate the formation
        # a_x1d = calculate_a(nx, a1)
        # a_y1d = calculate_a(ny, a2)
        nx2, ny2 = int(nx/2), int(ny/2)
        alpha_x = find_alpha(nx2, nx/4 * 1 + 1, 2)
        alpha_y = find_alpha(ny2, ny/4 * 1 + 1, 2)

        ax = lx * alpha_x**np.arange(nx2)
        ay = ly * alpha_y**np.arange(ny2)

        Px = ax.reshape(1, np.size(ax))
        Py = ax.reshape(1, np.size(ay))

        print("X", np.max(ax**2), np.sum(ax**2) - lx**2, np.max(ax**2)*nx/4, lx**2*nx/4)

        Px = np.hstack((Px, -Px))
        Px = np.vstack((Px, np.zeros((1, np.size(Px))) ))
        Py = np.hstack((Py, -Py))
        Py = np.vstack((np.zeros((1, np.size(Py))), Py))

        self.P = np.hstack((Px,Py)).T + p0

        # Generate the scalar field
        if scalar_field is None:
            self.scalar_field = SigmaGauss(mu=[15,20], max_intensity=100, dev=15)
        else:
            self.scalar_field = scalar_field

    def plot(self, fig, ax, n_rm=0, title_sw=True, cbar_sw=True, legend_sw=True):
        nx, ny = self.nx, self.ny
        scalar_field = self.scalar_field
        
        P = np.copy(self.P[np.random.choice(range(self.P.shape[0]), self.P.shape[0]-n_rm, replace=False), :])
        pc = np.mean(P, axis=0)

        # Compute the measured sigma values, L^1 and L
        sigma_values = scalar_field.value(P)

        l_sigma = L_sigma(P - pc, sigma_values)
        l1_vec = scalar_field.L1(pc, P)

        # l_sigma = l_sigma/np.sqrt(l_sigma[0]**2 + l_sigma[1]**2)
        # l1_vec = l1_vec/np.sqrt(l1_vec[0]**2 + l1_vec[1]**2)

        # ----------------------------------------------------------------------
        # PLOT
        # ----------------------------------------------------------------------

        if title_sw:
            title = r"$N_\parallel$ = {0:d}, $N_-$ = {1:d}".format(nx,ny)
            ax.set_title(title, pad=15)

        # Draw the scalar field

        scalar_field_plotter = PlotterScalarField(scalar_field)
        scalar_field_plotter.draw(fig=fig, ax=ax, xlim=60, ylim=40, n=300, contour_levels=5, cbar_sw=cbar_sw)

        # Draw the agents
        for n in range(nx + ny - n_rm):
            ax.add_patch(plt.Circle(P[n], 0.2, color="royalblue", alpha=0.8, lw=0, zorder=3))

        # Draw the gradient at pc, L^1 and L
        kw = {"lw": 2, "hw": 0.4, "hl": 0.7}
        scalar_field_plotter.draw_grad(pc, ax, fct=200, zorder=3, **kw)
        vector2d(ax, pc, l_sigma*400, c="red"  , zorder=4, **kw)
        vector2d(ax, pc, l1_vec*400, c="green", zorder=4, **kw)

        # Generate the legend
        if legend_sw:
            arr1 = plt.scatter([],[],c='k'  ,marker=r'$\uparrow$',s=100)
            arr2 = plt.scatter([],[],c='red',marker=r'$\uparrow$',s=100)
            arr3 = plt.scatter([],[],c='green',marker=r'$\uparrow$',s=100)

            leg = Legend(ax, [arr1, arr2, arr3], 
                        [r"$\nabla \sigma (p_c)$ (Non-computed)",
                        r"$L_{\sigma}$: Actual computed ascending direction",
                        r"$L_\sigma^1$ (Non-computed)"],
                        loc="upper left", prop={'size': 9})
            ax.add_artist(leg)

#######################################################################################