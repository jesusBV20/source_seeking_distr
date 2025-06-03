"""
"""

__all__ = ["PlotProp1"]

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from matplotlib.legend import Legend # legend artist
from matplotlib import ticker

# -----------------------------------------

from ssl_simulator.math import norm_2
from ssl_simulator.visualization import vector2d
from ssl_simulator.components.scalar_fields import ScalarField, PlotterScalarField

#######################################################################################

class PlotProp1:
    def __init__(self, P: np.ndarray, scalar_field: ScalarField):
        self.N = P.shape[0]
        self.P = P

        pc = np.sum(P, axis=0)/P.shape[0]
        self.X = P - pc
        self.D = np.max(np.linalg.norm(self.X, axis=1))

        self.scalar_field = scalar_field

        self.compute_lambda_min()
        self.data_dist = []
        self.delta_list = []
        self.data_ineq = []
        self.data_rate = []

    def compute_lambda_min(self):
        P = np.zeros((2,2))
        for i in range(self.N):
            P += self.X[i:i+1,:].T @ self.X[i:i+1,:]
        P /= self.N
        P_eig = np.linalg.eigvals(P)
        self.P_min = np.min(P_eig)

    def analyse_field(self, epsilons: list[float], debug=False):
        ep_min = epsilons[0]
        ep_max = epsilons[1]
    
        ps = np.linspace(self.scalar_field.mu[0] + ep_min, self.scalar_field.mu[0] + ep_max, 300)
        ps = [ps, ps*0 + self.scalar_field.mu[1]]

        grad_norms, hess_norms = [], []
        for x,y in zip(ps[0], ps[1]):
            grad = self.scalar_field.grad(np.array([x,y]))
            H = self.scalar_field.hessian(np.array([x,y]))
            grad_norms.append(np.linalg.norm(grad))
            hess_norms.append(norm_2(H))
        
        if debug:
            plt.plot(grad_norms)
            plt.show()
            plt.plot(hess_norms)
            plt.show()

        K_min = np.min(grad_norms)
        M = np.max(hess_norms)/2

        return K_min, M

    def compute_example(self, dist_range: list[float], delta_list: list[float], its=100):
        self.data_dist = np.linspace(dist_range[0], dist_range[1], its)
        self.delta_list = delta_list
        self.data_ineq = []
        self.data_rate = []

        for delta in delta_list:
            ineq_list = []
            rate_list = []
            for d in self.data_dist:
                K_min, M = self.analyse_field(epsilons = [d-delta/2, d+delta/2])
                
                ineq_list.append(self.P_min/self.D**2*K_min - M*self.D)
                rate_list.append(K_min/M)

            self.data_ineq.append(ineq_list)
            self.data_rate.append(rate_list)

        self.data_ineq = np.array(self.data_ineq).T
        self.data_rate = np.array(self.data_rate).T

    def plot(
            self, 
            dpi=100, 
            figsize=(14,8), 
            xlim=[-5,60], 
            ylim=[15,55],
            fontsize=16
            ):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        grid = plt.GridSpec(2, 4, hspace=0.2, wspace=1)
        ax = fig.add_subplot(grid[:, 0:2])
        ax_right1 = fig.add_subplot(grid[0, 2:4])
        ax_right2 = fig.add_subplot(grid[1, 2:4])

        thr = self.D**3/self.P_min

        # Axis configuration
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("X [L]")
        ax.set_ylabel("Y [L]")
        ax.set_aspect("equal")
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10 / 4))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(10 / 4))

        ax_right1.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax_right1.xaxis.set_minor_locator(ticker.MultipleLocator(10 / 4))
        ax_right1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax_right1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1 / 4))
        ax_right2.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax_right2.xaxis.set_minor_locator(ticker.MultipleLocator(10 / 4))
        ax_right2.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax_right2.yaxis.set_minor_locator(ticker.MultipleLocator(10 / 4))    

        # Right axis configuration
        ax_right1.set_ylabel(r"$h_{\mathcal{S}}(x)$")
        ax_right1.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        ax_right1.yaxis.tick_right()
        ax_right1.grid(True)
        ax_right2.set_xlabel(r"$||p_c - p_\sigma||$ [L]")
        ax_right2.set_ylabel(r"$K_\mathcal{S}^{\text{min}} \,/\,M_\mathcal{S}$")
        ax_right2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        ax_right2.yaxis.tick_right()
        ax_right2.grid(True)

        # -- Notation plot --

        # Draw the scalar field
        dx, dy = abs(xlim[0] - xlim[1]), abs(ylim[0] - ylim[1])
        kw_field = {"xlim": dx, "ylim": dy, "n": 1000, "contour_levels": 5}
        scalar_field_plotter = PlotterScalarField(self.scalar_field)
        scalar_field_plotter.draw(fig=fig, ax=ax, **kw_field)

        # Draw p_c and p_sigma
        pc = np.array([25, 0])

        ax.plot(pc[0], pc[1], "+k") 
        vector2d(ax, [0,0], pc-np.array([0, 0]), lw = 1, hw=0.6, hl=0.6, zorder=2)
        # ax.text(-2, 1, r"$p_\sigma$", fontsize=fontsize, zorder=3)
        ax.text(pc[0]/3, pc[1]+1, r"$p_c - p_\sigma$", fontsize=fontsize, zorder=3)

        # Draw the subset S
        delta = 4.8
        set_patch = plt.Circle(pc, delta, color="gray")
        set_patch.set_alpha(0.5)
        ax.add_patch(set_patch)

        h = - (delta + 1.5)
        vector2d(ax, pc + np.array([0,h]), np.array([-delta,0]), lw = 1, hw=0.45, hl=0.45, zorder=2)
        vector2d(ax, pc + np.array([0,h]), np.array([ delta,0]), lw = 1, hw=0.45, hl=0.45, zorder=2)
        ax.text(pc[0]-0.5, pc[1]+h-3, r"$\delta$", fontsize=fontsize, zorder=3)
        ax.text(pc[0]+1.8, pc[1]-2.1, r"$\mathcal{S}$", fontsize=fontsize, zorder=3)
        
        # Draw the agents
        ax.scatter(self.P[:,0]+pc[0], self.P[:,1]+pc[1], c="royalblue", alpha=1, lw=1, s=6, zorder=3)

        # -- Numerical simulation plots --
        ax_right1.axhline(0, color="k", ls="--", lw=1)
        ax_right2.axhline(thr, color="k", ls="--", lw=1)
        ax_right2.text(16, thr+5, r"$\frac{D^3}{\lambda_{\text{min}}\{P(x)\}}$", fontsize=fontsize+2, zorder=3)

        for i in range(len(self.delta_list)):
            delta = self.delta_list[i]
            ax_right1.plot(self.data_dist, self.data_ineq[:,i], label=r"$\delta$ = {:.1f}".format(delta))
            ax_right2.plot(self.data_dist, self.data_rate[:,i])
            
        ax_right1.legend(fancybox=True, prop={"size": 11})
        
        # -> Show the plot <-
        ax.grid(True)
        plt.show()

#######################################################################################