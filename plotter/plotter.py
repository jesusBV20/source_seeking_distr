"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Plotter class -
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from toolbox.plot_utils import vector2d, unicycle_patch

# ----------------------------------------------------------------------
# Simulator class
# ----------------------------------------------------------------------

class plotter:
    """
    This class...
    """
    def __init__(self, simulator):
        matplotlib.rc('font', **{'size' : 14})

        self.data = simulator.data
        self.sigma_field = simulator.sigma_field
        self.N = simulator.N
    
    def plot_simulation(self, dpi=100, xlim=[-5,80], ylim=[-5,80]):
        # Extract the requiered data from the simulation
        data_p = np.array(self.data["p"])
        data_phi = np.array(self.data["phi"])

        # Initialise the figure
        fig = plt.figure()
        ax = fig.subplots()

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Draw the scalar field
        kw_draw_field = {"xlim":2*50, "ylim":2*50, "n":400, "contour_levels":20}
        self.sigma_field.draw(fig, ax, **kw_draw_field)

        # Plot the trace of each agent
        for n in range(self.N):
            ax.plot(data_p[:,n,0], data_p[:,n,1], c="k", zorder=1, lw=0.8)

        # Agents icon
        for n in range(self.N):
            icon_i = unicycle_patch(data_p[0,n,:], data_phi[0,n], "royalblue", size=2)
            icon_f = unicycle_patch(data_p[-1,n,:], data_phi[-1,n], "royalblue", size=2)
            icon_i.set_alpha(0.5)
            ax.add_patch(icon_i)
            ax.add_patch(icon_f)

        # Show the plot!!
        plt.show()

    def plot_estimations(self, figsize=(12,4), dpi=100):
        """
        Plot the estimation of the centroid and the ascending direction during the whole simulation
        """
        # Extract the requiered data from the simulation
        data_t = np.array(self.data["t"])
        data_pc_hat = np.array(self.data["pc_hat"])
        data_mu = np.array(self.data["mu"])
        data_pc_comp = np.array(self.data["pc_comp"])
        data_mu_comp = np.array(self.data["mu_comp"])

        # Extract the requiered data from the scalar field
        data_p_sigma = self.sigma_field.mu

        # Compute the angle of the ascending direction vectors
        psi_mu = np.arctan2(data_mu[:,:,1], data_mu[:,:,0])
        psi_mu_comp = np.arctan2(data_mu_comp[:,1], data_mu_comp[:,0])

        # Initialise the figure and its axes
        fig = plt.figure(figsize=figsize, dpi=dpi)
        grid = plt.GridSpec(2, 2, hspace=0.1, wspace=0.25)
        ax_asc = fig.add_subplot(grid[:, 0:1])
        ax_ctr_x = fig.add_subplot(grid[0, 1:2], xticklabels=[])
        ax_ctr_y = fig.add_subplot(grid[1, 1:2])

        ax_asc.grid(True)
        ax_ctr_x.grid(True)
        ax_ctr_y.grid(True)

        ax_asc.set_xlabel(r"$t$ [T]")
        ax_ctr_y.set_xlabel(r"$t$ [T]")
        ax_asc.set_ylabel(r"$\theta_\mu$ [rad]")
        ax_ctr_x.set_ylabel(r"$(p_i - \hat x_i)^X$ [L]")
        ax_ctr_y.set_ylabel(r"$(p_i - \hat x_i)^Y$ [L]")

        # Plot the ascending directions estimations
        ax_asc.plot(data_t, psi_mu, c="k", alpha=0.7, lw=1)
        ax_asc.plot(data_t, psi_mu_comp, c="r", ls="--", label=r"$\theta^1_\sigma$")

        ax_asc.legend(fancybox=True, prop={'size': 15})

        # Plot the centroid estimations
        ax_ctr_x.plot(data_t,data_pc_hat[:,:,0], c="k", alpha=0.7, lw=1)
        ax_ctr_x.plot(data_t,data_pc_comp[:,0], c="r", ls="--", label=r"$p_c$")
        ax_ctr_x.plot(data_t, np.ones_like(data_t)*data_p_sigma[0], c="b", ls="--", lw=1.2, label=r"$p_\sigma$")

        ax_ctr_y.plot(data_t,data_pc_hat[:,:,1], c="k", alpha=0.7, lw=1)
        ax_ctr_y.plot(data_t,data_pc_comp[:,1], c="r", ls="--")
        ax_ctr_y.plot(data_t, np.ones_like(data_t)*data_p_sigma[1], c="b", ls="--", lw=1.2)

        ax_ctr_x.legend(fancybox=True, prop={'size': 15})

        # Show the plot!!
        plt.show()