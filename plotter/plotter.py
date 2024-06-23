"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Plotter class -
"""

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

from toolbox.plot_utils import vector2d, unicycle_patch

TEAM_COLORS = ["royalblue", "green"]
FIGSIZE = (12,8)

# ----------------------------------------------------------------------
# Simulator class
# ----------------------------------------------------------------------

class plotter:
    def __init__(self, simulator, team_tags=None, 
                 xlim=[-50,80], ylim=[-50,70], agents_size=3):
        matplotlib.rc('font', **{'size' : 14})

        self.data = simulator.data
        self.sigma_field = simulator.sigma_field
        self.N = simulator.N
        self.dt = simulator.dt

        self.team_tags = team_tags
        self.xlim, self.ylim = xlim, ylim
        self.size = agents_size

        dx = abs(self.xlim[0] - self.xlim[1])
        dy = abs(self.ylim[0] - self.ylim[1])
        self.kw_field = {"xlim":dx, "ylim":dy, "n":250}
        self.kw_patch = {"size":dx/130*3, "lw":0.5}

    # PLOTS ------------------------------------------------------------

    def plot_simulation(self, dpi=100):
        # Extract the requiered data from the simulation
        data_p = np.array(self.data["p"])
        data_phi = np.array(self.data["phi"])
        data_status = np.array(self.data["status"])

        if self.team_tags is None:
            agents_colors = ["royalblue" if data_status[-1,n] else "red" for n in range(self.N)]
        else:
            agents_colors = [TEAM_COLORS[self.team_tags[n]] if data_status[-1,n] else "red" for n in range(self.N)]



        # Initialise the figure
        fig = plt.figure(figsize=FIGSIZE, dpi=dpi)
        ax = fig.subplots()

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect("equal")

        ax.set_xlabel("X [L]")
        ax.set_ylabel("Y [L]")

        # Draw the scalar field
        self.sigma_field.draw(fig, ax, **self.kw_field)

        # Plot the trace of each agent
        for n in range(self.N):
            if self.team_tags is None:
                ax.plot(data_p[:,n,0], data_p[:,n,1], c="k", zorder=1, lw=0.8)
            else:
                ax.plot(data_p[:,n,0], data_p[:,n,1], c=agents_colors[n], zorder=1, lw=0.8)

        # Agents icon
        for n in range(self.N):
            icon_i = unicycle_patch(data_p[0,n,:], data_phi[0,n], "royalblue", **self.kw_patch)
            icon_f = unicycle_patch(data_p[-1,n,:], data_phi[-1,n], agents_colors[n], **self.kw_patch)
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
        data_status = np.array(self.data["status"])

        # Agents color
        if self.team_tags is None:
            agents_colors = ["royalblue" if data_status[-1,n] else "red" for n in range(self.N)]
        else:
            agents_colors = [TEAM_COLORS[self.team_tags[n]] if data_status[-1,n] else "red" for n in range(self.N)]

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

        # Plot the non-computed variables (for numerical validation)
        ax_ctr_y.plot(data_t, np.ones_like(data_t)*data_p_sigma[1], c="k", ls="--", lw=1.2, alpha=0.8)
        ax_ctr_x.plot(data_t, np.ones_like(data_t)*data_p_sigma[0], c="k", ls="--", lw=1.2, alpha=0.8, label=r"$p_\sigma$")

        if self.team_tags is None:
            ax_asc.plot(data_t, psi_mu_comp, c="r", ls="--", lw=1.2, zorder=3 , label=r"$\theta^1_\sigma$")
            ax_ctr_x.plot(data_t, data_pc_comp[:,0], c="r", ls="--", lw=1.2, zorder=3, label=r"$p_c$")
            ax_ctr_y.plot(data_t, data_pc_comp[:,1], c="r", ls="--", lw=1.2, zorder=3)

            ax_asc.legend(fancybox=True, prop={'size': 15})
            
        ax_ctr_x.legend(fancybox=True, prop={'size': 15})

        # Plot the estimations
        for n in range(self.N):
            ax_asc.plot(data_t[data_status[:,n]], psi_mu[data_status[:,n],n], c=agents_colors[n], alpha=0.5, lw=1)
            ax_ctr_x.plot(data_t[data_status[:,n]], data_pc_hat[data_status[:,n],n,0], c=agents_colors[n], alpha=0.5, lw=1)
            ax_ctr_y.plot(data_t[data_status[:,n]], data_pc_hat[data_status[:,n],n,1], c=agents_colors[n], alpha=0.5, lw=1)

        # Show the plot!!
        plt.show()

    # ANIMATIONS --------------------------------------------------------

    def anim_simulation(self, anim_tf=None, tail_frames=100, fps=60, dpi=100):
        """
        """
        # Extract the requiered data from the simulation
        data_t = np.array(self.data["t"])
        data_p = np.array(self.data["p"])
        data_phi = np.array(self.data["phi"])
        data_status = np.array(self.data["status"])
        
        agents_colors = [TEAM_COLORS[0] for n in range(self.N)]
        agents_colors_team = [TEAM_COLORS[self.team_tags[n]] if data_status[-1,n] else "red" for n in range(self.N)]
        
        # Animation variables
        # res = RES_DIC[res_label]

        if anim_tf is None:
            anim_tf = data_t[-1]
        elif anim_tf > data_t[-1]:
            anim_tf = data_t[-1]

        anim_frames = int(anim_tf/self.dt + 1)

        # Initialise the figure
        fig = plt.figure(figsize=FIGSIZE, dpi=dpi)
        ax = fig.subplots()

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect("equal")

        title = ax.set_title("Frame = {0:>4} | Tf = {1:>5.2f} [T] | N = {2:>4} robots".format(0, 0, self.N))
        ax.set_xlabel("X [L]")
        ax.set_ylabel("Y [L]")

        # Draw the scalar field
        self.sigma_field.draw(fig, ax, **self.kw_field)

        # Trace of each agent
        traces_list = []
        for n in range(self.N):
            line, = ax.plot(data_p[0,n,0], data_p[0,n,1], c=agents_colors[n], zorder=1, lw=0.8)
            traces_list.append(line)

        # Agents icon
        icons_list = []
        for n in range(self.N):
            icon = unicycle_patch(data_p[0,n,:], data_phi[0,n], agents_colors[n], **self.kw_patch)
            ax.add_patch(icon)
            icons_list.append(icon)

        # Function to update the animation
        def animate(i):
            # Change the colors when the first agent dies
            if np.any(np.logical_not(data_status[i,:])):
                colors = agents_colors_team
            else:
                colors = agents_colors

            # Agents
            for n in range(self.N):
                icons_list[n].remove()
                icons_list[n] = unicycle_patch(data_p[i,n,:], data_phi[i,n], colors[n], **self.kw_patch)
   
                ax.add_patch(icons_list[n])

                if i > tail_frames:
                    traces_list[n].set_data(data_p[i-tail_frames:i,n,0], data_p[i-tail_frames:i,n,1])
                else:
                    traces_list[n].set_data(data_p[0:i,n,0], data_p[0:i,n,1])
                traces_list[n].set_color(colors[n])

                title.set_text("Frame = {0:>4} | Tf = {1:>5.2f} [T] | N = {2:>3} robots".format(i, i*self.dt, self.N))
        
        # Generate the animation
        print("Generating {0:d} frames...".format(anim_frames))
        anim = FuncAnimation(fig, animate, frames=tqdm(range(anim_frames), initial=1, position=0), 
                            interval=1/fps*1000)
        anim.embed_limit = 40

        # Close plots and return the animation class to be compiled
        plt.close()
        return anim