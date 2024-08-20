"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Single integrator simulation plotter class -
"""

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

# Our utils
from .animation import Animation
from ..toolbox.plot_utils import vector2d
from ..toolbox.math_utils import step_repeat

AGENTS_COLOR = "royalblue"
DEAD_AGENTS_COLOR = "red"
AGENTS_RAD = 0.2
FIGSIZE = (12, 8)

# --------------------------------------------------------------------------------------
# Simulator class
# --------------------------------------------------------------------------------------


class SingIntPlotter:
    def __init__(
        self, 
        simulator, 
        xlim=[-50, 80], 
        ylim=[-50, 70], 
        agents_size=0.2, 
        agents_alpha=1, 
        agents_lw_alpha=0.5, 
        lw=0.3,
    ):
        matplotlib.rc("font", **{"size": 14})

        self.data = simulator.data
        self.sigma_field = simulator.sigma_field
        self.N = simulator.N
        self.dt = simulator.dt
        self.tf = simulator.t
        self.its = simulator.it

        self.xlim, self.ylim = xlim, ylim
        self.agts_size = agents_size
        self.agts_alpha = agents_alpha
        self.agts_lw_alpha = agents_lw_alpha
        self.lw = lw

        dx = abs(self.xlim[0] - self.xlim[1])
        dy = abs(self.ylim[0] - self.ylim[1])
        self.kw_field = {"xlim": dx, "ylim": dy, "n": 1000}
        self.kw_patch = {"size": dx / 130 * 3, "lw": 0.5}

        self.animation = Animation()

    # PLOTS ----------------------------------------------------------------------------

    def plot_simulation(self, dpi=100, figsize=FIGSIZE):
        # Extract the requiered data from the simulation
        data_p = np.array(self.data["p"])
        data_status = np.array(self.data["status"])
        data_pc_hat = np.array(self.data["pc_hat"])

        agents_colors = [
            "royalblue" if data_status[-1, n] else "red" for n in range(self.N)
        ]

        # Initialise the figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.subplots()

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect("equal")
        ax.grid(True)

        ax.set_xlabel("X [L]")
        ax.set_ylabel("Y [L]")

        # Draw the scalar field
        self.sigma_field.draw(fig, ax, **self.kw_field)

        # Plot the trace of each agent
        for n in range(self.N):
            ax.plot(
                data_p[:, n, 0],
                data_p[:, n, 1],
                c=agents_colors[n],
                zorder=1,
                alpha=0.5,
                lw=self.lw,
            )

        # Agents icon
        for n in range(self.N):
            icon_i = plt.Circle(
                (data_p[0, n, 0], data_p[0, n, 1]), self.agts_size, color=agents_colors[n]
            )
            icon_f = plt.Circle(
                (data_p[-1, n, 0], data_p[-1, n, 1]), self.agts_size, color=agents_colors[n]
            )

            icon_i.set_alpha(0.5)
            ax.add_patch(icon_i)
            ax.add_patch(icon_f)

        # Computed centroids
        # ax.plot(
        #     np.squeeze(data_pc_hat[:, :, 0]),
        #     np.squeeze(data_pc_hat[:, :, 1]),
        #     ".r",
        #     markersize=1,
        # )

        # Show the plot!!
        plt.show()

    def plot_paper_fig(
        self,
        dpi=100,
        figsize=(17, 8),
        obstacles=None,
        est_window=[],
        lw_data=0.8,
        alpha=0.8,
        text_shift=[-15, 7],
        xtick_step = 1,
    ):
        # Extract the requiered data from the simulation
        data_p = np.array(self.data["p"])
        data_status = np.array(self.data["status"])
        data_pc_hat_log = np.array(self.data["pc_hat_log"])
        data_mu_log = np.array(self.data["mu_log"])
        data_pc = np.array(self.data["pc_comp"])
        data_mu = np.array(self.data["mu_comp"])
        data_sigma = np.array(self.data["sigma"])

        mu = self.sigma_field.mu
        data_dist = np.linalg.norm(data_p - mu, axis=2)

        agents_colors = [
            "royalblue" if data_status[-1, n] else "red" for n in range(self.N)
        ]

        # Initialise the figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        grid = plt.GridSpec(3, 7, hspace=0.4, wspace=1)
        ax = fig.add_subplot(grid[:, 0:3])
        ax_data_pcx = fig.add_subplot(grid[0, 3:5], xticklabels=[])
        ax_data_pcy = fig.add_subplot(grid[1, 3:5], xticklabels=[])
        ax_data_mu = fig.add_subplot(grid[2, 3:5])
        ax_data_sigma = fig.add_subplot(grid[0, 5:7], xticklabels=[])
        ax_data_dist = fig.add_subplot(grid[1:3, 5:7])

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xlabel("X [L]")
        ax.set_ylabel("Y [L]")
        ax.set_aspect("equal")
        ax.grid(True)

        ax_data_pcx.set_ylabel(r"$p_i^X - \hat x_i^X$ [L]")
        ax_data_pcx.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax_data_pcx.yaxis.tick_right()
        ax_data_pcx.grid(True)
        ax_data_pcy.set_ylabel(r"$p_i^Y - \hat x_i^Y$ [L]")
        ax_data_pcy.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax_data_pcy.yaxis.tick_right()
        ax_data_pcy.grid(True)
        ax_data_dist.set_xlabel(r"$t$ [T]")
        ax_data_mu.set_ylabel(r"$\arctan(\mu_i^Y/\mu_i^X)$ [rad]")
        ax_data_mu.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax_data_mu.yaxis.tick_right()
        ax_data_mu.grid(True)

        ax_data_sigma.set_xlabel(r"")
        ax_data_sigma.set_ylabel(r"$\sigma(p_i)$ [u]")
        ax_data_sigma.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax_data_sigma.yaxis.tick_right()
        ax_data_sigma.grid(True)
        ax_data_dist.set_xlabel(r"$t$ [T]")
        ax_data_dist.set_ylabel(r"$\|p_\sigma - p_i\|$ [L]")
        ax_data_dist.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax_data_dist.yaxis.tick_right()
        ax_data_dist.grid(True)

        #############
        # MAIN axis
        #############
        # Draw the scalar field
        self.sigma_field.draw(fig, ax, **self.kw_field)

        # Plot the trace of each agent
        for n in range(self.N):
            ax.plot(
                data_p[:, n, 0],
                data_p[:, n, 1],
                c=agents_colors[n],
                zorder=1,
                alpha=self.agts_lw_alpha,
                lw=self.lw,
            )

        # Agents icon
        x = data_pc[0, 0] + text_shift[0]
        y = data_pc[0, 1] + text_shift[1]
        ax.text(x, y, "t = " + str(round(0, 2)))
        for n in range(self.N):
            icon_i = plt.Circle(
                (data_p[0, n, 0], data_p[0, n, 1]), self.agts_size, color=agents_colors[n]
            )
            icon_i.set_alpha(0.5)
            ax.add_patch(icon_i)

        t_list = [self.tf / 4, self.tf / 2, self.tf - self.dt]
        for t in t_list:
            it = int(t / self.dt)
            x = data_pc[it, 0] - 15
            y = data_pc[it, 1] + 7
            ax.text(x, y, "t = " + str(round(t, 2)))
            for n in range(self.N):
                icon = plt.Circle(
                    (data_p[it, n, 0], data_p[it, n, 1]),
                    self.agts_size,
                    color=agents_colors[n],
                )
                icon.set_alpha(self.agts_alpha)
                ax.add_patch(icon)

        # Obstacles
        if obstacles is not None:
            for obstacle in obstacles:
                x, y, r = obstacle
                obst_patch = plt.Circle((x, y), r, color="w")
                ax.add_patch(obst_patch)

        # Computed centroids
        # ax.plot(
        #     np.squeeze(data_pc_hat[:, :, 0]),
        #     np.squeeze(data_pc_hat[:, :, 1]),
        #     ".r",
        #     markersize=1,
        # )

        #############
        # DATA axis
        #############
        t1, t2 = est_window
        it1, it2 = int(np.floor(t1 / self.dt)), int(np.floor(t2 / self.dt))
        its = it2 - it1 + 1
        its_c, its_m = data_pc_hat_log.shape[1], data_mu_log.shape[1]
        time_vec = np.linspace(t1, t2, its)

        dt_c, dt_m = self.dt / (its_c - 1), self.dt / (its_m - 1)
        time_its_c_vec = np.arange(t1, t2 + dt_c, dt_c)[0:its_c*(its-1)-2]
        time_its_m_vec = np.arange(t1, t2 + dt_m, dt_m)[0:its_m*(its-1)-2]
        time_its_c_vec = step_repeat(time_its_c_vec, its_c)
        time_its_m_vec = step_repeat(time_its_m_vec, its_m)

        time_vec_str = ["{:.2f}".format(round(i,2)) for i in time_vec]
        xticks_lables = np.array([""]*len(time_vec), dtype="U16")
        xticks_lables[0::xtick_step] = time_vec_str[0::xtick_step]
    
        ax_data_pcx.set_xticks(time_vec)
        ax_data_pcy.set_xticks(time_vec)
        ax_data_mu.set_xticks(time_vec, xticks_lables)

        time_step_vec = np.repeat(time_vec, 2)[1:-1]
        data_pcx = np.repeat(data_pc[it1 : it2 + 1, 0], 2)[:-2]
        data_pcy = np.repeat(data_pc[it1 : it2 + 1, 1], 2)[:-2]
        data_mux = np.repeat(data_mu[it1 : it2 + 1, 0], 2)[:-2]
        data_muy = np.repeat(data_mu[it1 : it2 + 1, 1], 2)[:-2]
        data_mu_theta = np.arctan2(data_muy, data_mux)

        ax_data_pcx.plot(time_step_vec, data_pcx, "-k", lw=2, zorder=3)
        ax_data_pcy.plot(time_step_vec, data_pcy, "-k", lw=2, zorder=3)
        ax_data_mu.plot(time_step_vec, data_mu_theta, "-k", lw=2, zorder=3)

        agents_colors2 = [
            "royalblue" if data_status[it2, n] else "red" for n in range(self.N)
        ]

        for n in range(self.N):
            data_pcx = data_pc_hat_log[it1 : it2 + 1, :, n, 0][:-1]
            data_pcy = data_pc_hat_log[it1 : it2 + 1, :, n, 1][:-1]
            data_mux = data_mu_log[it1 : it2 + 1, :, n, 0][:-1]
            data_muy = data_mu_log[it1 : it2 + 1, :, n, 1][:-1]

            data_pcx = data_pcx.reshape(len(time_its_c_vec))
            data_pcy = data_pcy.reshape(len(time_its_c_vec))
            data_mux = data_mux.reshape(len(time_its_m_vec))
            data_muy = data_muy.reshape(len(time_its_m_vec))
            data_mu_theta = np.arctan2(data_muy, data_mux)

            kw_args = {"c": agents_colors2[n], "lw": lw_data, "alpha": alpha}
            ax_data_pcx.plot(time_its_c_vec, data_pcx, **kw_args)
            ax_data_pcy.plot(time_its_c_vec, data_pcy, **kw_args)
            ax_data_mu.plot(time_its_m_vec, data_mu_theta, **kw_args)

        time_vec = np.linspace(0, self.tf, self.its)

        ax_data_sigma.axhline(self.sigma_field.value(mu), c="k", ls="--", lw=2)
        ax_data_dist.axhline(0, c="k", ls="--", lw=2)

        for n in range(self.N):
            kw_args = {"c": agents_colors[n], "lw": lw_data, "alpha": alpha}
            ax_data_sigma.plot(time_vec, data_sigma[:, n, 0], **kw_args)
            ax_data_dist.plot(time_vec, data_dist[:, n], **kw_args)
        # Show the plot!!
        plt.show()

    def plot_estimations(self, figsize=(12, 4), dpi=100):
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
        agents_colors = [
            "royalblue" if data_status[-1, n] else "red" for n in range(self.N)
        ]

        # Extract the requiered data from the scalar field
        data_p_sigma = self.sigma_field.mu

        # Compute the angle of the ascending direction vectors
        psi_mu = np.arctan2(data_mu[:, :, 1], data_mu[:, :, 0])
        psi_mu_comp = np.arctan2(data_mu_comp[:, 1], data_mu_comp[:, 0])

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
        ax_ctr_y.plot(
            data_t,
            np.ones_like(data_t) * data_p_sigma[1],
            c="k",
            ls="--",
            lw=1.2,
            alpha=0.8,
        )
        ax_ctr_x.plot(
            data_t,
            np.ones_like(data_t) * data_p_sigma[0],
            c="k",
            ls="--",
            lw=1.2,
            alpha=0.8,
            label=r"$p_\sigma$",
        )

        ax_asc.plot(
            data_t,
            psi_mu_comp,
            c="r",
            ls="--",
            lw=1.2,
            zorder=3,
            label=r"$\theta^1_\sigma$",
        )

        ax_ctr_x.plot(
            data_t,
            data_pc_comp[:, 0],
            c="r",
            ls="--",
            lw=1.2,
            zorder=3,
            label=r"$p_c$",
        )

        ax_ctr_y.plot(data_t, data_pc_comp[:, 1], c="r", ls="--", lw=1.2, zorder=3)

        ax_asc.legend(fancybox=True, prop={"size": 15})

        ax_ctr_x.legend(fancybox=True, prop={"size": 15})

        # Plot the estimations
        for n in range(self.N):
            ax_asc.plot(
                data_t[data_status[:, n]],
                psi_mu[data_status[:, n], n],
                c=agents_colors[n],
                alpha=0.5,
                lw=1,
            )
            ax_ctr_x.plot(
                data_t[data_status[:, n]],
                data_pc_hat[data_status[:, n], n, 0],
                c=agents_colors[n],
                alpha=0.5,
                lw=1,
            )
            ax_ctr_y.plot(
                data_t[data_status[:, n]],
                data_pc_hat[data_status[:, n], n, 1],
                c=agents_colors[n],
                alpha=0.5,
                lw=1,
            )

        # Show the plot!!
        plt.show()

    # ANIMATIONS -----------------------------------------------------------------------

    def anim_simulation(
        self, anim_tf=None, tail_frames=100, fps=60, dpi=100, figsize=FIGSIZE
    ):
        """ """
        # Extract the requiered data from the simulation
        data_t = np.array(self.data["t"])
        data_p = np.array(self.data["p"])
        data_status = np.array(self.data["status"])

        agents_colors = [
            "royalblue" if data_status[0, n] else "red" for n in range(self.N)
        ]

        # Animation variables
        # res = RES_DIC[res_label]

        if anim_tf is None:
            anim_tf = data_t[-1]
        elif anim_tf > data_t[-1]:
            anim_tf = data_t[-1]

        anim_frames = self.its

        # Initialise the figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.subplots()

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect("equal")

        title = ax.set_title(
            "Frame = {0:>4} | Tf = {1:>5.2f} [T] | N = {2:>4} robots".format(
                0, 0, self.N
            )
        )
        ax.set_xlabel("X [L]")
        ax.set_ylabel("Y [L]")

        # Draw the scalar field
        self.sigma_field.draw(fig, ax, **self.kw_field)

        # Trace of each agent
        traces_list = []
        for n in range(self.N):
            (line,) = ax.plot(
                data_p[0, n, 0],
                data_p[0, n, 1],
                c=agents_colors[n],
                zorder=1,
                lw=self.lw,
            )
            traces_list.append(line)

        # Agents icon
        icons_list = []
        for n in range(self.N):
            icon = plt.Circle(
                (data_p[0, n, 0], data_p[0, n, 1]), self.agts_size, color=agents_colors[n]
            )
            ax.add_patch(icon)
            icons_list.append(icon)

        # -----------------------------------------------------
        # Update the information of the Animation class
        self.animation.set_data(self.N, data_p, data_status, self.dt)
        self.animation.set_graphics(
            ax=ax,
            title=title,
            traces=traces_list,
            icons=icons_list,
            tail_frames=tail_frames,
            size=self.agts_size,
        )

        # Generate the animation
        print("Generating {0:d} frames...".format(anim_frames))
        anim = FuncAnimation(
            fig,
            self.animation.animate,
            frames=tqdm(range(anim_frames), initial=1, position=0),
            interval=1 / fps * 1000,
        )
        anim.embed_limit = 40

        # Close plots and return the animation class to be compiled
        plt.close()
        return anim


# --------------------------------------------------------------------------------------
