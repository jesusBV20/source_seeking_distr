"""
"""

__all__ = ["PlotterSimDataSI"]

import numpy as np

# Graphic tools
import matplotlib.pyplot as plt

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs, load_class, first_larger_index
from ssl_simulator.math import unit_vec
from ssl_simulator.visualization import config_axis, config_data_axis, Plotter

from ssl_simulator.components.scalar_fields import ScalarField, PlotterScalarField

#######################################################################################

class PlotterSimDataSI(Plotter):
    def __init__(self, data, settings, num_patches=2, **kwargs):
        super().__init__(**kwargs)

        self.data = data
        self.settings = settings

        self.scalar_field = None
        self.field_plotter = None
        self.title = None
        self.num_patches = num_patches

        # Default visual properties
        kw_ax = dict(xlims=None, ylims=None)
        kw_field = dict(contour_levels=8, n=1000)
        kw_patch = dict(s=4, fc="royalblue", ec="k", lw=0.5, zorder=3)
        
        self.kw_lines = dict(c="royalblue", lw=0.2, zorder=2)
        self.kw_lines_data = dict(lw=1, alpha=0.3, c="royalblue", zorder=3)
        self.kw_lines_data_zero  = dict(lw=1, c="k", ls="-", zorder=2)
        self.kw_lines_datax = dict(lw=1, alpha=0.4, c="skyblue", zorder=3)
        self.kw_lines_datax2 = dict(lw=1, c="royalblue", ls="--", zorder=4)
        self.kw_lines_datay = dict(lw=1, alpha=0.4, c="lightcoral", zorder=3)
        self.kw_lines_datay2 = dict(lw=1, c="darkred", ls="--", zorder=4)
        
        # Update defaults with user-specified values
        self.kw_ax = parse_kwargs(kwargs, kw_ax)
        self.kw_field = parse_kwargs(kwargs, kw_field)
        self.kw_patch = parse_kwargs(kwargs, kw_patch)

        # Create subplots
        grid = plt.GridSpec(3, 9, hspace=0.1, wspace=1.5)
        self.ax = self.fig.add_subplot(grid[:, 0:3])
        self.ax_data_pc    = self.fig.add_subplot(grid[0:2, 3:5])
        self.ax_data_epc   = self.fig.add_subplot(grid[2, 3:5])
        self.ax_data_muc   = self.fig.add_subplot(grid[0:2, 5:7])
        self.ax_data_emuc  = self.fig.add_subplot(grid[2, 5:7])

        self.ax_data_ex     = self.fig.add_subplot(grid[0, 7:9])
        self.ax_data_sigma = self.fig.add_subplot(grid[1, 7:9])
        self.ax_data_dist  = self.fig.add_subplot(grid[2, 7:9])

        self.axs_data = [self.ax_data_pc, self.ax_data_epc, 
                         self.ax_data_muc, self.ax_data_emuc,
                         self.ax_data_ex, self.ax_data_sigma, self.ax_data_dist]
        
        self.axs_xtiks = [False, True, False, True, False, False, True]
        self.axs_tiksy = [6, 5, 6, 5, 5, 5, 5]

    # ---------------------------------------------------------------------------------

    def _config_axes(self):
        self.ax.set_title(self.title)
        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$ [L]")
        self.ax.set_aspect("equal")

        self.ax_data_epc.set_xlabel(r"$t$ [T]")
        self.ax_data_epc.set_ylabel(r"$\frac{||p_c^i(t) - p_c(t)||}{||p_c||}$")
        self.ax_data_emuc.set_ylabel(r"$\frac{||\mu_c^i(t) - \mu_c(t)||}{||\mu_c(t)||}$")
        self.ax_data_ex.set_ylabel(r"$\frac{||x_i(t) - x_d^i||}{||x_d^i||}$")
        self.ax_data_pc.set_ylabel(r"$p_c^i(t)$ [L]") # p_i - \hat{x}_i
        self.ax_data_muc.set_ylabel(r"$\frac{\mu_c^i(t)}{||\mu_c^i(t)||}$")
        self.ax_data_sigma.set_ylabel(r"$\sigma(p_i(t))$ [u]")
        self.ax_data_dist.set_ylabel(r"$||p_c(t) - p_\sigma||$ [L]")

        config_axis(self.ax, **self.kw_ax)

        for i,ax in enumerate(self.axs_data):
            config_data_axis(ax, y_right=True, x_tick=self.axs_xtiks[i], 
                             max_major_ticks_y=self.axs_tiksy[i])

    def _get_settings(self):
        self.scalar_field = load_class(
            module_name = "ssl_simulator.components.scalar_fields", 
            class_name = self.settings["field_class"]["__class__"], 
            base_class=ScalarField,
            **self.settings["field_class"]["__params__"]
        )
        self.field_plotter = PlotterScalarField(self.scalar_field)
        
        n  = self.settings["graph"]["__params__"]["N"]
        ep_x, ep_mu, kf = self.settings["ep_x"], self.settings["ep_mu"], self.settings["k_form"]
        self.title = f"N={n}, $\epsilon_x=${ep_x}, $\epsilon_\mu=${ep_mu}, $k_f=${kf}"

    def draw(self, num_patches=None, text_offsets=None, t_list=None, **kwargs):
        if num_patches:
            self.num_patches = num_patches
        self.text_offsets = text_offsets
        self.t_list = t_list

        self.update()

    def update(self):

        # Clean previous plots
        self.ax.clear()
        for ax in self.axs_data:
            ax.clear()
        
        self._config_axes()

        # ------------------------------------------------
        # Get scalar_field and title from settings
        self._get_settings()

        # Extract derired data
        time = self.data["time"]
        p = np.array(self.data["p"].tolist())[:,:,:]
        sigma = np.array(self.data["sigma"].tolist())[:,:,0]

        pc_centralized  = self.data["pc_centralized"][:,:]
        muc_centralized = self.data["muc_centralized"][:,:]
        pc_estimated    = self.data["pc_estimated"][:,:,:]
        muc_estimated   = self.data["muc_estimated"][:,:,:]

        status = np.array(self.data["status"], dtype=bool)
        status_not = np.logical_not(status)

        if self.settings is not None:
            xstar = self.settings["p_star"][:,:] - self.data["pc_centralized"][0,:]

        # ------------------------------------------------
        # MAIN AXIS

        # Plot the robots
        if self.t_list is None:
            idx_list = np.linspace(0, p.shape[0]-1, self.num_patches, dtype=int)
        else:
            idx_list = [first_larger_index(time, x) for x in self.t_list]
            
        for i,idx in enumerate(idx_list):
            if idx:
                self.ax.scatter(p[idx,:,0], p[idx,:,1], **self.kw_patch)
                if self.text_offsets is not None:
                    self.ax.text(pc_centralized[idx,0] + self.text_offsets[i][0], 
                                pc_centralized[idx,1] + self.text_offsets[i][1], 
                                f"t = {time[idx]:.2f} T", va='center', ha='center', fontsize=14)
                
        self.ax.plot(p[:,:,0], p[:,:,1], **self.kw_lines)

        # Plot the scalar field
        self.field_plotter.draw(fig=self.fig, ax=self.ax, **self.kw_field)
        self.field_plotter.update()
        
        # ------------------------------------------------
        # DATA AXES

        # -- AXIS 1
        pc_error = np.linalg.norm((pc_centralized[:,None,:] - pc_estimated), axis=2) / np.linalg.norm(pc_centralized[:,None,:], axis=2)
        self.ax_data_epc.plot(time, pc_error, **self.kw_lines_data)
        self.ax_data_epc.axhline(0, **self.kw_lines_data_zero)
        
        # -- AXIS 2
        mu_error = np.linalg.norm((muc_centralized[:,None,:] - muc_estimated), axis=2) / np.linalg.norm(muc_centralized[:,None,:], axis=2)
        self.ax_data_emuc.plot(time, mu_error, **self.kw_lines_data)
        self.ax_data_emuc.axhline(0, **self.kw_lines_data_zero)

        # -- AXIS 3
        if self.settings is not None:
            x_centralized = p - pc_centralized[:,None,:]
            p_star_error = np.linalg.norm((xstar[None,:] - x_centralized), axis=2) / np.linalg.norm(xstar[None,:], axis=2)
            self.ax_data_ex.plot(time, p_star_error, **self.kw_lines_data)
        self.ax_data_ex.axhline(0, **self.kw_lines_data_zero)

        # -- AXIS 4
        self.ax_data_pc.plot(time, pc_estimated[:,:,0], **self.kw_lines_datax)
        self.ax_data_pc.plot(time, pc_estimated[:,:,1], **self.kw_lines_datay)
        self.ax_data_pc.plot(time, pc_centralized[:,0], **self.kw_lines_datax2, label=r"actual X")
        self.ax_data_pc.plot(time, pc_centralized[:,1], **self.kw_lines_datay2, label=r"actual Y")
        self.ax_data_pc.axhline(0, **self.kw_lines_data_zero)

        self.ax_data_pc.plot(0, 0, **parse_kwargs({"alpha":1}, self.kw_lines_datax), label=r"distrb. X")
        self.ax_data_pc.plot(0, 0, **parse_kwargs({"alpha":1}, self.kw_lines_datay), label=r"distrb. Y")
        self.ax_data_pc.legend(fancybox=True, prop={"size": 11}, ncols=2, loc="upper left")

        # -- AXIS 5
        muc_estimated_unit = unit_vec(muc_estimated, axis=2)
        muc_centralized_unit = unit_vec(muc_centralized, axis=1)
        self.ax_data_muc.plot(time, muc_estimated_unit[:,:,0], **self.kw_lines_datax)
        self.ax_data_muc.plot(time, muc_estimated_unit[:,:,1], **self.kw_lines_datay)
        self.ax_data_muc.plot(time, muc_centralized_unit[:,0], **self.kw_lines_datax2, label=r"actual X")
        self.ax_data_muc.plot(time, muc_centralized_unit[:,1], **self.kw_lines_datay2, label=r"actual Y")
        self.ax_data_muc.axhline(0, **self.kw_lines_data_zero)
        self.ax_data_muc.plot(0, 0, **parse_kwargs({"alpha":1}, self.kw_lines_datax), label=r"distr X")
        self.ax_data_muc.plot(0, 0, **parse_kwargs({"alpha":1}, self.kw_lines_datay), label=r"distr Y")
        # self.ax_data_muc.legend(fancybox=True, prop={"size": 12}, ncols=2, loc="upper left")

        # -- AXIS 6
        self.ax_data_sigma.plot(time, sigma, **self.kw_lines_data)
        self.ax_data_sigma.axhline(self.scalar_field.value(self.scalar_field.mu), c="k", ls="--", lw=1.5)
        self.ax_data_sigma.text(0, self.scalar_field.value(self.scalar_field.mu)*0.93, 
                                r"$\sigma(p_\sigma)$", va='center', ha='left', fontsize=16)
        
        # -- AXIS 7
        dist = pc_centralized - self.scalar_field.mu[None,:]
        dist = np.linalg.norm(dist, axis = 1)
        self.ax_data_dist.plot(time, dist, lw=1.5, c="k")
        self.ax_data_dist.axhline(0, **self.kw_lines_data_zero)

        self._config_axes()
        ylims = self.ax_data_pc.get_ylim()
        self.ax_data_pc.set_ylim(ylims[0], ylims[1]*1.5)
        
#######################################################################################