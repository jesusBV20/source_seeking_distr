"""
"""

__all__ = ["PlotterSimData2SI"]

import numpy as np

# Graphic tools
import matplotlib.pyplot as plt

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs, load_class, first_larger_index
from ssl_simulator.visualization import config_axis, config_data_axis, Plotter

from ssl_simulator.components.scalar_fields import ScalarField, PlotterScalarField

#######################################################################################

class PlotterSimData2SI(Plotter):
    def __init__(self, data, settings, num_patches=2, t_list=None, **kwargs):
        super().__init__(**kwargs)

        self.data = data
        self.settings = settings

        self.scalar_field = None
        self.field_plotter = None
        self.title = None
        self.num_patches = num_patches
        self.t_list = t_list

        # Default visual properties
        kw_ax = dict(xlims=None, ylims=None)
        kw_field = dict(contour_levels=8, n=1000)
        kw_patch = dict(s=4, fc="royalblue", ec="k", lw=0.5, zorder=3)
        kw_patch_dead = dict(fc="darkred", lw=0)
        
        self.kw_lines = dict(c="royalblue", lw=0.2, zorder=2)
        self.kw_lines_dead = parse_kwargs(dict(c="darkred"), self.kw_lines)
        self.kw_lines_data = dict(lw=1, alpha=0.3, c="royalblue", zorder=3)
        self.kw_lines_data_dead = parse_kwargs(dict(c="darkred"), self.kw_lines_data)
        self.kw_lines_data_zero  = dict(lw=1, c="k", ls="-", zorder=2)
        
        # Update defaults with user-specified values
        self.kw_ax = parse_kwargs(kwargs, kw_ax)
        self.kw_field = parse_kwargs(kwargs, kw_field)
        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_patch_dead = parse_kwargs(kw_patch_dead, self.kw_patch)

        # Create subplots
        grid = plt.GridSpec(2, 3, hspace=0.1, wspace=1)
        self.ax = self.fig.add_subplot(grid[:, 0:3])
        self.ax_data_sigma = self.fig.add_subplot(grid[0, 2])
        self.ax_data_dist  = self.fig.add_subplot(grid[1, 2])

        self.axs_data = [self.ax_data_sigma, self.ax_data_dist]
        self.axs_xtiks = [False, True]
        self.axs_tiksy = [5, 6]

    # ---------------------------------------------------------------------------------

    def _config_axes(self):
        self.ax.set_title(self.title)
        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$ [L]")
        self.ax_data_dist.set_xlabel(r"$t$ [T]")
        self.ax.set_aspect("equal")

        self.ax_data_sigma.set_ylabel(r"$\sigma(p_i(t))$ [u]")
        self.ax_data_dist.set_ylabel(r"$||p_c(t) - p_\sigma||$ [L]")

        config_axis(self.ax, **self.kw_ax)

        for i,ax in enumerate(self.axs_data):
            config_data_axis(ax, y_right=True, x_tick=self.axs_xtiks[i], 
                             max_major_ticks_x=3, max_major_ticks_y=self.axs_tiksy[i])

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

    def draw(self, num_patches=None, text_offsets=None, t_list=None):
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
        time = self.data["time"][:]
        p = self.data["p"][:,:,:]
        sigma = self.data["sigma"][:,:,0]
        sigma_mu = self.data["sigma"][:,:,0]
        pc_centralized  = self.data["pc_centralized"][:,:]
        

        status = np.array(self.data["status"], dtype=bool)
        status_not = np.logical_not(status)

        # ------------------------------------------------
        # MAIN AXIS

        # Plot the robots
        if self.t_list is None:
            idx_list = np.linspace(0, p.shape[0]-1, self.num_patches, dtype=int)
        else:
            idx_list = [first_larger_index(time, x) for x in self.t_list]

        for i,idx in enumerate(idx_list):
            self.ax.scatter(p[idx,status[idx,:],0], p[idx,status[idx,:],1], **self.kw_patch)
            self.ax.scatter(p[idx,status_not[idx,:],0], p[idx,status_not[idx,:],1], 
                            **self.kw_patch_dead)
            if self.text_offsets:
                self.ax.text(pc_centralized[idx,0] + self.text_offsets[i][0], 
                            pc_centralized[idx,1] + self.text_offsets[i][1], 
                            f"t = {time[idx]:.0f} T", va='center', ha='center', fontsize=14)
                
        self.ax.plot(p[:,status[-1,:],0], p[:,status[-1,:],1], **self.kw_lines)
        self.ax.plot(p[:,status_not[-1,:],0], p[:,status_not[-1,:],1], **self.kw_lines_dead)

        # Plot the scalar field
        self.field_plotter.draw(fig=self.fig, ax=self.ax, **self.kw_field)
        
        # ------------------------------------------------
        # DATA AXES

        # -- AXIS 1
        self.ax_data_sigma.plot(time, sigma[:,status[-1,:]], **self.kw_lines_data)
        self.ax_data_sigma.plot(time, sigma[:,status_not[-1,:]], **self.kw_lines_data_dead)
        self.ax_data_sigma.axhline(self.scalar_field.value(self.scalar_field.mu), c="k", ls="--", lw=1.5)
        self.ax_data_sigma.text(0, self.scalar_field.value(self.scalar_field.mu)*0.94, 
                                        r"$\sigma(p_\sigma)$", va='center', ha='left', fontsize=14)

        # -- AXIS 2
        dist = pc_centralized - self.scalar_field.mu[None,:]
        dist = np.linalg.norm(dist, axis = 1)
        self.ax_data_dist.plot(time, dist, lw=1.5, c="k")
        self.ax_data_dist.axhline(0, **self.kw_lines_data_zero)

        self._config_axes()
        
#######################################################################################