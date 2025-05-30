"""
"""

__all__ = ["PlotterSimData2Uny"]

import numpy as np

# Graphic tools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs, load_class, first_larger_index
from ssl_simulator.visualization import config_axis, config_data_axis, unicycle_patch,  Plotter

from ssl_simulator.components.scalar_fields import ScalarField, PlotterScalarField

#######################################################################################

class PlotterSimData2Uny(Plotter):
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
        kw_patch = dict(size=2, fc="royalblue", ec="k", lw=0.5, zorder=4)
        kw_patch_dead = dict(fc="darkred", lw=0)
        
        self.kw_lines = dict(c="royalblue", lw=0.2, alpha=0.6, zorder=3)
        self.kw_line_mu = dict(lw=0.8, c="k", ls="--", zorder=2)
        
        # Update defaults with user-specified values
        self.kw_ax = parse_kwargs(kwargs, kw_ax)
        self.kw_field = parse_kwargs(kwargs, kw_field)
        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_patch_dead = parse_kwargs(kw_patch_dead, self.kw_patch)

        # Create subplots
        grid = plt.GridSpec(2, 2, hspace=0.05, wspace=0.05)
        self.ax1 = self.fig.add_subplot(grid[0, 0])
        self.ax2 = self.fig.add_subplot(grid[0, 1])
        self.ax3 = self.fig.add_subplot(grid[1, 0])
        self.ax4 = self.fig.add_subplot(grid[1, 1])
        # self.ax5 = self.fig.add_subplot(grid[1, 1])
        # self.ax6 = self.fig.add_subplot(grid[1, 2])
        self._create_cbar_axis()

        self.axs = [self.ax1, self.ax2, self.ax3, self.ax4]
        self.tiksx = [False, False, True, True]
        self.tiksy = [True, False, True, False]

    # ---------------------------------------------------------------------------------

    def _create_cbar_axis(self):
        # Draw figure and get proper positions
        self.fig.canvas.draw()

        # Get position of ax3 and ax6
        pos2 = self.ax2.get_position()
        pos4 = self.ax4.get_position()

        # Calculate bounds for colorbar axis (x0, y0, width, height)
        x0 = pos2.x1 + 0.01  # a bit to the right of ax3/ax6
        y0 = pos4.y0         # bottom of ax6
        height = pos2.y1 - pos4.y0  # full height from bottom of ax6 to top of ax3
        width = 0.015        # or any small value

        self.ax_cbar = self.fig.add_axes([x0, y0, width, height])

        # Hide axis decorations on ax_cbar (spines, ticks, labels)
        self.ax_cbar.tick_params(left=False, labelleft=False, right=False, labelright=False)
        for spine in self.ax_cbar.spines.values():
            spine.set_visible(False)
            
    def _config_axes(self):
        # self.ax1.set_ylabel(r"$Y$ [L]")
        self.ax3.set_ylabel(r"$Y$ [L]")
        self.ax3.set_xlabel(r"$X$ [L]")
        # self.ax5.set_xlabel(r"$X$ [L]")
        # self.ax6.set_xlabel(r"$X$ [L]")

        for i,ax in enumerate(self.axs):
            config_data_axis(ax, y_right=False, 
                             x_tick=self.tiksx[i], y_tick=self.tiksy[i], **self.kw_ax)
            ax.set_aspect("equal")

    def _get_settings(self):
        self.scalar_field = load_class(
            module_name = "ssl_simulator.components.scalar_fields", 
            class_name = self.settings["field_class"]["__class__"], 
            base_class=ScalarField,
            **self.settings["field_class"]["__params__"]
        )
        self.field_plotter = PlotterScalarField(self.scalar_field)

    def draw(self, num_patches=3, t_list=None):
        if num_patches:
            self.num_patches = num_patches
        self.t_list = t_list

        self.update()

    def update(self):

        # Clean previous plots
        for ax in self.axs:
            ax.clear()
        
        if self.t_list is None:
            raise ValueError("Error: t_list not set.")
        elif isinstance(self.t_list, (list, tuple)):
            if len(self.t_list) != 4:
                raise ValueError("Error: The length of t_list should be 4.")
        else:
            raise TypeError("Error: t_list must be a list or tuple.")
            

        self._config_axes()

        # ------------------------------------------------
        # Get scalar_field and title from settings
        self._get_settings()

        # Extract derired data
        time = self.data["time"][:]
        p = self.data["p"][:,:,:]
        theta = self.data["theta"][:,:]
        sigma_mu = self.data["scalar_field_mu"]
        
        # ------------------------------------------------
        # MAIN AXIS
        n_robots = p.shape[1]
        self.title = f"n={n_robots}"

        # Plot the robots
        if self.t_list is None:
            idx_list = np.linspace(0, p.shape[0]-1, self.num_patches, dtype=int)
        else:
            idx_list = [first_larger_index(time, x) for x in self.t_list]

        for i,idx in enumerate(idx_list):

            for j in range(n_robots):
                icon = unicycle_patch(
                    [p[idx,j,0], p[idx,j,1]], theta[idx,j], 
                    **self.kw_patch)
                self.axs[i].add_patch(icon)

            for k in range(i):
                for j in range(n_robots):
                    icon = unicycle_patch(
                        [p[idx_list[k],j,0], p[idx_list[k],j,1]], theta[idx_list[k],j], 
                        **self.kw_patch)
                    icon.set_alpha(0.15)
                    self.axs[i].add_patch(icon)

            self.axs[i].text(-120, 95, 
                f"t = {time[idx]:.0f} T", va='center', ha='left', fontsize=18)
            if i == 0:
                self.axs[i].text(65, 95, 
                    f"N = {n_robots:d}", va='center', ha='left', fontsize=18)

            self.axs[i].plot(p[:idx,:,0], p[:idx,:,1], **self.kw_lines)
            self.axs[i].plot(sigma_mu[:idx,0], sigma_mu[:idx,1], **self.kw_line_mu, 
                             label=r"$p_\sigma(t)$")
            
            if i == 0:
                self.axs[i].legend(fancybox=True, prop={"size": 14}, ncols=2, loc="lower right")

            # Plot the scalar field
            self.scalar_field.mu = sigma_mu[idx,:]
            if i != 3:
                self.field_plotter.draw(fig=self.fig, ax=self.axs[i], 
                                        cbar_sw=False, **self.kw_field)
            else:
                self.field_plotter.draw(fig=self.fig, ax=self.axs[i], 
                                        cbar_ax=self.ax_cbar, **self.kw_field)
                
        self._config_axes()
        
#######################################################################################