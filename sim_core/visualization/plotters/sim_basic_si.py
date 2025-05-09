"""
"""

__all__ = ["PlotterSimBasicSI"]

import numpy as np
from collections.abc import Iterable

# Graphic tools
import matplotlib.pyplot as plt

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs, load_class
from ssl_simulator.visualization import config_axis, Plotter

from ssl_simulator.components.scalar_fields import ScalarField, PlotterScalarField

#######################################################################################

class PlotterSimBasicSI(Plotter):
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
        kw_patch_dead = dict(fc="darkred", linewidths=0)

        # Update defaults with user-specified values
        self.kw_ax = parse_kwargs(kwargs, kw_ax)
        self.kw_field = parse_kwargs(kwargs, kw_field)
        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_patch_dead = parse_kwargs(kw_patch_dead, self.kw_patch)
        self.kw_lines = dict(c="royalblue", lw=0.2, zorder=2)
        self.kw_lines_dead = parse_kwargs(dict(c="darkred"), self.kw_lines)

        # Create subplots
        self.ax = self.fig.subplots()
        self._config_axes()

    # ---------------------------------------------------------------------------------

    def _config_axes(self):
        self.ax.set_title(self.title)
        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$ [L]")
        self.ax.set_aspect("equal")
        config_axis(self.ax, **self.kw_ax)

    def _get_settings(self):
        self.scalar_field = load_class(
            module_name = "ssl_simulator.components.scalar_fields", 
            class_name = self.settings["field_class"]["__class__"], 
            base_class=ScalarField,
            **self.settings["field_class"]["__params__"]
        )
        self.field_plotter = PlotterScalarField(self.scalar_field)
        
        n  = self.settings["graph"]["__params__"]["N"]
        ep_x, ep_mu = self.settings["ep_x"], self.settings["ep_mu"]
        self.title = f"n={n}, $\epsilon_x$={ep_x}, $\epsilon_\mu$={ep_mu}"


    def draw(self, num_patches=2, **kwargs):
        
        # ------------------------------------------------
        # Get scalar_field and title from settings
        self._get_settings()
        
        # Extract derired data
        x = np.array(self.data["p"].tolist())[1:,:,0]
        y = np.array(self.data["p"].tolist())[1:,:,1]

        status = np.array(self.data["status"].tolist())
        status_not = np.logical_not(status)

        # ------------------------------------------------
        # Plot the robots
        idx_list = np.linspace(0, x.shape[0]-1, num_patches, dtype=int)

        for idx in idx_list:
            self.ax.scatter(x[idx,status[idx,:]], y[idx,status[idx,:]], **self.kw_patch)
            self.ax.scatter(x[idx,status_not[idx,:]], y[idx,status_not[idx,:]], 
                            **self.kw_patch_dead)
        self.ax.plot(x[:,status[-1,:]], y[:,status[-1,:]], **self.kw_lines)
        self.ax.plot(x[:,status_not[-1,:]], y[:,status_not[-1,:]], **self.kw_lines_dead)

        # Plot the scalar field
        self.field_plotter.draw(fig=self.fig, ax=self.ax, **self.kw_field)
        
        # ------------------------------------------------

        # Configure axes
        self._config_axes()

#######################################################################################