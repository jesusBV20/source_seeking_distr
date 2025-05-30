"""
"""

__all__ = ["PlotterSimBasicUny"]

import numpy as np

# Graphic tools
import matplotlib.pyplot as plt

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import Plotter, config_data_axis, unicycle_patch

from ssl_simulator.components.scalar_fields import PlotterScalarField

#######################################################################################

class PlotterSimBasicUny(Plotter):
    def __init__(self, data, scalar_field, **kwargs):
        super().__init__(**kwargs)

        self.data = data
        self.scalar_field = scalar_field
        self.field_plotter = PlotterScalarField(scalar_field)

        # Default visual properties
        kw_ax = {
            "y_right": False,
            "xlims": None,
            "ylims": None  
        }
        
        kw_patch = {
            "size": 2,
            "fc": "royalblue",
            "ec": "k",
            "lw": 0.5,
            "zorder": 3,
        }

        kw_patch_dead = {
            "c": "darkred",
            "linewidths": 0,
        }

        # Update defaults with user-specified values
        self.kw_ax = parse_kwargs(kwargs, kw_ax)
        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_patch_dead = parse_kwargs(kw_patch_dead, self.kw_patch)
        
        # Create subplots
        self.ax = self.fig.subplots()
        self.config_axes()

    # ---------------------------------------------------------------------------------

    def config_axes(self):
        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$ [L]")
        self.ax.set_aspect("equal")
        config_data_axis(self.ax, **self.kw_ax)

    def plot(self, num_patches=2, **kwargs):

        # Lines visual properties
        kw_lines = {
            "c": "royalblue",
            "lw": 0.2,
            "zorder": 2
        }

        kw_lines_dead = {
            "c": "darkred",
        }

        kw_lines = parse_kwargs(kwargs, kw_lines)
        kw_lines_dead= parse_kwargs(kw_lines_dead, kw_lines)
        
        # ------------------------------------------------

        # Extract derired data
        x = np.array(self.data["p"].tolist())[1:,:,0]
        y = np.array(self.data["p"].tolist())[1:,:,1]
        theta = np.array(self.data["theta"].tolist())[1:,:]

        status = np.array(self.data["status"].tolist())
        status_not = np.logical_not(status)

        # ------------------------------------------------
        # Plot the robots
        idx_list = np.linspace(0, x.shape[0]-1, num_patches, dtype=int)

        for idx in idx_list:
            for i in range(x.shape[1]):
                if status[idx,i]:
                    kw_patch = self.kw_patch
                else:
                    kw_patch = self.kw_patch_dead

                icon = unicycle_patch(
                    [x[idx,i], y[idx,i]], theta[idx,i], 
                    **kw_patch)
              
                if idx == 0:
                    icon.set_alpha(0.5)

                self.ax.add_patch(icon)

        self.ax.plot(x[:,status[-1,:]], y[:,status[-1,:]], **kw_lines)
        self.ax.plot(x[:,status_not[-1,:]], y[:,status_not[-1,:]], **kw_lines_dead)

        # Plot the scalar field
        kw_field = dict(contour_levels=8, n=1000)
        kw_field = parse_kwargs(kwargs, kw_field)
        self.field_plotter.draw(fig=self.fig, ax=self.ax, **kw_field)
        
        # ------------------------------------------------

        # Configure axes
        self.config_axes()

#######################################################################################