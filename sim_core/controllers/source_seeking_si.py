"""
"""

__all__ = ["SourceSeekingSI"]

import numpy as np

from ssl_simulator.math import unit_vec, L_sigma
from ssl_simulator.components.scalar_fields import ScalarField
from ssl_simulator.components.network import Graph

from ssl_simulator import Controller

#######################################################################################

class SourceSeekingSI(Controller):
    def __init__(self, scalar_field: ScalarField, graph: Graph, speed: float):

        self.scalar_field = scalar_field
        self.graph = graph

        # Controller settings
        self.speed = speed

        # ---------------------------
        # Controller output variables
        self.control_vars = {
            "u": None,
        }

        # Controller variables to be tracked by logger
        self.tracked_vars = {
            "speed": self.speed,
            "status": None,
            "sigma": None,
            "pc_centralized": None,
            "mu_centralized": None,
        }

        # Controller data
        self.init_data()

    def compute_control(self, time, state):
        """
        """
        p = state["p"]
        status = np.copy(self.graph.agents_status)

        # Compute centralized variables for logging
        pc_centralized = np.mean(p[status, :], axis=0)
        mu_centralized = L_sigma(
                p[status, :] - pc_centralized,
                self.scalar_field.value(p[status, :]),
            )
        
        # Initialize and update control inputs
        self.control_vars["u"] = self.speed * unit_vec(mu_centralized) * np.ones_like(p) * status[:, None]

        # Update tracked variables
        self.tracked_vars["status"] = status
        self.tracked_vars["sigma"] = self.scalar_field.value(p)[:, None]
        self.tracked_vars["pc_centralized"] = pc_centralized
        self.tracked_vars["mu_centralized"] = mu_centralized

        return self.control_vars
    
    #######################################################################################