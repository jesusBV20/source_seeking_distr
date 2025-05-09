"""
"""

__all__ = ["SourceSeekingUny"]

import numpy as np

from ssl_simulator.math import unit_vec, L_sigma, angle_of_vectors
from ssl_simulator.components.scalar_fields import ScalarField
from ssl_simulator.components.network import Graph

from ssl_simulator import Controller

#######################################################################################

class SourceSeekingUny(Controller):
    def __init__(self, scalar_field: ScalarField, graph: Graph, kd: float = 1):

        self.scalar_field = scalar_field
        self.graph = graph

        # Controller settings
        self.kd = kd

        # ---------------------------
        # Controller output variables
        self.control_vars = {
            "u": None,
        }

        # Controller variables to be tracked by logger
        self.tracked_vars = {
            "status": None,
            "sigma": None,
            "sigma_grad": None,
            "mu_centralized": None,
            "pc_centralized": None,
            "scalar_field_mu": None,
        }

        # Controller settings to be tracked by logger
        self.tracked_settings = {
            "k_form": kd,
            "field_class": self.scalar_field,
        }

        # Controller data
        self.init_data()

    def mu_tracking_control(self, mu, theta):
        """
        Compute the control law to track the ascending direction mu
        """
        mu = unit_vec(mu)
        p_dot_unit = np.array([np.cos(theta), np.sin(theta)]).T
        omega = -self.kd * angle_of_vectors(mu, p_dot_unit)
        return omega
    
    def compute_control(self, time, state):
        """
        """
        p = state["p"]
        theta = state["theta"]
        status = np.copy(self.graph.agents_status)

        # Compute centralized variables for logging
        pc_centralized = np.mean(p[status, :], axis=0)
        mu_centralized = L_sigma(
                p[status, :] - pc_centralized,
                self.scalar_field.value(p[status, :]),
            )
        
        # Compute the mu tracking control input (omega)
        self.control_vars["u"] = self.mu_tracking_control(mu_centralized * np.ones_like(p), theta)

        # Update tracked variables
        self.tracked_vars["status"] = status
        self.tracked_vars["sigma"] = self.scalar_field.value(p)[:, None]
        self.tracked_vars["sigma_grad"] = self.scalar_field.grad(pc_centralized)[0]
        self.tracked_vars["pc_centralized"] = pc_centralized
        self.tracked_vars["mu_centralized"] = mu_centralized
        self.tracked_vars["scalar_field_mu"] = np.array(self.scalar_field.mu)

        return self.control_vars
    
    #######################################################################################