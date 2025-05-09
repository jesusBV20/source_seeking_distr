"""
"""

__all__ = ["SourceSeekingDistrSI"]

import numpy as np

from ssl_simulator.math import unit_vec, calc_mu_centralized
from ssl_simulator.components.scalar_fields import ScalarField
from ssl_simulator.components.network import Graph

from ssl_simulator import Controller

#######################################################################################

class SourceSeekingDistrSI(Controller):
    def __init__(
        self, scalar_field: ScalarField, graph: Graph,
        speed: float, p_star: np.ndarray = None, k_form: float = 0):

        self.scalar_field = scalar_field
        self.graph = graph

        # Controller settings
        self.speed = speed
        self.k_form = k_form
        self.set_pstar(p_star)

        # ---------------------------
        # Controller output variables
        self.control_vars = {
            "u": None,
        }

        # Controller variables to be tracked by logger
        self.tracked_vars = {
            "speed": self.speed,
            
            "Z": None,
            "lambda2": None,
            "status": None,
            "sigma": None,
            "sigma_grad": None,
            "p_star": None,

            "pc_estimated": None,
            "muc_estimated": None,
            "pc_centralized": None,
            "muc_centralized": None,
        }

        # Controller settings to be tracked by logger
        self.tracked_settings = {
            "k_form": k_form,
            "p_star": self.p_star,
        }

        # Controller data
        self.init_data()

    def set_pstar(self, p_star):
        self.p_star = p_star
        self.p_starb = p_star.flatten()

    def compute_control(self, time, state):
        """
        """
        p = state["p"]
        status = np.copy(self.graph.agents_status)

        x_hat = state["x_hat"]
        mu_hat = state["mu_hat"]
        mu = state["mu"]

        # ---------------------------
        pb = state["p"].flatten()
        Lb = self.graph.Lb
        # x_hatb = x_hat.flatten()
        # Lb = self.graph.Lb
        # u_formation = - self.k_form * (Lb.dot(x_hatb) - Lb.dot(self.x_starb)).reshape(p.shape)
        u_formation = - self.k_form * (Lb.dot(pb) - Lb.dot(self.p_starb)).reshape(p.shape)
        # ---------------------------

        muc_estimated = mu - mu_hat

        u = self.speed * unit_vec(muc_estimated) + u_formation
        self.control_vars["u"] = u * status[:,None]

        # ---------------------------
        # Compute centralized variables for logging
        pc_centralized = np.mean(p[status, :], axis=0)
        muc_centralized = calc_mu_centralized(
            p[status, :] - pc_centralized,
            self.scalar_field.value(p[status, :]),
        )
        
        # Update tracked variables
        self.tracked_vars["Z"] = np.array(self.graph.Z)
        self.tracked_vars["lambda2"] = self.graph.lambda2
        self.tracked_vars["status"] = status
        self.tracked_vars["sigma"] = self.scalar_field.value(p)[:, None]
        self.tracked_vars["sigma_grad"] = self.scalar_field.grad(p)
        self.tracked_vars["p_star"] = self.p_star
        self.tracked_vars["pc_centralized"] = pc_centralized
        self.tracked_vars["muc_centralized"] = muc_centralized
        self.tracked_vars["pc_estimated"] = p - x_hat
        self.tracked_vars["muc_estimated"] = muc_estimated

        return self.control_vars
    
    #######################################################################################