"""
"""

__all__ = ["SourceSeekingSI"]

import numpy as np

from ssl_simulator import RobotModel
from ssl_simulator.components.scalar_fields import ScalarField
from ssl_simulator.components.network import Graph

#######################################################################################

class SourceSeekingSI(RobotModel):
    def __init__(self, initial_state, scalar_field: ScalarField, graph: Graph,
                 epsilon_x: float = 1, epsilon_mu: float = 1):
        
        self.scalar_field = scalar_field
        self.graph = graph
        self.epsilon_x = epsilon_x
        self.epsilon_mu = epsilon_mu
    
        # Robot model state
        self.state = {
            "p": initial_state[0],
            "x_hat": np.zeros_like(initial_state[0]),
            "mu_hat": np.zeros_like(initial_state[0]),
            "mu": np.zeros_like(initial_state[0]),
        }

        # Robot model state time variation
        self.state_dot = {}
        for key,value in self.state.items():
            self.state_dot.update({key+"_dot": value*0})

        # Robot model settings to be tracked by logger
        self.tracked_settings = {
            "ep_x": epsilon_x,
            "ep_mu": epsilon_mu,
            "field_class": scalar_field,
            "graph": graph,
        }

        # Robot model data
        self.init_data()

    # ---------------------------------------------------------------------------------
    def dynamics(self, state, control_vars):
        p = state["p"]

        # Run a network iteration and update the estimated variables
        pb = p.flatten()
        Lb = self.graph.Lb

        x_hat = state["x_hat"]
        x_hatb = x_hat.flatten()
        mu_hat_b = state["mu_hat"].flatten()

        mu = self.scalar_field.value(p)[:, None] * x_hat
        mub = mu.flatten()
        self.state["mu"] = mu

        x_hat_dot_b = - (Lb.dot(x_hatb) - Lb.dot(pb)) / self.epsilon_x
        mu_hat_dot_b = - (Lb.dot(mu_hat_b) - Lb.dot(mub)) / self.epsilon_mu
        self.state_dot["x_hat_dot"] = x_hat_dot_b.reshape(p.shape)
        self.state_dot["mu_hat_dot"] = mu_hat_dot_b.reshape(p.shape)
        
        # Update p_dot with the control variable
        self.state_dot["p_dot"] = next(iter(control_vars.values())) * np.ones(state["p"].shape)
        return self.state_dot

#######################################################################################