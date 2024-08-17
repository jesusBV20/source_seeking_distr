"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Single integrator distributed computing simulator class -
"""

import numpy as np

from scipy.integrate import odeint

# Our utils
from ..toolbox.math_utils import build_B, build_L_from_B, angle_of_vectors
from ..toolbox.math_utils import dyn_centroid_estimation, dyn_mu_estimation
from ..toolbox.math_utils import unit_vec, L_sigma

# ----------------------------------------------------------------------
# Simulator class
# ----------------------------------------------------------------------


class SingIntSimulator:
    def __init__(
        self, q0, Z, sigma_field, dt=1 / 60, kc=1, kl=1, kd=1, its_c=100, its_l=100
    ):
        # Initial state
        self.q0 = q0
        self.p = q0[0]
        self.v = q0[1]

        self.N = self.p.shape[0]
        self.status = np.ones(self.N, dtype=bool)  # (0:non-active, 1:active)

        self.set_Z(Z)

        # Scalar field
        self.sigma_field = sigma_field
        self.sigma = np.zeros((self.N, 1))
        self.sigma_mu = self.sigma_field.mu

        # Simulation parameters and control variables
        self.kc = kc
        self.kl = kl
        self.kd = kd

        self.pc_hat = np.zeros_like(self.p)
        self.pc_comp = np.zeros(2)
        self.x = np.zeros_like(self.p)

        self.mu = np.zeros_like(self.p)
        self.mu_comp = np.zeros_like(self.p)

        # Shape control
        self.mod_shape = False
        self.xd = self.x
        self.ks = 1 / 3

        # Integrator and ED solver parameters
        self.t = 0
        self.dt = dt
        self.tc = np.linspace(0, its_c, its_c + 1)
        self.tl = np.linspace(0, its_l, its_l + 1)

        # Estimation log variables
        self.pc_hat_log = np.zeros((len(self.tc), self.N, 2))
        self.mu_log = np.zeros((len(self.tl), self.N, 2))

        # Simulation data providerdata_pc_hat
        self.data = {
            "t": [],
            "p": [],
            "pc_hat": [],
            "mu": [],
            "pc_hat_log": [],
            "mu_log": [],
            "pc_comp": [],
            "mu_comp": [],
            "status": [],
        }

    def set_Z(self, Z):
        """
        Set the new Z and build the Laplacian matrix
        """
        self.Z = Z
        self.B = build_B(Z, self.N)
        self.gen_L()

    def gen_L(self):
        # Modify the actual B considering dead units
        B_kill = np.copy(self.B)
        for i in np.where(self.status == 0)[0]:
            for j in range(self.B.shape[1]):
                if self.B[i, j] != 0:
                    B_kill[:, j] = 0
        self.B = np.copy(B_kill)

        # Generate the Laplacian matrix
        self.L = build_L_from_B(self.B)
        self.Lb = np.kron(self.L, np.eye(2))

        # Compute algebraic connectivity
        eig_vals = np.linalg.eigvals(self.L)
        self.lambda2 = np.min(eig_vals[abs(eig_vals) > 1e-7])

    def update_data(self):
        """
        Update the data dictionary with a new entry
        """
        self.data["t"].append(self.t)
        self.data["p"].append(self.p)
        self.data["pc_hat"].append(self.pc_hat)
        self.data["mu"].append(self.mu)
        self.data["pc_hat_log"].append(self.pc_hat_log)
        self.data["mu_log"].append(self.mu_log)
        self.data["pc_comp"].append(self.pc_comp)
        self.data["mu_comp"].append(self.mu_comp)
        self.data["status"].append(np.copy(self.status))

    def get_pc_estimation(self):
        """
        Distributed estimation of the centroid
        """
        pb = self.p.flatten()

        x_hat_0 = np.zeros_like(pb)
        x_hat = odeint(
            dyn_centroid_estimation, x_hat_0, self.tc, args=(self.Lb, pb, self.kc)
        )
        self.pc_hat_log = np.copy(pb - x_hat).reshape(self.pc_hat_log.shape)

        pc_hat = (pb - x_hat[-1]).reshape(self.p.shape)
        x_hat = x_hat[-1].reshape(self.p.shape)
        return pc_hat, x_hat

    def get_mu_estimation(self):
        """
        Distributed estimation of the ascending direction
        """
        mu_i = self.sigma * self.x

        lhat_0 = np.copy(mu_i.flatten())
        lhat = odeint(dyn_mu_estimation, lhat_0, self.tl, args=(self.Lb, self.kl))
        self.mu_log = np.copy(lhat).reshape(self.mu_log.shape)

        if np.linalg.norm(lhat[-1].reshape(self.x.shape)) < 1e-4:
            mu = self.x * 0
        else:
            mu = unit_vec(lhat[-1].reshape(self.x.shape))
        return mu

    def kill_agents(self, agents_index):
        """
        Update the Lalplacian matrix to kill the connections of the
        specified agents, and update their status to (0)-"non-active"
        """
        if not isinstance(agents_index, list):
            agents_index = [agents_index]

        # Update the agents status
        self.status[agents_index] = 0

        # Generate the new Laplacian matrix
        self.gen_L()

    def shape_control(self):
        return self.ks * (self.xd - self.x)

    def unicycle_kinematics(self):
        """
        Unicycle kinematics
        """
        p_dot = self.v * np.array([np.cos(self.phi), np.sin(self.phi)]).T
        phi_dot = self.omega
        return p_dot, phi_dot

    # ----- EULER INTEGRATION
    def int_step(self):
        """
        Euler integration (Step-wise)
        """
        # Centroid estimation
        self.pc_hat, self.x = self.get_pc_estimation()
        self.pc_comp = np.mean(self.p[self.status, :], axis=0)

        self.sigma = self.sigma_field.value(self.p)[:, None]

        # Ascending direction estimation
        self.mu = self.get_mu_estimation()
        self.mu_comp = L_sigma(
            self.p[self.status, :] - self.pc_hat[self.status, :],
            self.sigma_field.value(self.p[self.status, :]),
        )

        # Robot dynamics integration
        p_dot = self.v * np.ones_like(self.p) * self.mu

        # Don't move if you are close to the source
        # d_thr = 3
        # p_dot = (
        #     p_dot * (np.linalg.norm(self.sigma_mu - self.p, axis=1) > d_thr)[:, None]
        # )

        # Shape controller
        if self.mod_shape:
            p_dot = p_dot + self.shape_control()
        p_dot[~self.status, :] = 0

        self.t = self.t + self.dt
        self.p = self.p + self.dt * p_dot

        # Update output data
        self.update_data()
