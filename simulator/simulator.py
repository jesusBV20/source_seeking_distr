"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Simulator class (distributed computing) -
"""

import numpy as np
from scipy.integrate import odeint

from simulator.utils import build_B, build_L_from_B, angle_of_vectors
from simulator.utils import dyn_centroid_estimation, dyn_mu_estimation

from toolbox.math_utils import unit_vec, L_sigma

# ----------------------------------------------------------------------
# Simulator class
# ----------------------------------------------------------------------

class simulator:
    """
    This class...
    """
    def __init__(self, q0, Z, sigma_field, dt=1/60, kc=1, kl=1, kd=1,
                 its_c=100, its_l=100):
        # Initial state
        self.q0 = q0
        self.p = q0[0]
        self.v = q0[1]
        self.phi = q0[2]

        self.Z = Z
        self.N = self.p.shape[0]

        # Build the Laplacian matrix
        self.B = build_B(Z, self.N)
        L = build_L_from_B(self.B)
        self.Lb = np.kron(L, np.eye(2))

        # Compute algebraic connectivity
        eig_vals = np.linalg.eigvals(L)
        self.lambda2 = np.min(eig_vals[abs(eig_vals) > 1e-7])

        # Scalar field
        self.sigma_field = sigma_field
        self.sigma = np.zeros((self.N,1))

        # Simulation parameters and control variables
        self.kc = kc
        self.kl = kl
        self.kd = kd

        self.pc_hat = np.zeros_like(self.p)
        self.pc_comp = np.zeros(2)
        self.x = np.zeros_like(self.p)

        self.mu = np.zeros_like(self.p)
        self.mu_comp = np.zeros_like(self.p)

        self.omega = np.zeros(self.N)
        self.status = np.ones(self.N, dtype=bool) # (0:non-active, 1:active)

        # Integrator and ED solver parameters
        self.t = 0
        self.dt = dt
        self.tc = np.linspace(0, its_c, its_c+1)
        self.tl = np.linspace(0, its_l, its_l+1)

        # Simulation data providerdata_pc_hat
        self.data = {"t": [], "p": [], "phi": [], "pc_hat": [], "mu": [],
                     "pc_comp": [], "mu_comp": [], "status": []}

    def update_data(self):
        """
        Update the data dictionary with a new entry
        """
        self.data["t"].append(self.t)
        self.data["p"].append(self.p)
        self.data["phi"].append(self.phi)
        self.data["pc_hat"].append(self.pc_hat)
        self.data["mu"].append(self.mu)
        self.data["pc_comp"].append(self.pc_comp)
        self.data["mu_comp"].append(self.mu_comp)
        self.data["status"].append(self.status)
        
    def get_pc_estimation(self):
        """
        Distributed estimation of the centroid
        """
        pb = self.p.flatten()
        x_hat_0 = np.zeros_like(pb)
        x_hat = odeint(dyn_centroid_estimation, x_hat_0, self.tc, args=(self.Lb,pb,self.kc))

        pc_hat = (pb - x_hat[-1]).reshape(self.p.shape)
        x_hat = x_hat[-1].reshape(self.p.shape)
        return pc_hat, x_hat
    
    def get_mu_estimation(self):
        """
        Distributed estimation of the ascending direction
        """
        mu_i = self.sigma * self.x
    
        lhat_0 = np.copy(mu_i.flatten())
        lhat = odeint(dyn_mu_estimation, lhat_0, self.tl, args=(self.Lb,self.kl))
        
        mu = unit_vec(lhat[-1].reshape(self.x.shape))
        return mu

    def mu_tracking_control(self):
        """
        Compute the control law to track the ascending direction mu
        """
        p_dot_unit = np.array([np.cos(self.phi), np.sin(self.phi)]).T
        omega = - self.kd * angle_of_vectors(self.mu, p_dot_unit)
        return omega

    def kill_agents(self, agents_index):
        """
        Update the Lalplacian matrix to kill the connections of the
        specified agents, and update their status to (0)-"non-active"
        """
        if not isinstance(agents_index, list):
            agents_index = [agents_index]

        # Generate the new indicende matrix
        newB = np.copy(self.B)
        for i in agents_index:
            for j in range(newB.shape[1]):
                if self.B[i,j] != 0:
                    newB[:,j] = 0

        # Rebuild the Laplacian matrix
        self.B = np.copy(newB)
        L = build_L_from_B(self.B)
        self.Lb = np.kron(L, np.eye(2))

        # Update the agents status
        self.status[agents_index] = 0

    # ----- UNICYCLE  KINEMATICS
    def unicycle_kinematics(self):
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

        self.sigma = self.sigma_field.value(self.p)[:,None]

        # Ascending direction estimation
        self.mu = self.get_mu_estimation()
        self.mu_comp = L_sigma(self.p[self.status, :]-self.pc_hat[self.status, :], 
                               self.sigma_field.value(self.p[self.status, :]))

        # Compute the mu tracking control input
        self.omega = self.mu_tracking_control()

        # Robot dynamics integration
        p_dot, phi_dot = self.unicycle_kinematics()

        p_dot[~self.status, :] = 0 
        phi_dot[~self.status] = 0

        self.t = self.t + self.dt
        self.p = self.p + self.dt*p_dot
        self.phi = self.phi + self.dt*phi_dot

        # Update output data
        self.update_data()