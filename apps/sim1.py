import os

import numpy as np
import matplotlib.pyplot as plt

from ssl_simulator import SimulationEngine, add_src_to_path, create_dir
from ssl_simulator.math import XY_distrib
from ssl_simulator.robot_models import Unicycle2D
from ssl_simulator.visualization import PlotterUnySS

from ssl_simulator.components.scalar_fields import SigmaNonconvex
from ssl_simulator.components.network import Graph

add_src_to_path(__file__)
from apps import AppGameSS
from sim_core.controllers import SourceSeekingUny

def setup_sim1(dt=0.01):
    # Define the initial state
    N = 40

    pc, lims = np.array([-110,-50]), np.array([9,2])
    p = XY_distrib(N, pc, lims)
    speed = np.ones(N) * 12
    theta = np.random.rand(N) * np.pi

    x0 = [p, speed, theta]

    # Define the graph (Z doesn't matter in centralized)
    graph = Graph(N, [])

    # Define the scalar field
    k, mu, dev = 0.04, [0, 0], 10
    sigma_field = SigmaNonconvex(k=k, dev=dev, mu=mu, a=[1,0], b=[0,1])

    # Controller settings
    kd = 1

    # Select robot model, controller and init the simulation engine
    tail_len = 1

    robot_model = Unicycle2D(x0, omega_lims=[-1.5,1.5])
    controller = SourceSeekingUny(scalar_field=sigma_field, graph=graph, kd=kd)
    simulator_engine = SimulationEngine(robot_model, controller, time_step=dt, log_size=tail_len)

    fig, ax = plt.subplots(dpi=100, figsize=(16,9))
    simulator_plotter = PlotterUnySS(ax, simulator_engine.data, tail_len=tail_len)

    return fig, ax, sigma_field, simulator_engine, simulator_plotter

# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    app = AppGameSS(*setup_sim1())
    plt.show()