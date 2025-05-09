"""
"""

__all__ = ["anim_mu_estimation"]

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from matplotlib.legend import Legend # legend artist
from matplotlib.animation import FuncAnimation

# -----------------------------------------

from ssl_simulator.math import unit_vec, L_sigma
from ssl_simulator.math import build_B, build_L_from_B
from ssl_simulator.visualization import vector2d
from ssl_simulator.components.scalar_fields import PlotterScalarField

# -----------------------------------------

from ..utils import dyn_mu_estimation, kw_def_arrow

#######################################################################################

def anim_mu_estimation(P, Z, scalar_field, dt, tf, its=10, k=1, fps=30):
    """
    Funtion to animate the ascending direction estimation
    """
    N = P.shape[0]

    pc = np.sum(P, axis=0)/N
    pb = P.flatten()
    X = P - pc

    # Evaluate the scalar field and compute the L_sigma (the not estimated one)
    sigma_values = scalar_field.value(P)
    mu_i_comp = L_sigma(X, sigma_values)

    mu_i = sigma_values[:,None] * X
    mu_i_b = mu_i.flatten()

    # Build the Laplacian matrix
    B = build_B(Z,N)
    L = build_L_from_B(B)
    Lb = np.kron(L, np.eye(2))

    # Compute algebraic connectivity (lambda_2)
    eig_vals = np.linalg.eigvals(L)
    min_eig_val = np.min(eig_vals[abs(eig_vals) > 1e-7])

    # Simulation -------------------------------------------------------
    t_steps = int(tf/dt)
    t = np.linspace(0, t_steps, int(its*t_steps+1))
    
    mu_0 = np.copy(mu_i_b)
    mu = odeint(dyn_mu_estimation, mu_0, t, args=(Lb,k))

    mu = mu.reshape((int(its*t_steps+1), N, 2))
    # ------------------------------------------------------------------

    # -- Animation --
    fig = plt.figure()
    ax = fig.subplots()

    # Axis configuration
    scale = np.max(np.linalg.norm(X, axis=1))
    arr_scale = 0.8
    ds = scale + 1

    ax.axis([pc[0]-1.5*ds, pc[0]+1.5*ds, pc[1]-ds, pc[1]+1.6*ds])
    ax.set_aspect("equal")
    ax.grid(True)

    title = r"$N$ = {0:d}, $t_f$ = {1:.0f} ms, $f$ = {2:.1f} kHz".format(N, tf*1000, its/dt) + "\n"
    title = title + r"$k$ = {0:.2f}, $\lambda_2$ = {1:.2f}".format(k,min_eig_val)
    ax.set_title(title)
    
    ax.set_xlabel("$X$ [L]")
    ax.set_ylabel("$Y$ [L]")

    # Lines
    # ax.axhline(0, c="k", ls="-", lw=1.1)
    # ax.axvline(0, c="k", ls="-", lw=1.1)

    alpha_edges = 0.8/(1+np.log(N))
    for edge in Z:
        ax.plot([P[edge[0]-1,0], P[edge[1]-1,0]], [P[edge[0]-1,1], P[edge[1]-1,1]], 
                "k--", alpha=alpha_edges, zorder=-1)

    # Agents
    ax.scatter(P[:,0], P[:,1], color="k", s=15, zorder=5)

    # Points
    ax.scatter(pc[0], pc[1], c="k", marker="x", s=arr_scale*100, zorder=3)

    # Gradient arrow and computed ascending direction L1
    scalar_field_plotter = PlotterScalarField(scalar_field)
    scalar_field_plotter.draw_grad(pc, ax, kw_def_arrow(arr_scale))
    vector2d(ax, pc, unit_vec(mu_i_comp), c="blue", alpha=0.7, **kw_def_arrow(arr_scale), zorder=2)

    # Estimated ascending direction mu
    quiv_list = []
    for n in range(N):
        vector2d(ax, P[n,:], unit_vec(mu[0,n,:])*1.2, c="red", alpha=0.3, **kw_def_arrow(arr_scale), zorder=4)
        quiv = vector2d(ax, P[n,:], unit_vec(mu[0,n,:])*1.2, c="red", alpha=0.9, **kw_def_arrow(arr_scale), zorder=4)
        quiv_list.append(quiv)
    
    # Generate the legend
    arr1 = plt.scatter([],[],c="k",marker=r"$\uparrow$",s=60)
    arr2 = plt.scatter([],[],c="blue",marker=r"$\uparrow$",s=60)
    arr3 = plt.scatter([],[],c="red",marker=r"$\uparrow$",s=60)

    leg = Legend(ax, [arr1, arr2, arr3], 
                [r"$\nabla\sigma$ (Non-computed)",
                    r"$L_1$ (Non-computed)",
                    r"$\mu_i$: Actual computed ascending direction from $i$",
                ],
                loc="upper left", prop={"size": 12}, ncol=1)

    ax.add_artist(leg)

    # -- Building the animation --
    anim_frames = t_steps

    # Function to update the animation
    def animate(i):
        # Update the centroid estimation markers
        li = its*i
        mu_list = unit_vec(mu[li,:,:])
        for n in range(N):
            quiv_list[n].set_data(dx=mu_list[n,0], dy=mu_list[n,1])

        # Update the title
        title = r"$t_f$ = {0:.0f} ms, $f$ = {1:.1f} Hz".format((dt*i)*1000, its/dt) + "\n"
        title = title + r"$k$ = {0:.2f}, $\lambda_2$ = {1:.2f}".format(k, min_eig_val)
        ax.set_title(title)

    # Generate the animation
    print("Simulating {0:d} frames...".format(anim_frames))
    anim = FuncAnimation(fig, animate, frames=tqdm(range(anim_frames), initial=1, position=0), 
                         interval=1/fps*1000)
    anim.embed_limit = 40

    # Close plots and return the animation class to be compiled
    plt.close()
    return anim

#######################################################################################