"""
"""

__all__ = ["plot_mu_estimation"]

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from matplotlib.legend import Legend # legend artist

# -----------------------------------------

from ssl_simulator.math import unit_vec, L_sigma
from ssl_simulator.math import build_B, build_L_from_B
from ssl_simulator.visualization import vector2d
from ssl_simulator.components.scalar_fields import PlotterScalarField

# -----------------------------------------

from ..utils import dyn_mu_estimation, kw_def_arrow

#######################################################################################

def plot_mu_estimation(ax, P, Z, scalar_field, dt, tf, its=10,
                             legend=False, xlab=False, ylab=False):
    """
    Funtion that visualises the estimation of the ascending direction
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
    t = np.linspace(0, t_steps, int(its*t_steps))
    print(len(t))

    lhat_0 = np.copy(mu_i_b)
    lhat = odeint(dyn_mu_estimation, lhat_0, t, args=(Lb,1))

    mu0 = mu_i
    mu = lhat[-1].reshape(P.shape)
    # ------------------------------------------------------------------

    # -- Plot --
    
    # Axis configuration
    scale = np.max(np.linalg.norm(X, axis=1))
    arr_scale = 0.8
    ds = scale + 1

    ax.axis([pc[0]-ds, pc[0]+ds, pc[1]-ds, pc[1]+ds])
    ax.set_aspect("equal")
    ax.grid(True)

    title = r"$t$ = {0:.0f} ms, $f$ = {1:.1f} Hz".format(tf*1000, its/dt)
    ax.set_title(title)
    
    if xlab:
       ax.set_xlabel("$X$ [L]")
    if ylab:
        ax.set_ylabel("$Y$ [L]")

    # Lines
    # ax.axhline(0, c="k", ls="-", lw=1.1)
    # ax.axvline(0, c="k", ls="-", lw=1.1)

    alpha_edges = 0.8/(1+np.log(N))
    alpha_edges = 0.3
    for edge in Z:
        ax.plot([P[edge[0]-1,0], P[edge[1]-1,0]], [P[edge[0]-1,1], P[edge[1]-1,1]], 
                "k--", alpha=alpha_edges, zorder=-1)

    # Agents
    ax.scatter(P[:,0], P[:,1], color="k", s=15, zorder=5)
    # phi = np.pi/3
    # for n in range(N):
    #     icon = unicycle_patch(P[n,:], phi, "royalblue", **kw_def_patch(scale))
    #     ax.add_patch(icon)

    # Points
    ax.scatter(pc[0], pc[1], c="k", marker="x", s=arr_scale*100, zorder=3)

    # Gradient arrow and computed ascending direction L1
    scalar_field_plotter = PlotterScalarField(scalar_field)
    scalar_field_plotter.draw_grad(pc, ax, norm_fct=arr_scale*1.2, **kw_def_arrow(arr_scale))
    vector2d(ax, pc, unit_vec(mu_i_comp), c="blue", alpha=0.7, **kw_def_arrow(arr_scale), zorder=2)

    # Estimated ascending direction mu
    for n in range(N):
        vector2d(ax, P[n,:], unit_vec(mu0[n,:]), c="red", alpha=0.3, **kw_def_arrow(arr_scale), zorder=4)
        vector2d(ax, P[n,:], unit_vec(mu[n,:]), c="red", alpha=0.9, **kw_def_arrow(arr_scale), zorder=4)
    
    # Generate the legend
    if legend:
        arr1 = plt.scatter([],[],c="k",marker=r"$\uparrow$",s=60)
        arr2 = plt.scatter([],[],c="blue",marker=r"$\uparrow$",s=60)
        arr3 = plt.scatter([],[],c="red",marker=r"$\uparrow$",s=60)

        leg = Legend(ax, [arr1, arr2, arr3], 
                    [r"$\nabla\sigma$ (Non-computed)",
                     r"$L_\sigma$ (Non-computed)",
                     r"$\mu_c^i$: Actual computed ascending direction from $i$",
                    ],
                    loc="upper left", prop={"size": 11}, ncol=1)

        ax.add_artist(leg)

#######################################################################################