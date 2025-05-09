"""
"""

__all__ = ["plot_centroid_estimation", "plot_estimation_evolution"]

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from matplotlib.legend import Legend # legend artist

# -----------------------------------------

from ssl_simulator.math import build_B, build_L_from_B

# -----------------------------------------

from ..utils import dyn_centroid_estimation

#######################################################################################

def plot_centroid_estimation(ax, P, Z, dt, tf, its=10, lw=2, legend_fs=12, sz=100,
                             legend=False, xlab=False, ylab=False):
    """
    Funtion that visualises the estimation of the centroid
    """
    N = P.shape[0]

    pc = np.sum(P, axis=0)/N
    pb = P.flatten()

    scale = np.max(np.linalg.norm(P-pc,axis=1))

    # Build the Laplacian matrix
    B = build_B(Z,N)
    L = build_L_from_B(B)
    Lb = np.kron(L, np.eye(2))

    # Compute algebraic connectivity (lambda_2)
    eig_vals = np.linalg.eigvals(L)
    min_eig_val = np.min(eig_vals[eig_vals > 1e-7])

    # Simulation -------------------------------------------------------
    t_steps = int(tf/dt)
    t = np.linspace(0, t_steps, int(its*t_steps))
    print(len(t))
    
    xhat_0 = np.zeros_like(pb)
    xhat = odeint(dyn_centroid_estimation, xhat_0, t, args=(Lb,pb,1))

    xc_est0 = P
    xc_est = (pb - xhat[-1]).reshape(P.shape)
    print(f"pc:{pc}, p1:{P[0,:]}, xhat1:{xhat[-1][0:2]}, pc1:{xc_est[0,:]}")
    # ------------------------------------------------------------------

    # -- Plotting --
    # Axis configuration
    ds = scale + scale/5
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

    alpha_edges = 0.3
    for edge in Z:
        ax.plot([P[edge[0]-1,0], P[edge[1]-1,0]], [P[edge[0]-1,1], P[edge[1]-1,1]], "k--", alpha=alpha_edges)

    # Agents
    ax.scatter(P[:,0], P[:,1], color="k", s=15)
    # phi = np.pi/3
    # for n in range(N):
    #     icon = unicycle_patch(X[n,:], phi, "royalblue", **kw_def_patch(scale))
    #     ax.add_patch(icon)

    # Points
    ax.scatter(pc[0], pc[1], c="k", marker="x", s=scale*sz, zorder=4, lw=lw)
    ax.scatter(xc_est0[:,0], xc_est0[:,1], c="red", marker="x", alpha=0.4, s=scale*sz, lw=lw)
    ax.scatter(xc_est[:,0], xc_est[:,1]  , c="red", marker="x", s=scale*sz, lw=lw)


    # Generate the legend
    if legend:
        mrk1 = plt.scatter([],[],c="k"  ,marker="x",s=60)
        mrk2 = plt.scatter([],[],c="red",marker="x",s=60)

        leg = Legend(ax, [mrk1, mrk2], 
                    [r"$p_c$ (Non-computed)",
                    r"${p_{c}}^i$: Actual computed centroid from $i$"],
                    loc="upper left", prop={"size": legend_fs}, ncol=1)

        ax.add_artist(leg)


def plot_estimation_evolution(P, Z, dt, tf, its=10, k=1, dpi=100):
    """
    Funtion to visualise the estimated centroid over time
    """
    N = P.shape[0]

    pc = np.sum(P, axis=0)/N
    pb = P.flatten()

    scale = np.max(np.linalg.norm(P-pc,axis=1))

    # Build the Laplacian matrix
    B = build_B(Z,N)
    L = build_L_from_B(B)
    Lb = np.kron(L, np.eye(2))

    # Compute algebraic connectivity (lambda_2)
    eig_vals = np.linalg.eigvals(L)
    min_eig_val = np.min(eig_vals[eig_vals > 1e-7])

    # Simulation -------------------------------------------------------
    t_steps = int(tf/dt)
    t = np.linspace(0, t_steps, int(its*t_steps+1))
    
    xhat_0 = np.zeros_like(pb)
    xhat = odeint(dyn_centroid_estimation, xhat_0, t, args=(Lb,pb,k))

    xc_est0 = P
    xc_est = (pb - xhat[-1]).reshape(P.shape)
    # ------------------------------------------------------------------

    # -- Plotting --
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax1, ax2 = fig.subplots(2,1)

    # Axis configuration
    ax1.grid(True)
    ax2.grid(True)

    title = r"$t_f$ = {0:.0f} ms, $f$ = {1:.1f} Hz, its = {2:d}".format(tf*1000, its/dt, int(its/dt*tf)) + ", "
    title = title + r"$k$ = {0:.2f}, $\lambda_2$ = {1:.2f}".format(k, min_eig_val)
    ax1.set_title(title)
    
    ax2.set_xlabel("$t$ [T]")
    ax1.set_ylabel("$x$ [L]")
    ax2.set_ylabel("$y$ [L]")

    # Lines
    ax1.axhline(0, c="k", ls="-", lw=1.1)
    ax1.axvline(0, c="k", ls="-", lw=1.1)
    ax2.axhline(0, c="k", ls="-", lw=1.1)
    ax2.axvline(0, c="k", ls="-", lw=1.1)

    for i in range(P.shape[0]):
        ax1.plot(pb[2*i] - xhat[:,2*i], label=i)
        ax2.plot(pb[2*i+1] - xhat[:,2*i+1])

        ax1.legend()

#######################################################################################