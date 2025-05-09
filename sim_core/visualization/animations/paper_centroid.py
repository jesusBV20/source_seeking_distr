"""
"""

__all__ = ["anim_centroid_estimation"]

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from matplotlib.legend import Legend # legend artist
from matplotlib.animation import FuncAnimation

# -----------------------------------------

from ssl_simulator.math import build_B, build_L_from_B

# -----------------------------------------

from ..utils import dyn_centroid_estimation

#######################################################################################

def anim_centroid_estimation(P, Z, dt, tf, its=10, k=1, fps=30):
    """
    Funtion to animate the centroid estimation
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

    xhat = xhat.reshape((int(its*t_steps+1), N, 2))
    pc_hat = P - xhat
    # ------------------------------------------------------------------

    # -- Animation --
    fig = plt.figure()
    ax = fig.subplots()

    # Axis configuration
    ds = scale + scale/5
    ax.axis([pc[0]-ds, pc[0]+ds, pc[1]-ds, pc[1]+ds])
    ax.set_aspect("equal")
    ax.grid(True)

    title = r"$t_f$ = {0:.0f} ms, $f$ = {1:.1f} Hz".format(tf*1000, its/dt) + "\n"
    title = title + r"$k$ = {0:.2f}, $\lambda_2$ = {1:.2f}".format(k, min_eig_val)
    ax.set_title(title)

    ax.set_xlabel("$Y$ [L]")
    ax.set_ylabel("$X$ [L]")

    # Lines
    ax.axhline(0, c="k", ls="-", lw=1.1)
    ax.axvline(0, c="k", ls="-", lw=1.1)

    for edge in Z:
        ax.plot([P[edge[0]-1,0], P[edge[1]-1,0]], [P[edge[0]-1,1], P[edge[1]-1,1]], "k--", alpha=0.6)

    # Agents
    ax.scatter(P[:,0], P[:,1], color="k", s=15)

    # Points
    ax.scatter(pc[0], pc[1], c="k", marker="x", s=100, zorder=4, lw=2)
    ax.plot(pc_hat[0,:,0], pc_hat[0,:,1], "r", linestyle = "None", marker="x", markersize=10, alpha=0.4, lw=2)
    pts, = ax.plot(pc_hat[0,:,0], pc_hat[0,:,1], "r", linestyle = "None", marker="x", markersize=10, lw=2)


    # Generate the legend
    mrk1 = plt.scatter([],[],c="k"  ,marker="x",s=60)
    mrk2 = plt.scatter([],[],c="red",marker="x",s=60)

    leg = Legend(ax, [mrk1, mrk2], 
                [r"$p_c$ (Non-computed)",
                r"${p_{c}}^i$: Actual computed centroid from $i$"],
                loc="upper left", prop={"size": 11}, ncol=1)

    ax.add_artist(leg)

    # -- Building the animation --
    anim_frames = t_steps

    # Function to update the animation
    def animate(i):
        # Update the centroid estimation markers
        li = its*i
        pts.set_data(pc_hat[li,:,0], pc_hat[li,:,1])

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