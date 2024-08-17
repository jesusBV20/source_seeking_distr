"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Souce seeking distributed plotting utilities -
"""

import numpy as np

# Graphic tools (external)
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# --------------------------------------------------------------------------------------


def kw_def_arrow(scale):
    return {
        "s": scale,
        "lw": 2 * scale ** (1 / 5),
        "hw": 0.1 * scale ** (1 / 5),
        "hl": 0.2 * scale ** (1 / 5),
    }


def kw_def_patch(scale):
    return {"size": 0.15 * scale, "lw": 0.2 * scale ** (1 / 2)}


def vector2d(ax, P0, Pf, c="k", ls="-", s=1, lw=0.7, hw=0.1, hl=0.2, alpha=1, zorder=1):
    """
    Function to easy plot a 2D vector
    """
    quiv = ax.arrow(
        P0[0],
        P0[1],
        s * Pf[0],
        s * Pf[1],
        lw=lw,
        color=c,
        ls=ls,
        head_width=hw,
        head_length=hl,
        length_includes_head=True,
        alpha=alpha,
        zorder=zorder,
    )
    return quiv


def unicycle_patch(XY, yaw, color, size=1, lw=0.5):
    """
    A function to draw the unicycle robot patch.

    Attributes
    ----------
        XY: list
            position [X, Y] of the unicycle patch
        yaw: float
            unicycle heading
        size: float
            unicycle patch scale
        lw: float
            unicycle patch linewidth
    """
    Rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])

    apex = 45 * np.pi / 180  # 30 degrees apex angle
    b = np.sqrt(1) / np.sin(apex)
    a = b * np.sin(apex / 2)
    h = b * np.cos(apex / 2)

    z1 = size * np.array([a / 2, -h * 0.3])
    z2 = size * np.array([-a / 2, -h * 0.3])
    z3 = size * np.array([0, h * 0.6])

    z1 = Rot.dot(z1)
    z2 = Rot.dot(z2)
    z3 = Rot.dot(z3)

    verts = [
        (XY[0] + z1[1], XY[1] + z1[0]),
        (XY[0] + z2[1], XY[1] + z2[0]),
        (XY[0] + z3[1], XY[1] + z3[0]),
        (0, 0),
    ]

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)

    return patches.PathPatch(path, fc=color, lw=lw)


def alpha_cmap(cmap, alpha):
    """
    Scalar field color map
    """
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Define the alphas in the range from 0 to 1
    alphas = np.linspace(alpha, alpha, cmap.N)

    # Define the background as white
    BG = np.asarray(
        [
            1.0,
            1.0,
            1.0,
        ]
    )

    # Mix the colors with the background
    for i in range(cmap.N):
        my_cmap[i, :-1] = my_cmap[i, :-1] * alphas[i] + BG * (1.0 - alphas[i])

    # Create new colormap which mimics the alpha values
    my_cmap = ListedColormap(my_cmap)
    return my_cmap


# --------------------------------------------------------------------------------------
