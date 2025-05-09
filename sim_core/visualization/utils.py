"""
"""

#######################################################################################

def kw_def_arrow(scale):
    return {
        "s": scale,
        "lw": 2 * scale ** (1 / 5),
        "hw": 0.1 * scale ** (1 / 5),
        "hl": 0.2 * scale ** (1 / 5),
    }

# SIMULATOR MATH UTILS

# ----------------------------------------------------------------------
# Centroid and ascending direction estimation dynamics
# ----------------------------------------------------------------------

def dyn_centroid_estimation(xhat, t, Lb, p, k=1):
    xhat_dt = -k * (Lb.dot(xhat) - Lb.dot(p))
    return xhat_dt


def dyn_mu_estimation(mu, t, Lb, k=1):
    mu_dt = -k * (Lb.dot(mu))
    return mu_dt

#######################################################################################
