import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import gmm
from constants import GAMMA, M1, M2, PHI, S1, S2, SIGMA, XI1, XI2, Z1, Z2

z = np.array([Z1, Z2])
bound_det = np.sqrt(M1 * M2)
bound_det_inv = 1 / bound_det

xi = np.array([XI1, XI2])


def boutfield(x, m1=M1, m2=M2, z1=Z1, z2=Z2, return_bool=False):
    x = np.array(x)

    angle = np.rad2deg(np.arctan2(x[:, 1], x[:, 0]))
    subthreshold = (m2 * x[:, 0] + z2) ** 2 + (m1 * x[:, 1] + z1) ** 2 < 1

    left = (angle < -PHI) & (~subthreshold)
    right = (angle > PHI) & (~subthreshold)

    high_bout_rate = ~subthreshold
    low_bout_rate = subthreshold

    fvs = np.zeros((len(x), 9))

    fvs[:, :6] = [-31.2, 0, 31.2, 25, 4, 25]

    fvs[:, -3:] = [0.0, 1.0, 0.0]
    fvs[left, -3:] = [1.0, 0.0, 0.0]
    fvs[right, -3:] = [0.0, 0.0, 1.0]

    fvs[high_bout_rate, -3:] *= 5
    fvs[low_bout_rate, -3:] *= 0

    if return_bool:
        return fvs, ~subthreshold

    return fvs


initial_stim = (0, 0)
