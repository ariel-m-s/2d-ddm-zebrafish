from time import time as t

import numpy as np
from scipy.optimize import brute

PHI = 60
X_SIGMA = 1
Y_SIGMA = 1
GAMMA = 2.0


def boutfield(x, alpha, m1, m2, z2):
    x = np.array(x)

    subthreshold = (m1 * x[:, 0]) ** 2 + (m2 * x[:, 1] + z2) ** 2 < 1

    left = (alpha < -PHI) & (~subthreshold)
    right = (alpha > PHI) & (~subthreshold)

    high_bout_rate = ~subthreshold
    low_bout_rate = subthreshold

    fvs = np.zeros((len(x), 9))

    fvs[:, :6] = [-30, 0, 30, 0.1, 0.1, 0.1]
    fvs[:, -3:] = [0.0, 1.0, 0.0]
    fvs[left, -3:] = [1.0, 0.0, 0.0]
    fvs[right, -3:] = [0.0, 0.0, 1.0]

    fvs[high_bout_rate, -3:] *= 5
    fvs[low_bout_rate, -3:] *= 0

    return fvs


def _compute_bout_rates(x, m1, m2, z2, s1, s2, xi2):
    xi = np.array([0, xi2])
    x_no_stim = x + (xi / GAMMA)
    alpha = np.rad2deg(np.arctan2(x_no_stim[:, 0], x_no_stim[:, 1]))

    baseline_pi = boutfield(x_no_stim, alpha, m1, m2, z2).mean(axis=0)[-3:]
    baseline_sumpi = baseline_pi.sum()

    print(baseline_pi / baseline_sumpi * 100)
    print((baseline_pi / baseline_sumpi).sum() * 100)

    bout_rates = []
    c = 1

    for theta in [0, 45, 90, 135, 180]:
        d1 = s1 * c * np.sin(np.deg2rad(theta))
        d2 = s2 * c * np.cos(np.deg2rad(theta))
        drift = np.array([d1, d2])

        x_with_drift = x + (drift / GAMMA)
        alpha = np.rad2deg(np.arctan2(x_with_drift[:, 0], x_with_drift[:, 1]))

        pi = boutfield(x_with_drift, alpha, m1, m2, z2).mean(axis=0)[-3:]
        bout_rates.append(pi.sum())

        print(pi / baseline_sumpi * 100)
        print((pi / baseline_sumpi).sum() * 100)

        # print(drift)
        # print(alpha.mean())
        # print(theta)
        # print(pi)
        # print()

    return np.array(bout_rates)


def compute_pis(x, m1, m2, z2, s1, s2, xi2):
    xi = np.array([0, xi2])

    x__static = x + (xi / GAMMA)
    alpha__static = np.rad2deg(np.arctan2(x__static[:, 0], x__static[:, 1]))

    baseline_pi = boutfield(x__static, alpha__static, m1, m2, z2).mean(axis=0)[-3:]
    baseline_pi = np.append(baseline_pi, baseline_pi.sum())

    pis = []
    pis.append(baseline_pi)

    for theta in [0, 45, 90, 135, 180]:
        c1 = np.sin(np.deg2rad(theta))
        c2 = np.cos(np.deg2rad(theta))
        drift = np.array([s1 * c1, s2 * c2]) + xi

        x__stim = x + (drift / GAMMA)
        alpha__stim = np.rad2deg(np.arctan2(x__stim[:, 0], x__stim[:, 1]))

        pi = boutfield(x__stim, alpha__stim, m1, m2, z2).mean(axis=0)[-3:]
        pi = np.append(pi, pi.sum())

        pis.append(pi)

    return np.array(pis)


# DATASET A:
# baseline_boutrate = 0.9
# ground_truth = (
#     np.array(
#         [
#             [0.196, 0.608, 0.196, 1.000],
#             [0.225, 0.970, 0.225, 1.419],
#             [0.107, 0.838, 0.423, 1.368],
#             [0.163, 0.554, 0.474, 1.191],
#             [0.255, 0.417, 0.334, 1.005],
#             [0.276, 0.402, 0.276, 0.955],
#         ]
#     )
#     * baseline_boutrate
# )

# DATASET B:
baseline_boutrate = 0.7
ground_truth = (
    np.array(
        [
            [0.183, 0.633, 0.183, 1.000],
            [0.302, 1.050, 0.302, 1.653],
            [0.096, 0.916, 0.562, 1.574],
            [0.150, 0.583, 0.581, 1.315],
            [0.228, 0.423, 0.404, 1.055],
            [0.287, 0.406, 0.287, 0.980],
        ]
    )
    * baseline_boutrate
)

####################
# PARAMETER RANGES #
####################

m1_min = 0.5
m1_max = 1.5

m2_min = 0.5
m2_max = 1.5

z2_min = 0.0
z2_max = 0.5

s1_min = 0.0
s1_max = 0.5

s2_min = 0.0
s2_max = 0.5

xi2_min = 0.0
xi2_max = 0.5

#######################
# OBJECTIVE FUNCTIONS #
#######################

def _objective(params, x):
    pis = compute_pis(x, *params)
    absolute_errors = np.abs(pis - ground_truth)
    return np.max(absolute_errors)


#########################################################
# CASE 1: m1 = m2, s1 = s2, xi2 = 0; free params: m, z2 #
#########################################################

name = "case_1"

search_space = ((m2_min, m2_max), (z2_min, z2_max), (s2_min, s2_max))


def objective(params, x):
    m, z2, s = params
    return _objective([m, m, z2, s, s, 0], x)


########################################################
# CASE 2: s1 = s2, xi2 = 0; free params: m1, m2, z2, s #
########################################################

# name = "case_2"

# search_space = ((m1_min, m1_max), (m2_min, m2_max), (z2_min, z2_max), (s2_min, s2_max))


# def objective(params, x):
#     m1, m2, z2, s = params
#     return _objective([m1, m2, z2, s, s, 0], x)


############################################################
# CASE 3: m1 = m2, z2 = 0, s1 = s2; free params: m, s, xi2 #
############################################################

# search_space = ((m2_min, m2_max), (s2_min, s2_max), (xi2_min, xi2_max))


# def objective(params, x):
#     m, s, xi2 = params
#     return _objective([m, m, 0, s, s, xi2], x)


#################################################################
# CASE 4: m1 = m2, z2 = 0, s1 = s2; free params: m, s1, s2, xi2 #
#################################################################

# search_space = ((m2_min, m2_max), (s1_min, s1_max), (s2_min, s2_max), (xi2_min, xi2_max))


# def objective(params, x):
#     m, s1, s2, xi2 = params
#     return _objective([m, m, 0, s1, s2, xi2], x)


########################################################
# CASE 5: m1 = m2, s1 = s2; free params: m, z2, s, xi2 #
########################################################

# z2_min = -0.5
# z2_max = 1.5

# xi2_min = -0.5
# xi2_max = 1.5

# search_space = ((m2_min, m2_max), (z2_min, z2_max), (s2_min, s2_max), (xi2_min, xi2_max))


# def objective(params, x):
#     m, z2, s, xi2 = params
#     return _objective([m, m, z2, s, s, xi2], x)


########################################################
# CASE 6: m1 = m2, xi2 = 0; free params: m, z2, s1, s2 #
########################################################

# name = "case_6"

# search_space = ((m2_min, m2_max), (z2_min, z2_max), (s1_min, s1_max), (s2_min, s2_max))


# def objective(params, x):
#     m, z2, s1, s2 = params
#     return _objective([m, m, z2, s1, s2, 0], x)


########################################################
# CASE 7: s1 = s2, z2 = 0; free params: m1, m2, s, xi2 #
########################################################

# xi2_min = 0.0
# xi2_max = 1.0

# name = "case_7"

# search_space = ((m1_min, m1_max), (m2_min, m2_max), (s2_min, s2_max), (xi2_min, xi2_max))


# def objective(params, x):
#     m1, m2, s, xi2 = params
#     return _objective([m1, m2, 0, s, s, xi2], x)


###########################
# END OBJECTIVE FUNCTIONS #
###########################

if __name__ == "__main__":
    ##################
    # INITIALIZATION #
    ##################

    # SET SEED FOR CONSISTENT RESULTS.
    np.random.seed(0)

    # SET CONFIGURATION.
    FIT = True
    N_SAMPLES = int(1e5)
    N_SUBDIVISIONS = 21
    N_PARAMS = len(search_space)
    N_COMBINATIONS = N_SUBDIVISIONS**N_PARAMS
    N_WORKERS = -1
    N_DECIMALS = 3
    N_TOP = 20

    # PRINT CONFIGURATION.
    print(f"\nGAMMA: {GAMMA}")
    print(f"\nX SAMPLES: {N_SAMPLES:.2e}")
    print(f"SUBDIVISIONS: {N_SUBDIVISIONS}")
    print(f"PARAMS: {N_PARAMS}")
    print(f"COMBINATIONS: {N_COMBINATIONS:.2e}")
    print(f"WORKERS: {N_WORKERS}")
    print(f"DECIMALS: {N_DECIMALS}")
    print(f"TOP: {N_TOP}\n")

    # SAMPLE DECISION VARIABLE (X).
    # SEED NEEDED FOR CONSISTENT RESULTS.
    mean = np.zeros((2,))
    cov = np.array([[X_SIGMA**2, 0], [0, Y_SIGMA**2]]) / (2 * GAMMA)
    x_samples = np.random.multivariate_normal(mean, cov, N_SAMPLES)

    # pis = compute_pis(x_samples, m1=1, m2=1.1, z2=0.2, s1=0.25, s2=0.25, xi2=0)
    # print(pis.round(2))

    # CASE 1: m, z2, s
    # params = [1.050, 1.050, 0.250, 0.325, 0.325, 0]

    # CASE 2: m1, m2, z2, s
    # params = [1.100, 1.000, 0.275, 0.275, 0.275, 0]

    # CASE 2 considering total bout rate: m1, m2, z2, s
    # params = [1.100, 0.950, 0.275, 0.300, 0.300, 0]

    # CASE 3: m, s, xi2
    # params = [1.100, 1.100, 0, 0.250, 0.250, 0.275]

    # print(_objective(params, x_samples))
    # pis = compute_pis(x_samples, *params)
    # baseline_sum = pis[0][3]
    # print((pis / baseline_sum).round(2))

    if FIT:
        print("\nFitting...\n")
        t0 = t()

        result = brute(
            objective,
            search_space,
            Ns=N_SUBDIVISIONS,
            full_output=True,
            finish=None,
            workers=N_WORKERS,
            args=(x_samples,),
        )

        ##########################
        # PRINT AND SAVE RESULTS #
        ##########################

        indices_flat = np.argpartition(result[3].flatten(), N_TOP)[:N_TOP]
        indices_flat = indices_flat[np.argsort(result[3].flatten()[indices_flat])]
        indices = np.unravel_index(indices_flat, result[3].shape)

        top_params = []

        for i in range(N_TOP):
            index = tuple(indices[j][i] for j in range(N_PARAMS))
            params = [result[2][j][index] for j in range(N_PARAMS)]
            loss = result[3][index]

            # print params and loss nicely and round them to N_DECIMALS decimal places.
            print(", ".join([f"{param:.{N_DECIMALS}f}" for param in params]), f"loss: {loss:.{N_DECIMALS}f}")

            top_params.append([*params, loss])

        with open(f"top_params_{name}.csv", "w") as f:
            for params in top_params:
                f.write(", ".join([f"{param:.{N_DECIMALS}f}" for param in params]) + "\n")

        print(f"\nFitting took {t() - t0:.{N_DECIMALS}f} seconds.")

    else:
        print("\nNot fitting.")
