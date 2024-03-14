"""
This script fits the model to the data using brute force optimization.
The script uses the `brute` function from `scipy.optimize` to search the
parameter space for the best parameters that minimize the loss function.
"""

from time import time as t

import numpy as np
from scipy.optimize import brute

import constants
from decision_making import boutfield, get_alpha
from evidence_accumulation import drift, get_c, x_cov, x_mean


def compute_weights(x: np.array, gamma: float, M: np.array, z: np.array, S: np.array, xi: np.array) -> np.array:
    """
    Compute the weights of the three-component Gaussian mixture under
    different stimulus conditions.

    Args:
        x: The decision variable.
        gamma: The leak.
        M: The motor gain.
        z: The response bias.
        S: The sensory gain.
        xi: The stimulus bias.

    Returns:
        The weights of the three-component Gaussian mixture under different
        stimulus conditions. The shape of the array is (n_stimuli, 4), where
        n_stimuli is the number of stimuli and 4 is the weights of the three
        components and the sum of the weights.
    """
    # Compute the weights of the three-component Gaussian mixture under
    # the static stimulus (baseline condition). The static stimulus has
    # zero coherence.

    c__static = get_c(coherence=0)
    d__static = drift(c=c__static, S=S, xi=xi)
    x_mean__static = x_mean(drift=d__static, gamma=gamma)

    x__static = x + x_mean__static
    alpha__static = get_alpha(x__static)

    baseline_weight = boutfield(x=x__static, alpha=alpha__static, M=M, z=z).mean(axis=1)
    baseline_weight = np.append(baseline_weight, baseline_weight.sum())

    weights = []
    weights.append(baseline_weight)

    # Compute the weights of the three-component Gaussian mixture under
    # different stimulus conditions. The stimuli are presented at different
    # angles and coherences. These angles are in degrees and the coherences
    # range from 0 to 1, not 0% to 100%.

    for theta__degrees in [0, 45, 90, 135, 180]:
        c__stimulus = get_c(theta__degrees=theta__degrees, coherence=1)
        d__stimulus = drift(c=c__stimulus, S=S, xi=xi)
        x_mean__stimulus = x_mean(drift=d__stimulus, gamma=gamma)

        # print(f"theta: {theta__degrees}")
        # print(f"c: {c__stimulus}")

        x__stimulus = x + x_mean__stimulus
        alpha__stimulus = get_alpha(x__stimulus)

        weight = boutfield(x=x__stimulus, alpha=alpha__stimulus, M=M, z=z).mean(axis=1)
        weight = np.append(weight, weight.sum())

        weights.append(weight)

    return np.array(weights)


################
# GROUND TRUTH #
################

baseline_boutrate = 0.9

ground_truth = (
    np.array(
        [
            [0.196, 0.608, 0.196, 1.000],
            [0.225, 0.970, 0.225, 1.419],
            [0.107, 0.838, 0.423, 1.368],
            [0.163, 0.554, 0.474, 1.191],
            [0.255, 0.417, 0.334, 1.005],
            [0.276, 0.402, 0.276, 0.955],
        ]
    )
    * baseline_boutrate
)

####################
# PARAMETER RANGES #
####################

gamma_min = 1.5
gamma_max = 2.5

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


def _objective(params, x) -> float:
    gamma, m1, m2, z2, s1, s2, xi2 = params

    M = np.array([[m1, 0], [0, m2]])
    z = np.array([0, z2]).reshape(-1, 1)
    S = np.array([[s1, 0], [0, s2]])
    xi = np.array([0, xi2]).reshape(-1, 1)

    weights = compute_weights(x, gamma, M, z, S, xi)

    absolute_errors = np.abs(weights - ground_truth)
    return np.max(absolute_errors)


#########################################################
# CASE 1: m1 = m2, s1 = s2, xi2 = 0; free params: m, z2 #
#########################################################

name = "case_1"

search_space = ((m2_min, m2_max), (z2_min, z2_max), (s2_min, s2_max))


def objective(params, x):
    m, z2, s = params
    return _objective([constants.GAMMA, m, m, z2, s, s, 0], x)


########################################################
# CASE 2: s1 = s2, xi2 = 0; free params: m1, m2, z2, s #
########################################################

# name = "case_2"

# search_space = ((m1_min, m1_max), (m2_min, m2_max), (z2_min, z2_max), (s2_min, s2_max))


# def objective(params, x):
#     m1, m2, z2, s = params
#     return _objective([constants.GAMMA, m1, m2, z2, s, s, 0], x)


############################################################
# CASE 3: m1 = m2, z2 = 0, s1 = s2; free params: m, s, xi2 #
############################################################

# search_space = ((m2_min, m2_max), (s2_min, s2_max), (xi2_min, xi2_max))


# def objective(params, x):
#     m, s, xi2 = params
#     return _objective([constants.GAMMA, m, m, 0, s, s, xi2], x)


#################################################################
# CASE 4: m1 = m2, z2 = 0, s1 = s2; free params: m, s1, s2, xi2 #
#################################################################

# search_space = ((m2_min, m2_max), (s1_min, s1_max), (s2_min, s2_max), (xi2_min, xi2_max))


# def objective(params, x):
#     m, s1, s2, xi2 = params
#     return _objective([constants.GAMMA, m, m, 0, s1, s2, xi2], x)


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
#     return _objective([constants.GAMMA, m, m, z2, s, s, xi2], x)


########################################################
# CASE 6: m1 = m2, xi2 = 0; free params: m, z2, s1, s2 #
########################################################

# name = "case_6"

# search_space = ((m2_min, m2_max), (z2_min, z2_max), (s1_min, s1_max), (s2_min, s2_max))


# def objective(params, x):
#     m, z2, s1, s2 = params
#     return _objective([constants.GAMMA, m, m, z2, s1, s2, 0], x)


########################################################
# CASE 7: s1 = s2, z2 = 0; free params: m1, m2, s, xi2 #
########################################################

# xi2_min = 0.0
# xi2_max = 1.0

# name = "case_7"

# search_space = ((m1_min, m1_max), (m2_min, m2_max), (s2_min, s2_max), (xi2_min, xi2_max))


# def objective(params, x):
#     m1, m2, s, xi2 = params
#     return _objective([constants.GAMMA, m1, m2, 0, s, s, xi2], x)


###########################
# END OBJECTIVE FUNCTIONS #
###########################

if __name__ == "__main__":
    ##################
    # INITIALIZATION #
    ##################

    # Set the seed for reproducibility. The seed is set here to ensure
    # that the same random samples are used for each optimization run.
    np.random.seed(0)

    N_SAMPLES = int(1e5)
    N_SUBDIVISIONS = 21
    N_PARAMS = len(search_space)
    N_COMBINATIONS = N_SUBDIVISIONS**N_PARAMS
    N_WORKERS = -1
    N_DECIMALS = 3
    N_TOP = 20

    print(f"\nSAMPLES: {N_SAMPLES:.2e}")
    print(f"SUBDIVISIONS: {N_SUBDIVISIONS}")
    print(f"PARAMS: {N_PARAMS}")
    print(f"COMBINATIONS: {N_COMBINATIONS:.2e}")
    print(f"WORKERS: {N_WORKERS}")
    print(f"DECIMALS: {N_DECIMALS}")
    print(f"TOP: {N_TOP}\n")

    # Sample the decision variable from a bivariate Gaussian distribution.
    # The random seed is needed to ensure that the same samples are used
    # for each optimization run.
    x_samples_mean = np.zeros((2,))
    x_samples_cov = x_cov(sigma=constants.SIGMA, gamma=constants.GAMMA)
    x_samples = np.random.multivariate_normal(x_samples_mean, x_samples_cov, N_SAMPLES).T

    # m = 1
    # s = 1
    # result = compute_weights(x_samples, constants.GAMMA, np.array([[m, 0], [0, m]]), np.array([[0], [0]]), np.array([[s, 0], [0, s]]), np.array([[0], [0]]).reshape(-1, 1))
    # np.set_printoptions(precision=2, suppress=True)
    # print(result)
    # exit()

    print("\nFitting...\n")
    t0 = t()

    # Run the brute force optimization. The `brute` function searches the
    # parameter space for the best parameters that minimize the loss function.
    # The `args` parameter is used to pass the decision variable samples to
    # the loss function.
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

    # Print the top N_TOP parameter combinations and their corresponding loss.
    # The parameter combinations are sorted by the loss in ascending order.

    indices_flat = np.argpartition(result[3].flatten(), N_TOP)[:N_TOP]
    indices_flat = indices_flat[np.argsort(result[3].flatten()[indices_flat])]
    indices = np.unravel_index(indices_flat, result[3].shape)

    top_params = []

    for i in range(N_TOP):
        index = tuple(indices[j][i] for j in range(N_PARAMS))
        params = [result[2][j][index] for j in range(N_PARAMS)]
        loss = result[3][index]

        print(", ".join([f"{param:.{N_DECIMALS}f}" for param in params]), f"loss: {loss:.{N_DECIMALS}f}")

        top_params.append([*params, loss])

    # Save the top N_TOP parameter combinations to a CSV file. The CSV file
    # contains the parameter combinations and their corresponding loss. The
    # file name is `top_params_{name}.csv`, where `name` is the name of the case.
    with open(f"top_params_{name}.csv", "w") as f:
        for params in top_params:
            f.write(", ".join([f"{param:.{N_DECIMALS}f}" for param in params]) + "\n")

    print(f"\nDone! Elapsed time: {t() - t0:.2f} seconds.")
