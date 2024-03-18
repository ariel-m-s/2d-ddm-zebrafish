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

################
# GROUND TRUTH #
################

# A three-component Gaussian mixture is used to model the distribution
# of swimming angles. The mixture has three components: left, forward,
# and right. One Guassian mixture is used for each stimulus condition.
# The weights of the three components are relative to the baseline
# (static stimulus) condition, which are stored in the `ground_truth`
# array. The shape of the array is (n_stimuli, 4), where n_stimuli is
# the number of stimuli and 4 is the weights of the three components
# and the sum of the weights.

ground_truth = np.array(
    [
        [0.196, 0.608, 0.196, 1.000],
        [0.225, 0.970, 0.225, 1.419],
        [0.107, 0.838, 0.423, 1.368],
        [0.163, 0.554, 0.474, 1.191],
        [0.255, 0.417, 0.334, 1.005],
        [0.276, 0.402, 0.276, 0.955],
    ]
)

# The weights of the Gaussian mixture are proportional to the bout rate
# at every stimulus condition. The baseline bout rate is approximately
# 0.9 Hz. The weights are multiplied by the baseline bout rate to get
# the bout rate at every stimulus condition.

baseline_boutrate = 0.9
ground_truth *= baseline_boutrate


####################
# PARAMETER RANGES #
####################

# The parameter ranges are used to define the search space for the
# brute force optimization. The search space is defined by the minimum
# and maximum values of the parameters.

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

#############
# FUNCTIONS #
#############


def _compute_bout_rates(x: np.array, gamma: float, M: np.array, z: np.array, S: np.array, xi: np.array) -> np.array:
    """
    Compute the rates of left, forward, and right bouts (i.e., the 'three components')
    under different stimulus conditions.

    Args:
        x: The decision variable.
        gamma: The leak.
        M: The motor gain.
        z: The response bias.
        S: The sensory gain.
        xi: The stimulus bias.

    Returns:
        The rates of left, forward, and right bouts under different stimulus
        conditions. The shape of the array is (n_stimuli, 4), where n_stimuli
        is the number of stimuli and 4 is the bout rates of the three
        components and the sum of the bout rates.
    """
    # Initialize the array to store the bout rates at every stimulus condition.
    bout_rates = []

    # Compute the rates of the three components under the static stimulus
    # (baseline condition). The static stimulus has 0% coherence.

    # Compute the drift and the mean of the decision variable under the static stimulus.
    c__static = get_c(coherence=0)
    d__static = drift(c=c__static, S=S, xi=xi)
    x_mean__static = x_mean(drift=d__static, gamma=gamma)

    # Displace the decision variable by the mean under the static stimulus.
    x__static = x + x_mean__static
    alpha__static = get_alpha(x__static)

    # Compute the bout rates under the static stimulus.
    baseline_bout_rate = boutfield(x=x__static, alpha=alpha__static, M=M, z=z).mean(axis=1)
    baseline_bout_rate = np.append(baseline_bout_rate, baseline_bout_rate.sum())

    # Append the bout rates under the static stimulus to the array.
    bout_rates.append(baseline_bout_rate)

    # Compute the rates of the three components under different stimulus conditions.
    # The stimuli are presented at different angles (in degrees).

    for theta__degrees in [0, 45, 90, 135, 180]:
        # Compute the drift and the mean of the decision variable under the stimulus.
        c__stimulus = get_c(theta__degrees=theta__degrees, coherence=1)
        d__stimulus = drift(c=c__stimulus, S=S, xi=xi)
        x_mean__stimulus = x_mean(drift=d__stimulus, gamma=gamma)

        # Displace the decision variable by the mean under the stimulus.
        x__stimulus = x + x_mean__stimulus
        alpha__stimulus = get_alpha(x__stimulus)

        # Compute the bout rates under the stimulus.
        bout_rate = boutfield(x=x__stimulus, alpha=alpha__stimulus, M=M, z=z).mean(axis=1)
        bout_rate = np.append(bout_rate, bout_rate.sum())

        # Append the bout rates under the stimulus to the array.
        bout_rates.append(bout_rate)

    return np.array(bout_rates)


#######################
# OBJECTIVE FUNCTIONS #
#######################

# The objective function is used to calculate the loss for the given
# parameter combination. This function should take into account the
# relative increases and decreases in the weights of the Gaussian
# mixture components. Additionally, the function should consider the
# bout rates at every stimulus condition.

# In this case, the loss is the maximum absolute error between the
# bout rates at every stimulus condition and the ground truth.
# However, other objective functions can be used, depending on the
# subjectivity of the modeler and the specific qualitative and
# quantitative features that are being prioritized.


def _objective(params, x) -> float:
    """
    Calculate the loss for the given parameter combination.

    Args:
        params: The parameter combination.
        x: The decision variable samples.

    Returns:
        The loss for the given parameter combination.
    """
    # Unpack the parameters. The parameters are the leak (gamma), the
    # motor gains (m1, m2), the response bias (z2), the sensory gains
    # (s1, s2), and the stimulus bias (xi2). This are the parameters
    # that are being optimized, which correspond to a subset of the
    # parameters of the model. The other parameters are fixed to 0
    # because of some assumptions.
    gamma, m1, m2, z2, s1, s2, xi2 = params

    # Transform the parameters to the appropriate shapes, as needed by
    # the uncostrained version of the model.
    M = np.array([[m1, 0], [0, m2]])
    z = np.array([0, z2]).reshape(-1, 1)
    S = np.array([[s1, 0], [0, s2]])
    xi = np.array([0, xi2]).reshape(-1, 1)

    weights = _compute_bout_rates(x, gamma, M, z, S, xi)

    absolute_errors = np.abs(weights - ground_truth)
    return np.max(absolute_errors)


# Different cases can be considered to optimize the parameters of the
# model. The cases are defined by the fixed and free parameters. The
# cases define the following arguments:
# - The name of the case, which is used to save the top parameter
#   combinations to a CSV file.
# - The search space, which is the parameter space to search for the
#   best parameters that minimize the loss function.
# - The objective function, which is used to calculate the loss for
#   the given parameter combination. This function also defines the
#   values for the parameters that are out of the search space (fixed).

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

# case_3 = "case_3"

# search_space = ((m2_min, m2_max), (s2_min, s2_max), (xi2_min, xi2_max))


# def objective(params, x):
#     m, s, xi2 = params
#     return _objective([constants.GAMMA, m, m, 0, s, s, xi2], x)


#################################################################
# CASE 4: m1 = m2, z2 = 0, s1 = s2; free params: m, s1, s2, xi2 #
#################################################################

# name = "case_4"

# search_space = ((m2_min, m2_max), (s1_min, s1_max), (s2_min, s2_max), (xi2_min, xi2_max))


# def objective(params, x):
#     m, s1, s2, xi2 = params
#     return _objective([constants.GAMMA, m, m, 0, s1, s2, xi2], x)


########################################################
# CASE 5: m1 = m2, s1 = s2; free params: m, z2, s, xi2 #
########################################################

# name = "case_5"

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

    # A number of arguments are defined to control the optimization process:
    #     N_SAMPLES: The number of samples to use for the decision variable (x).
    #     N_SUBDIVISIONS: The number of subdivisions for each search space parameter.
    #     N_WORKERS: The number of workers to use for parallel processing (-1 to use all available workers).
    #     N_DECIMALS: The number of decimals to round the parameter combinations and the loss.
    #     N_TOP: The number of top parameter combinations to print and save.
    N_SAMPLES = int(1e5)
    N_SUBDIVISIONS = 21
    N_WORKERS = -1
    N_DECIMALS = 3
    N_TOP = 20

    # Calculate the number of parameters and the number of combinations.
    N_PARAMS = len(search_space)
    N_COMBINATIONS = N_SUBDIVISIONS**N_PARAMS

    # Print the optimization settings.
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

    # m = 1.100
    # z2 = 0.150
    # s = 0.300
    # params = [
    #     constants.GAMMA,
    #     np.array([[m, 0], [0, m]]),
    #     np.array([[0], [z2]]).reshape(-1, 1),
    #     np.array([[s, 0], [0, s]]),
    #     np.array([[0], [0]]).reshape(-1, 1),
    # ]
    # loss = objective([m, z2, s], x_samples)
    # print(loss)
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
        # Set the `finish` parameter to True to refine the best parameter combinations.
        finish=True,
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
