"""
This module contains the constants and parameters used in the simulation. The constants
define the behavior of the fish, such as the duration of a bout, the refractory period,
and the bout distributions. The parameters define the parameters of the model, such as
the leak, motor gains, sensory gains, and stimulus bias.

Example:
    To access the bout duration constant, use `constants.BOUT_DURATION`.
"""

import numpy as np
from scipy.stats import norm

##################
# FISH CONSTANTS #
##################

# The duration of a bout (in seconds). This is the time the fish spends swimming in
# a particular direction.
BOUT_DURATION = 0.15

# The refractory period (in seconds). This is the time the fish needs to wait before
# making another decision.
REFRACTORY_PERIOD = 0.25

# The maximum rate at which the fish can make decisions (in Hz). This is the inverse
# of the refractory period.
MAX_BOUT_RATE = 1 / REFRACTORY_PERIOD

# The bout duration must be less than or equal to the refractory period. This is
# because the fish cannot make a decision while it is still swimming in a particular
# direction. It must wait until the bout is over before making another decision.
assert BOUT_DURATION <= REFRACTORY_PERIOD

######################
# BOUT DISTRIBUTIONS #
######################

# The distributions for the angles of the bouts (in degrees).
LEFT_ANGLE_DISTRIBUTION = norm(loc=-31, scale=25)
FORWARD_ANGLE_DISTRIBUTION = norm(loc=0, scale=4)
RIGHT_ANGLE_DISTRIBUTION = norm(loc=31, scale=25)

########################
# SIMULATION TIME STEP #
########################

# The time step for the simulation (in seconds). This is the time step used in the
# simulation. The smaller the time step, the more accurate the simulation, but the
# longer it takes to run. Too large a time step can lead to artifacts in the
# simulation. Good values are between 0.005 and 0.020 seconds.

TIME_STEP = 0.010

######################
# DEFAULT PARAMETERS #
######################

# Leak (in 1/s). [Equation 5]
GAMMA = 2

# Leak when a decision is made.
# Make GAMMA_RESET = GAMMA to disable reset.
# Make GAMMA_RESET > GAMMA to enable reset.
GAMMA_RESET = GAMMA

# Motor gains and response bias. [Equation 19]

# Diagonal elements of the matrix M.
M1 = M2 = 1

# Off-diagonal elements of the matrix M.
M21 = M12 = 0

# Response bias.
Z1 = Z2 = 0

# Sensory gains and stimulus bias (in 1/s). [Equation 17]

# Diagonal elements of the matrix S.
S1 = S2 = 0

# Off-diagonal elements of the matrix S.
S21 = S12 = 0

# Stimulus bias.
XI1 = XI2 = 0

####################################
# MODEL CONSTANTS (NOT PARAMETERS) #
####################################

# Angle at which the decision categories are separated. [Equation 21]
PHI = np.deg2rad(60)

# Covariance matrix of the diffusion. [Equation 6]
SIGMA = np.array([[1, 0], [0, 1]])

##############
# PARAMETERS #
##############

# # CIRCULAR MODEL - DATASET A (just 100% coherence)
M1 = M2 = 1.050
Z1 = 0
Z2 = 0.250
S1 = S2 = 0.325

# # ELLIPTICAL MODEL - DATASET A (just 100% coherence)
# M1 = 1.100
# M2 = 0.950
# Z1 = 0
# Z2 = 0.275
# S1 = S2 = 0.300

# # CIRCULAR MODEL - DATASET B (100%, 50% and 25% coherence)
# M1 = M2 = 1.000
# Z1 = 0
# Z2 = 0.200
# S1 = S2 = 0.425

# # ELLIPTICAL MODEL - DATASET B (100%, 50% and 25% coherence)
# M1 = 1.050
# M2 = 0.850
# Z1 = 0
# Z2 = 0.300
# S1 = S2 = 0.400

###############################
# MULTIDIMENSIONAL PARAMETERS #
###############################

# Construct the matrices M, S, and the vectors Z and XI.
M = np.array([[M1, M21], [M12, M2]])
S = np.array([[S1, S21], [S12, S2]])
Z = np.array([Z1, Z2]).reshape(-1, 1)
XI = np.array([XI1, XI2]).reshape(-1, 1)
