import numpy as np
from scipy.stats import norm

##################
# FISH CONSTANTS #
##################

BOUT_DURATION = 0.15
REFRACTORY_PERIOD = 0.25
MAX_BOUT_RATE = 1 / REFRACTORY_PERIOD

assert BOUT_DURATION <= REFRACTORY_PERIOD

######################
# BOUT DISTRIBUTIONS #
######################

LEFT_ANGLE_DISTRIBUTION = norm(loc=-31, scale=25)
FORWARD_ANGLE_DISTRIBUTION = norm(loc=0, scale=4)
RIGHT_ANGLE_DISTRIBUTION = norm(loc=31, scale=25)

########################
# SIMULATION TIME STEP #
########################

TIME_STEP = 0.01

######################
# DEFAULT PARAMETERS #
######################

# Motor gains.
M1 = M2 = 1
M21 = M12 = 0

# Response biases.
Z1 = Z2 = 0

# Sensory gains.
S1 = S2 = 0
S21 = S12 = 0

# Stimulus biases.
XI1 = XI2 = 0

# Leak / decay rate.
GAMMA = 2
GAMMA_RESET = GAMMA

####################################
# MODEL CONSTANTS (NOT PARAMETERS) #
####################################

PHI = np.deg2rad(60)
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

M = np.array([[M1, M21], [M12, M2]])
S = np.array([[S1, S21], [S12, S2]])
Z = np.array([Z1, Z2]).reshape(-1, 1)
XI = np.array([XI1, XI2]).reshape(-1, 1)
