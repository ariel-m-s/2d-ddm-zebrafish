"""
This module contains the experiment parameters. The experiment parameters define the
stimulus structure and the stimulus IDs for each experiment. The stimulus structure
defines the timing and duration of the stimuli. The stimulus IDs define the stimulus
IDs for each stimulus angle and coherence level.

Example:
    To access the stimulus structure for experiment 0, use `experiment.STIMULUS_STRUCTURE`.

    To access the stimulus IDs for experiment 0, use `experiment.STIM_IDS_EXPERIMENT_0`.

    To access the stimulus IDs for experiment 1, use `experiment.STIM_IDS_EXPERIMENT_1`.

    To access the stimulus IDs for experiment 2, use `experiment.STIM_IDS_EXPERIMENT_2`.
"""

# Sturcture of STIMULUS_IDS:
# {
#     (theta, coherence): stimulus_id,
#     ...
# }

# The stimulus angles are in degrees, not radians.
# The stimulus strength (coherence) ranges from 0 to 1, not 0% to 100%.

STIM_IDS_EXPERIMENT_0 = {
    (90, 1): "000",
    (-90, 1): "001",
    (135, 1): "002",
    (-45, 1): "003",
    (180, 1): "004",
    (0, 1): "005",
    (-135, 1): "006",
    (45, 1): "007",
    None: "008",
}

STIM_IDS_EXPERIMENT_1 = {
    (90, 1): "000",
    (-90, 1): "001",
    (135, 1): "002",
    (-45, 1): "003",
    (180, 1): "004",
    (0, 1): "005",
    (-135, 1): "006",
    (45, 1): "007",
    (90, 0.25): "008",
    (-90, 0.25): "009",
    (135, 0.25): "010",
    (-45, 0.25): "011",
    (180, 0.25): "012",
    (0, 0.25): "013",
    (-135, 0.25): "014",
    (45, 0.25): "015",
    None: "016",
}

STIM_IDS_EXPERIMENT_2 = {
    (90, 1): "000",
    (-90, 1): "001",
    (135, 1): "002",
    (-45, 1): "003",
    (180, 1): "004",
    (0, 1): "005",
    (-135, 1): "006",
    (45, 1): "007",
    (90, 0.5): "008",
    (-90, 0.5): "009",
    (135, 0.5): "010",
    (-45, 0.5): "011",
    (180, 0.5): "012",
    (0, 0.5): "013",
    (-135, 0.5): "014",
    (45, 0.5): "015",
    (90, 0.25): "016",
    (-90, 0.25): "017",
    (135, 0.25): "018",
    (-45, 0.25): "019",
    (180, 0.25): "020",
    (0, 0.25): "021",
    (-135, 0.25): "022",
    (45, 0.25): "023",
    None: "024",
}

STIMULUS_IDS = STIM_IDS_EXPERIMENT_2

# Structure of STIMULUS_STRUCTURE:
# [
#     (stimulus_is_on, duration_in_seconds),
#     ...
# ]

# The stimulus durations are in seconds.
# The stimulus is either ON or OFF (True or False).

STIMULUS_STRUCTURE = [
    (False, 5),
    (True, 10),
    (False, 5),
]
