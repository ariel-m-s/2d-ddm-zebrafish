"""
This module contains utility functions for loading the experimental or simulated data.
The trial data is saved in pickle format, and the trials metadata is saved in CSV format.

########################################################################################
DISCLAIMER:
This module assumes the data is saved in the organized structure, not the original (raw)
structure. To convert experimental data from the original structure to the organized
structure, use the `organize_data.py` script. Simulated data is already saved in the
organized structure, so no conversion is needed.
########################################################################################

The following functions are defined:
- load_trials: Load the trials from the given CSV file.
- get_trial_data: Get the trial data for the given trial ID.
- extract_bouts: Extract the bout data from the given directory to be used in the analysis.

These functions are used to load and process the trial data in the analysis pipeline.

Example:
    # Load the trials metadata.
    trials = load_trials("data/organized/trials.csv")

    # Get the trial data for the trial ID 100.
    trial_data = get_trial_data("data/organized", 100)

    # OR extract the bout data from the given directory to be used in the analysis.
    angles__trials, times__trials, angles__all, times__all = extract_bouts("data/organized", 0)

The above example loads the trials metadata from the CSV file and gets the trial data for the trial ID 100.
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiment import (
    STIM_IDS_EXPERIMENT_0,
    STIM_IDS_EXPERIMENT_1,
    STIM_IDS_EXPERIMENT_2,
)

#############
# FUNCTIONS #
#############


def load_trials(trials_path):
    """
    Load the trials from the given CSV file. The CSV file contains the trial data.

    Args:
        trials_path (str): The path to the CSV file containing the trial data.

    Returns:
        pd.DataFrame: The trials data.
    """
    trials = pd.read_csv(trials_path, index_col="trial_id")

    return trials


def get_trial_data(data_path, trial_id):
    """
    Get the trial data for the given trial ID. The trial data is saved in pickle format,
    and the file name is the trial ID.

    Args:
        data_path (str): The path to the directory containing the trial data.
        trial_id (int): The trial ID.

    Returns:
        dict: The trial data.
    """
    # Construct the path to the trial data file. The file name is the trial ID.
    trial_path = os.path.join(data_path, f"{trial_id}.dat")

    # Load the trial data from the file. The trial data is saved in pickle format.
    with open(trial_path, "rb") as f:
        data = pickle.load(f)

    return data


def extract_bouts(directory_name: str, experiment_idx: int):
    """
    Extract the bout data from the given directory. The bout data is the turn angle and start time
    for each bout. The data is extracted for each stimulus angle and coherence level.

    Args:
        directory_name (str): The name of the directory containing the trial data.
        experiment_idx (int): The index of the experiment.

    Returns:
        dict: The turn angles for each stimulus angle and coherence level.
        dict: The start times for each stimulus angle and coherence level.
    """
    # Load the trials metadata.
    trials = load_trials(os.path.join(directory_name, "trials.csv"))

    # Initialize the data structures to store the turn angles and start times for each stimulus
    # angle and coherence.
    angles__trials = defaultdict(lambda: defaultdict(list))
    times__trials = defaultdict(lambda: defaultdict(list))

    for trial_id in tqdm(trials.index):
        # Get the trial data for the given trial ID.
        data = get_trial_data(data_path=directory_name, trial_id=trial_id)

        # There are three experiments, and each experiment has a different number of coherence levels.
        # The stimulus IDs are different for each experiment, so the stimulus IDs are filtered based
        # on the experiment index.

        data_length = len(data.keys())

        if experiment_idx == 0:
            if data_length != (9 * 3):
                continue
            STIM_IDS = STIM_IDS_EXPERIMENT_0
        elif experiment_idx == 1:
            if data_length != (17 * 3):
                continue
            STIM_IDS = STIM_IDS_EXPERIMENT_1
        elif experiment_idx == 2:
            if data_length != (25 * 3):
                continue
            STIM_IDS = STIM_IDS_EXPERIMENT_2

        # Filter the stimulus IDs based on the coherence level. The stimulus IDs are stored in a
        # nested dictionary with the following structure:
        # {
        #     angle: stimulus_id,
        #     ...
        # }
        STIM_IDS_25 = {(k and k[0]): v for (k, v) in STIM_IDS.items() if (k is None or k[1] == 0.25)}
        STIM_IDS_50 = {(k and k[0]): v for (k, v) in STIM_IDS.items() if (k is None or k[1] == 0.5)}
        STIM_IDS_100 = {(k and k[0]): v for (k, v) in STIM_IDS.items() if (k is None or k[1] == 1)}

        for (coherence, stim_ids) in [(0.25, STIM_IDS_25), (0.5, STIM_IDS_50), (1, STIM_IDS_100)]:
            # Check if there are exactly 9 stimulus angles for the given coherence level. If not,
            # skip the trial. This filters out errors in the stimulus presentation and ensures that
            # the data is consistent.
            if len(stim_ids.keys()) != 9:
                continue

            for stim in stim_ids.keys():
                # Get the stimulus ID for the given stimulus angle. The stimulus ID is used to
                # identify the stimulus in the trial data. The stimulus ID is a string that
                # represents the stimulus angle and coherence level.
                stim_id = stim_ids[stim]

                # Extract the bout start and end data for the given stimulus ID.

                bout_start_data = data[f"bouts_start_stimulus_{stim_id}"]
                bout_end_data = data[f"bouts_end_stimulus_{stim_id}"]

                start_time = bout_start_data["timestamp"]
                end_time = bout_end_data["timestamp"]
                angle_before_bout = bout_start_data["fish_accumulated_orientation_lowpass"]
                angle_after_bout = bout_end_data["fish_accumulated_orientation_lowpass"]

                # Filter the data based on the error code. The error code indicates if there was an
                # error in the data collection. If there was an error, the data is not used in the analysis.

                error_mask = (bout_start_data["errorcode"] == 0) & (bout_end_data["errorcode"] == 0)

                angle_before_bout = angle_before_bout[error_mask]
                angle_after_bout = angle_after_bout[error_mask]
                start_time = start_time[error_mask]
                end_time = end_time[error_mask]

                # Calculate the turn angle for the bout. The turn angle is the difference between the
                # angle after the bout and the angle before the bout.
                turn_angle = angle_after_bout - angle_before_bout

                # Append the turn angle and start time to the data structures.

                angles__trials[stim][coherence].append(turn_angle)
                times__trials[stim][coherence].append(start_time)

                if stim in (-45, -90, -135):
                    angles__trials[f"abs_{-stim}"][coherence].append(-turn_angle)
                    times__trials[f"abs_{-stim}"][coherence].append(start_time)

                elif stim in (45, 90, 135):
                    angles__trials[f"abs_{stim}"][coherence].append(turn_angle)
                    times__trials[f"abs_{stim}"][coherence].append(start_time)

    # Concatenate the turn angles and start times for each stimulus angle and coherence level into
    # a single array.

    angles__all = defaultdict(dict)
    times__all = defaultdict(dict)

    for coherence in [0.25, 0.5, 1]:
        for stim in [None, 0, "abs_45", "abs_90", "abs_135", 180]:
            if len(angles__trials[stim][coherence]) == 0:
                angles__all[stim][coherence] = np.array([])
                times__all[stim][coherence] = np.array([])
            else:
                angles__all[stim][coherence] = np.concatenate(angles__trials[stim][coherence])
                times__all[stim][coherence] = np.concatenate(times__trials[stim][coherence])

    return angles__trials, times__trials, angles__all, times__all


if __name__ == "__main__":
    directory_name = "../behavior/free_swimming_8fish_random_dot_kinematogram_data/org"

    print("Experiment 0")
    experiment_idx = 0
    angles__trials, times__trials, angles__all, times__all = extract_bouts(directory_name, experiment_idx)

    for stim in angles__all.keys():
        for coherence in angles__all[stim].keys():
            print(f"Stimulus: {stim}, Coherence: {coherence}")
            print(angles__all[stim][coherence])
            print(times__all[stim][coherence])
            print()

    print("Experiment 1")
    experiment_idx = 1
    angles__trials, times__trials, angles__all, times__all = extract_bouts(directory_name, experiment_idx)

    for stim in angles__all.keys():
        for coherence in angles__all[stim].keys():
            print(f"Stimulus: {stim}, Coherence: {coherence}")
            print(angles__all[stim][coherence])
            print(times__all[stim][coherence])
            print()

    print("Experiment 2")
    experiment_idx = 2
    angles__trials, times__trials, angles__all, times__all = extract_bouts(directory_name, experiment_idx)

    for stim in angles__all.keys():
        for coherence in angles__all[stim].keys():
            print(f"Stimulus: {stim}, Coherence: {coherence}")
            print(angles__all[stim][coherence])
            print(times__all[stim][coherence])
            print()
