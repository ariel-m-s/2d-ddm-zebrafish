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
- get_trial_data: Get the trial data for the given trial ID.
- load_trials: Load the trials from the given CSV file.

These functions are used to load and process the trial data in the analysis pipeline.

Example:
    # Load the trials metadata.
    trials = load_trials("data/organized/trials.csv")

    # Get the trial data for the trial ID 100.
    trial_data = get_trial_data("data/organized", 100)

The above example loads the trials metadata from the CSV file and gets the trial data for the trial ID 100.
"""

import os
import pickle

import pandas as pd


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
