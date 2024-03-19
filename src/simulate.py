"""
Simulate the behavior of fish in an experiment. The fish are simulated using a
two-dimensional drift-diffusion model. The fish are presented with visual stimuli
at different angles and coherences (strengths).
"""

import datetime as dt
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from time import time as t

import numpy as np
from tqdm import tqdm

import constants
from decision_making import sample_bout_angles
from evidence_accumulation import get_c
from experiment import (
    STIM_IDS_EXPERIMENT_0,
    STIM_IDS_EXPERIMENT_1,
    STIM_IDS_EXPERIMENT_2,
    STIMULUS_STRUCTURE,
)
from model import Simulator


def simulate_fish(n_trials: int, fish_num: int, stimulus_ids: dict, stimulus_structure: list, random_seed: int = None):
    """
    Simulate a fish undergoing multiple trials.

    Args:
        n_trials: The number of trials.
        fish_num: The fish number.
        random_seed: The random seed.
    """
    # Set the random seed. If the random seed is not provided, use the fish number.
    # This ensures that each fish has a unique random seed.
    random_seed = random_seed or fish_num
    np.random.seed(random_seed)

    # The trial indices are unique to each fish.
    trial_idxs = np.arange(n_trials) + (n_trials * fish_num) + 1

    # The simulator is used to simulate the fish's behavior.
    simulator = Simulator()

    # The initial orientation of the fish is random. The orientation is in degrees.
    initial_orientation = np.random.uniform(-180, 180)

    # Simulate each trial.
    for trial_idx in tqdm(trial_idxs, desc=f"Fish/CPU {fish_num}"):
        # Randomize the order of the stimuli. This ensures that the fish is not
        # biased by the order of the stimuli.
        randomized_stimuli = np.random.permutation(list(stimulus_ids.keys()))

        # The trial data is stored in a dictionary, which has the same structure
        # as the data that is collected in the experiments with real fish.
        trial_data = defaultdict(dict)

        # Simulate each stimulus in the randomized order.
        for (theta__degrees, coherence) in randomized_stimuli:
            # Simulate each part of the stimulus structure. The stimulus structure
            # is a list of tuples. Each tuple contains a boolean that indicates
            # if the stimulus is on or off and the duration of the stimulus.
            for (stimulus_is_on, duration_in_seconds) in stimulus_structure:
                # The duration of the stimulus is converted from seconds to steps.
                duration_in_steps = int(duration_in_seconds / simulator.time_step)

                # The stimulus vector (c) is the visual stimulus that the fish perceives. The
                # stimulus vector is a two-dimensional array that represents the stimulus in
                # Cartesian coordinates. If the stimulus is off, the stimulus vector is [0, 0].
                # If the stimulus is on, the stimulus vector is calculated based on the stimulus
                # angle (theta__degrees) and coherence.
                c = get_c(theta__degrees=theta__degrees, coherence=coherence) if stimulus_is_on else np.zeros(2)

                # Simulate the stimulus for the duration of the stimulus.
                for _ in range(duration_in_steps):
                    simulator.step(c=c)

            ###########################
            # SAVE THE SIMULATED DATA #
            ###########################

            stimulus_id = stimulus_ids[(theta__degrees, coherence)]
            decisions = simulator.decisions

            # Save the decision timestamps for the start and end of the stimulus.
            # The decision timestamps are in seconds.
            decisions_start_timestamp = decisions[0, :]
            decisions_end_timestamp = decisions[0, :] + constants.BOUT_DURATION
            trial_data[f"bouts_start_stimulus_{stimulus_id}"]["timestamp"] = decisions_start_timestamp
            trial_data[f"bouts_end_stimulus_{stimulus_id}"]["timestamp"] = decisions_end_timestamp

            # Sample the bout angles based on the decision categories. The decision
            # categories are LEFT, FORWARD, and RIGHT. The bout angles are in degrees.
            bout_angles = sample_bout_angles(decisions[1, :])

            # Calculate the accumulated orientation of the fish. The accumulated orientation
            # is the sum of the bout angles. The accumulated orientation is in degrees.
            orientations = np.cumsum(np.concatenate(([initial_orientation], bout_angles)))

            # Normalize the orientations to be in the range of -180 to 180 degrees.
            orientations = (orientations + 180) % 360 - 180

            # Update the initial orientation for the next stimulus presentation. The initial
            # orientation is the last orientation of the current stimulus presentation.
            initial_orientation = orientations[-1]

            # Save the accumulated orientation of the fish.
            trial_data[f"bouts_start_stimulus_{stimulus_id}"]["fish_accumulated_orientation_lowpass"] = orientations[
                :-1
            ]
            trial_data[f"bouts_end_stimulus_{stimulus_id}"]["fish_accumulated_orientation_lowpass"] = orientations[1:]

            # Save some additional empty fields for compatibility with the real data.
            trial_data[f"raw_stimulus_{stimulus_id}"] = {}
            zeros = np.zeros(decisions.shape[1])
            trial_data[f"bouts_start_stimulus_{stimulus_id}"]["fish_position_x"] = zeros
            trial_data[f"bouts_end_stimulus_{stimulus_id}"]["fish_position_x"] = zeros
            trial_data[f"bouts_start_stimulus_{stimulus_id}"]["fish_position_y"] = zeros
            trial_data[f"bouts_end_stimulus_{stimulus_id}"]["fish_position_y"] = zeros
            trial_data[f"bouts_start_stimulus_{stimulus_id}"]["errorcode"] = zeros
            trial_data[f"bouts_end_stimulus_{stimulus_id}"]["errorcode"] = zeros

            # Reset the simulator decision history and time for the next stimulus presentation.
            simulator.decisions = np.ndarray((2, 0))
            simulator.t = 0

        # Save the trial data to a file. The trial data is saved in pickle format.
        # The file name is the trial index.
        with open(os.path.join(DIRECTORY_NAME, f"{trial_idx}.dat"), "wb") as f:
            pickle.dump(trial_data, f)


if __name__ == "__main__":
    # The stimulus IDs and structure are experiment-specific. The stimulus IDs are
    # used to identify the stimuli in the simulated data. The stimulus structure is
    # used to simulate the stimuli.

    stimulus_ids = STIM_IDS_EXPERIMENT_0
    stimulus_structure = STIMULUS_STRUCTURE

    # Each CPU simulates one 'fish' undergoing N_TRIALS_PER_CPU trials. By
    # default, the seed for each fish is the fish number itself. This ensures
    # that each fish has a unique seed.

    N_TRIALS_PER_CPU = 150
    DIRECTORY_NAME = "simulated_data"

    # Create the directory to save the simulated data. If the directory already exists,
    # the directory is not created. The exist_ok parameter is set to True to prevent an
    # error if the directory already exists.
    os.makedirs(DIRECTORY_NAME, exist_ok=True)

    # Create the 'trials.csv' file. The 'trials.csv' file contains the metadata of the
    # simulated trials.
    with open(os.path.join(DIRECTORY_NAME, "trials.csv"), "w") as f:
        f.write("trial_id,setup_id,fish_num,trial_num,dish_idx,datetime,fish_age,fish_genotype\n")

    # Get the number of CPUs. The number of CPUs is used to parallelize the simulation.
    # Each CPU simulates one 'fish' undergoing N_TRIALS_PER_CPU trials.
    cpu_count = mp.cpu_count()
    processes = []

    print(f"Simulating {N_TRIALS_PER_CPU * cpu_count} trials across {cpu_count} CPUs...\n")
    t0 = t()

    # Simulate each fish on a separate CPU. Each fish undergoes N_TRIALS_PER_CPU trials.
    for fish_num in range(cpu_count):
        process = mp.Process(target=simulate_fish, args=(N_TRIALS_PER_CPU, fish_num, stimulus_ids, stimulus_structure))
        process.start()
        processes.append(process)

    # Wait for all processes to finish. The 'join' method blocks the main process until
    # all processes are finished. This ensures that the main process does not finish
    # before the other processes.
    for process in processes:
        process.join()

    print(f"\nDone! Elapsed time: {t() - t0:.2f} seconds.")

    total_n_trials = N_TRIALS_PER_CPU * cpu_count
    datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Append the metadata of the simulated trials to the 'trials.csv' file.
    with open(os.path.join(DIRECTORY_NAME, "trials.csv"), "a") as f:
        for trial_idx in range(1, total_n_trials + 1):
            fish_num = (trial_idx - 1) // N_TRIALS_PER_CPU
            trial_num = (trial_idx - 1) % N_TRIALS_PER_CPU
            f.write(f"{trial_idx},,{fish_num},{trial_num},,{datetime},,SIM\n")
