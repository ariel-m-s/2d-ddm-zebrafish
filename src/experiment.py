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
from simulator import Simulator

STIMULUS_IDS = {
    (-90, 1): "000",
    (90, 1): "001",
    (-135, 1): "002",
    (45, 1): "003",
    (180, 1): "004",
    (0, 1): "005",
    (135, 1): "006",
    (-45, 1): "007",
    (-90, 0.5): "008",
    (90, 0.5): "009",
    (-135, 0.5): "010",
    (45, 0.5): "011",
    (180, 0.5): "012",
    (0, 0.5): "013",
    (135, 0.5): "014",
    (-45, 0.5): "015",
    (-90, 0.25): "016",
    (90, 0.25): "017",
    (-135, 0.25): "018",
    (45, 0.25): "019",
    (180, 0.25): "020",
    (0, 0.25): "021",
    (135, 0.25): "022",
    (-45, 0.25): "023",
    (None, 0): "024",
}

# STIMULUS_IDS = {
#     (-90, 1): "000",
#     (90, 1): "001",
#     (-135, 1): "002",
#     (45, 1): "003",
#     (180, 1): "004",
#     (0, 1): "005",
#     (135, 1): "006",
#     (-45, 1): "007",
#     (None, 0): "008",
# }

STIMULUS_STRUCTURE = [
    (False, 5),
    (True, 10),
    (False, 5),
]

N_TRIALS_PER_CPU = 10

DIRECTORY_NAME = "simulated_data"


def simulate_fish(n_trials: int, fish_num: int, random_seed: int = None):
    random_seed = random_seed or fish_num
    np.random.seed(random_seed)

    trial_idxs = np.arange(n_trials) + (n_trials * fish_num) + 1

    simulator = Simulator()
    initial_orientation = np.random.uniform(-180, 180)

    for trial_idx in tqdm(trial_idxs, desc=""):
        randomized_stimuli = np.random.permutation(list(STIMULUS_IDS.keys()))
        trial_data = defaultdict(dict)

        for (theta__degrees, coherence) in randomized_stimuli:
            for (stimulus_is_on, duration_in_seconds) in STIMULUS_STRUCTURE:
                duration_in_steps = int(duration_in_seconds / simulator.time_step)

                if stimulus_is_on and (coherence != 0):
                    c = get_c(theta__degrees=theta__degrees, coherence=coherence)

                for _ in range(duration_in_steps):
                    simulator.step(c=c)

            stimulus_id = STIMULUS_IDS[(theta__degrees, coherence)]
            decisions = simulator.decisions

            decisions_start_timestamp = decisions[0, :]
            decisions_end_timestamp = decisions[0, :] + constants.BOUT_DURATION

            trial_data[f"bouts_start_stimulus_{stimulus_id}"]["timestamp"] = decisions_start_timestamp
            trial_data[f"bouts_end_stimulus_{stimulus_id}"]["timestamp"] = decisions_end_timestamp

            # print(decisions[1, :])
            bout_angles = sample_bout_angles(decisions[1, :])
            orientations = np.cumsum(np.concatenate(([initial_orientation], bout_angles)))
            orientations = (orientations + 180) % 360 - 180

            initial_orientation = orientations[-1]

            trial_data[f"bouts_start_stimulus_{stimulus_id}"]["fish_accumulated_orientation_lowpass"] = orientations[
                :-1
            ]
            trial_data[f"bouts_end_stimulus_{stimulus_id}"]["fish_accumulated_orientation_lowpass"] = orientations[1:]

            trial_data[f"raw_stimulus_{stimulus_id}"] = {}
            zeros = np.zeros(decisions.shape[0])

            trial_data[f"bouts_start_stimulus_{stimulus_id}"]["fish_position_x"] = zeros
            trial_data[f"bouts_end_stimulus_{stimulus_id}"]["fish_position_x"] = zeros
            trial_data[f"bouts_start_stimulus_{stimulus_id}"]["fish_position_y"] = zeros
            trial_data[f"bouts_end_stimulus_{stimulus_id}"]["fish_position_y"] = zeros
            trial_data[f"bouts_start_stimulus_{stimulus_id}"]["errorcode"] = zeros
            trial_data[f"bouts_end_stimulus_{stimulus_id}"]["errorcode"] = zeros

            simulator.decisions = np.ndarray((2, 0))
            simulator.t = 0

        with open(os.path.join(DIRECTORY_NAME, f"{trial_idx}.dat"), "wb") as f:
            pickle.dump(trial_data, f)


if __name__ == "__main__":
    os.makedirs(DIRECTORY_NAME, exist_ok=True)

    with open(os.path.join(DIRECTORY_NAME, "trials.csv"), "w") as f:
        f.write("trial_id,setup_id,fish_num,trial_num,dish_idx,datetime,fish_age,fish_genotype\n")

    cpu_count = mp.cpu_count()
    processes = []

    print(f"Simulating {N_TRIALS_PER_CPU * cpu_count} trials across {cpu_count} CPUs...\n")
    t0 = t()

    for fish_num in range(cpu_count):
        process = mp.Process(target=simulate_fish, args=(N_TRIALS_PER_CPU, fish_num))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print(f"\nDone! Elapsed time: {t() - t0:.2f} seconds.")

    total_n_trials = N_TRIALS_PER_CPU * cpu_count
    datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(DIRECTORY_NAME, "trials.csv"), "a") as f:
        for trial_idx in range(1, total_n_trials + 1):
            fish_num = (trial_idx - 1) // N_TRIALS_PER_CPU
            trial_num = (trial_idx - 1) % N_TRIALS_PER_CPU
            f.write(f"{trial_idx},,{fish_num},{trial_num},,{datetime},,SIM\n")
