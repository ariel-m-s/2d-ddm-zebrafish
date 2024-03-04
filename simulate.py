import os

import datetime as dt
import multiprocessing as mp
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import gmm
from boutfield import boutfield
from constants import (
    BOUT_DURATION,
    GAMMA,
    REFRACTORY_PERIOD,
    RESET_GAMMA,
    S1,
    S2,
    SIGMA,
    STIM_IDS_COMPLETE  as STIM_IDS,
    TIME_STEP,
)

DIR = "test"

class Integrator:
    def __init__(self):
        self.epsilon_mean = np.zeros((2,))
        self.epsilon_cov = np.array([[(SIGMA) ** 2, 0], [0, (SIGMA) ** 2]])

        self.x = np.zeros((2,))
        self.t = 0

        self.decisions = []
        self.time_since_last_decision = np.inf

    @property
    def turn_angle_distr(self):
        # return gmm.distr
        return gmm.main_norm_distr

    @property
    def turn_angle_rv(self):
        # return gmm.rv
        return gmm.main_norm_rv

    @property
    def epsilon(self):
        return np.random.multivariate_normal(self.epsilon_mean, self.epsilon_cov)

    def decide(self):
        is_refractory = self.time_since_last_decision <= REFRACTORY_PERIOD

        if not is_refractory:
            gmm_domain = np.linspace(-180, 180, 1000)
            gmm_params, bout_probability = boutfield([(self.x[1], self.x[0])], return_bool=True)
            gmm_params = gmm_params[0]

            # turn_angle_distr = self.turn_angle_distr(gmm_domain, *gmm_params)
            # bout_rate = np.trapz(turn_angle_distr, gmm_domain) * BASE_BOUT_RATE
            # bout_probability = bout_rate * TIME_STEP

            if np.random.rand() <= bout_probability:
                turn_angle = self.turn_angle_rv(*gmm_params)
                return np.array([self.t, turn_angle])

    def update(self, evidence_angle, evidence_magnitude):
        coherence1 = np.sin(np.deg2rad(evidence_angle)) * evidence_magnitude
        coherence2 = np.cos(np.deg2rad(evidence_angle)) * evidence_magnitude

        drift1 = S1 * coherence1
        drift2 = S2 * coherence2

        drift = np.array([drift1, drift2])

        gamma = GAMMA if (self.time_since_last_decision > 0) else RESET_GAMMA

        step = (drift - gamma * self.x) * TIME_STEP
        noise = self.epsilon * np.sqrt(TIME_STEP)
        # print(noise)

        dx = step + noise
        self.x += dx

        # dx = np.random.multivariate_normal(np.array([motion_x, motion_y]), self.cov) - self.unbiased_x
        # self.unbiased_x += dx * TIME_STEP / tau

        self.t += TIME_STEP

        decision = self.decide()

        if decision is None:
            self.time_since_last_decision += TIME_STEP

        else:
            self.decisions.append(decision)
            self.time_since_last_decision = 0

        return decision


def generate_data(n_trials, cpu_idx):
    print(cpu_idx)
    np.random.seed(cpu_idx)

    trial_idxs = np.arange(n_trials) + (n_trials * cpu_idx) + 1
    stims = STIM_IDS

    _5_seconds = int(5 / TIME_STEP)
    _10_seconds = int(10 / TIME_STEP)

    integrator = Integrator()
    orientation = np.random.uniform(-180, 180)

    for trial_idx in tqdm(trial_idxs, desc=""):

        stim_in_random_order = np.random.permutation(list(stims.keys()))
        data = defaultdict(dict)

        for (stim_angle, stim_coherence) in stim_in_random_order:
            for _ in range(_5_seconds):
                integrator.update(evidence_angle=0, evidence_magnitude=0.0)
            for _ in range(_10_seconds):
                integrator.update(evidence_angle=(stim_angle or 0), evidence_magnitude=(stim_coherence / 100))
            for _ in range(_5_seconds):
                integrator.update(evidence_angle=0, evidence_magnitude=0.0)

            stim_id = stims[(stim_angle, stim_coherence)]

            decisions = np.array(integrator.decisions)
            zeros = np.zeros(decisions.shape[0])

            # if 2d, keep the same, if 1d, add a dimension
            if len(decisions.shape) == 1:
                decisions = None

            data[f"raw_stimulus_{stim_id}"] = {}

            data[f"bouts_start_stimulus_{stim_id}"]["timestamp"] = (
                decisions[:, 0] if (decisions is not None) else np.array([])
            )
            data[f"bouts_end_stimulus_{stim_id}"]["timestamp"] = (
                (decisions[:, 0] + BOUT_DURATION) if (decisions is not None) else np.array([])
            )

            data[f"bouts_start_stimulus_{stim_id}"]["fish_position_x"] = zeros
            data[f"bouts_end_stimulus_{stim_id}"]["fish_position_x"] = zeros

            data[f"bouts_start_stimulus_{stim_id}"]["fish_position_y"] = zeros
            data[f"bouts_end_stimulus_{stim_id}"]["fish_position_y"] = zeros

            data[f"bouts_start_stimulus_{stim_id}"]["errorcode"] = zeros
            data[f"bouts_end_stimulus_{stim_id}"]["errorcode"] = zeros

            # try:
            #     print("YES")
            #     previous_orientation = data[f"bouts_end_stimulus_{stim_id}"]["fish_accumulated_orientation_lowpass"][-1]
            # except KeyError:
            #     previous_orientation = initial_orientation

            data[f"bouts_start_stimulus_{stim_id}"]["fish_accumulated_orientation_lowpass"] = orientation

            data[f"bouts_end_stimulus_{stim_id}"]["fish_accumulated_orientation_lowpass"] = (
                (orientation + decisions[:, 1]) if (decisions is not None) else np.array([])
            )

            try:
                orientation = data[f"bouts_end_stimulus_{stim_id}"]["fish_accumulated_orientation_lowpass"][-1]
            except IndexError:
                pass

            integrator.decisions = []
            integrator.t = 0

        with open(os.path.join("data", DIR, f"{trial_idx}.dat"), "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    N_TRIALS = 5200

    dir = os.path.join("data", DIR)
    os.makedirs(dir, exist_ok=True)

    # create trials.csv

    with open(os.path.join(dir, "trials.csv"), "w") as f:
        f.write("trial_id,setup_id,fish_num,trial_num,dish_idx,datetime,fish_age,fish_genotype\n")

    # how many cpus are available?
    cpu_count = mp.cpu_count()
    trials_per_fish = N_TRIALS // cpu_count

    processes = []

    for cpu_idx in range(cpu_count):
        process = mp.Process(target=generate_data, args=(trials_per_fish, cpu_idx))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(dir, "trials.csv"), "a") as f:
        for trial_idx in range(1, N_TRIALS + 1):
            fish_num = (trial_idx - 1) // trials_per_fish
            trial_num = (trial_idx - 1) % trials_per_fish
            f.write(f"{trial_idx},,{fish_num},{trial_num},,{datetime},,SIM\n")
