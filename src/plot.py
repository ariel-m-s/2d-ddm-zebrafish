import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm

from load_data import get_trial_data, load_trials

STIM_IDS_EXPERIMENT_0 = {
    (-90, 100): "000",
    (90, 100): "001",
    (-135, 100): "002",
    (45, 100): "003",
    (180, 100): "004",
    (0, 100): "005",
    (135, 100): "006",
    (-45, 100): "007",
    None: "008",
}

STIM_IDS_EXPERIMENT_1 = {
    (-90, 100): "000",
    (90, 100): "001",
    (-135, 100): "002",
    (45, 100): "003",
    (180, 100): "004",
    (0, 100): "005",
    (135, 100): "006",
    (-45, 100): "007",
    (-90, 25): "008",
    (90, 25): "009",
    (-135, 25): "010",
    (45, 25): "011",
    (180, 25): "012",
    (0, 25): "013",
    (135, 25): "014",
    (-45, 25): "015",
    None: "016",
}

STIM_IDS_EXPERIMENT_2 = {
    (-90, 100): "000",
    (90, 100): "001",
    (-135, 100): "002",
    (45, 100): "003",
    (180, 100): "004",
    (0, 100): "005",
    (135, 100): "006",
    (-45, 100): "007",
    (-90, 50): "008",
    (90, 50): "009",
    (-135, 50): "010",
    (45, 50): "011",
    (180, 50): "012",
    (0, 50): "013",
    (135, 50): "014",
    (-45, 50): "015",
    (-90, 25): "016",
    (90, 25): "017",
    (-135, 25): "018",
    (45, 25): "019",
    (180, 25): "020",
    (0, 25): "021",
    (135, 25): "022",
    (-45, 25): "023",
    None: "024",
}

STIM_START = 5
STIM_END = 15

COHERENCES = [100, 50, 25]
# COHERENCES = [100]
N_COHERENCE_LEVELS = len(COHERENCES)

ABS_STIM = True
SAVE_FIGS = True

FIGS_TO_SHOW = ["1b", "1c", "1d", "1e", "1f"]
FIGS_TO_SHOW = ["three_class_figures", "turn_angle_histogram"]

DATA_DIR = "free_swimming_8fish_random_dot_kinematogram_data/org"
data_type = "behavior"

DATA_DIR = "../2d-ddm-zebrafish/simulated_data"
data_type = "simulated"


CATEGORICAL_THRESHOLDS = 10.3


def stim_sym_to_int(abs_stim):
    return int(abs_stim.split("_")[1]) if isinstance(abs_stim, str) else abs_stim


trials = load_trials(os.path.join(DATA_DIR, "trials.csv"))

turn_angles__trials = defaultdict(lambda: defaultdict(list))
start_times__trials = defaultdict(lambda: defaultdict(list))
is_complete__trials = defaultdict(lambda: defaultdict(list))

for trial_id in tqdm.tqdm(trials.index):
    data = get_trial_data(data_path=DATA_DIR, trial_id=trial_id)

    data_length = len(data.keys())

    if data_length == (9 * 3):
        if 25 in COHERENCES:
            continue
        STIM_IDS = STIM_IDS_EXPERIMENT_0
    elif data_length == (17 * 3):
        if 50 in COHERENCES:
            continue
        STIM_IDS = STIM_IDS_EXPERIMENT_1
    elif data_length == (25 * 3):
        STIM_IDS = STIM_IDS_EXPERIMENT_2
    else:
        continue

    STIM_IDS_25 = {(k and k[0]): v for (k, v) in STIM_IDS.items() if (k is None or k[1] == 25)}
    STIM_IDS_50 = {(k and k[0]): v for (k, v) in STIM_IDS.items() if (k is None or k[1] == 50)}
    STIM_IDS_100 = {(k and k[0]): v for (k, v) in STIM_IDS.items() if (k is None or k[1] == 100)}

    for (coherence, stim_ids) in [(25, STIM_IDS_25), (50, STIM_IDS_50), (100, STIM_IDS_100)]:
        if len(stim_ids.keys()) != 9:
            continue

        this_trial_num = trials.loc[trial_id]["trial_num"]
        this_fish_age = trials.loc[trial_id]["fish_age"]

        this_setup_id = trials.loc[trial_id]["setup_id"]
        this_fish_num = trials.loc[trial_id]["fish_num"]
        this_fish_id = f"{this_setup_id}_{this_fish_num}"

        for stim in stim_ids.keys():
            stim_id = stim_ids[stim]

            bout_start_data = data[f"bouts_start_stimulus_{stim_id}"]
            bout_end_data = data[f"bouts_end_stimulus_{stim_id}"]

            start_time = bout_start_data["timestamp"]
            end_time = bout_end_data["timestamp"]

            angle_before_bout = bout_start_data["fish_accumulated_orientation_lowpass"]
            angle_after_bout = bout_end_data["fish_accumulated_orientation_lowpass"]

            # only save turn_angle data for bouts that start and end within the stimulus
            # duration

            x_start = bout_start_data["fish_position_x"]
            y_start = bout_start_data["fish_position_y"]
            radius_start = np.sqrt(x_start**2 + y_start**2)

            x_end = bout_end_data["fish_position_x"]
            y_end = bout_end_data["fish_position_y"]
            radius_end = np.sqrt(x_end**2 + y_end**2)

            if data_type == "behavior":
                error_mask = (bout_start_data["errorcode"] == 0) & (bout_end_data["errorcode"] == 0)

                angle_before_bout = angle_before_bout[error_mask]
                angle_after_bout = angle_after_bout[error_mask]
                start_time = start_time[error_mask]
                end_time = end_time[error_mask]
                x_start = x_start[error_mask]
                y_start = y_start[error_mask]
                x_end = x_end[error_mask]
                y_end = y_end[error_mask]
                radius_start = radius_start[error_mask]
                radius_end = radius_end[error_mask]

                max_radius = 1.0
                radius_mask = (radius_start < max_radius) & (radius_end < max_radius)

                mask = radius_mask

            elif data_type == "simulated":
                mask = np.ones_like(start_time, dtype=bool)

            if not np.all(mask):
                continue

            if data_type == "behavior":
                turn_angle = angle_before_bout - angle_after_bout
            elif data_type == "simulated":
                turn_angle = angle_after_bout - angle_before_bout
            else:
                raise ValueError(f"Invalid data type: {data_type}")

            turn_angles__trials[stim][coherence].append(turn_angle[mask])
            start_times__trials[stim][coherence].append(start_time[mask])
            is_complete__trials[stim][coherence].append(np.all(mask))

            if stim in (-45, -90, -135):
                turn_angles__trials[f"abs_{-stim}"][coherence].append(-turn_angle[mask])
                start_times__trials[f"abs_{-stim}"][coherence].append(start_time[mask])
                is_complete__trials[f"abs_{-stim}"][coherence].append(np.all(mask))

            elif stim in (45, 90, 135):
                turn_angles__trials[f"abs_{stim}"][coherence].append(turn_angle[mask])
                start_times__trials[f"abs_{stim}"][coherence].append(start_time[mask])
                is_complete__trials[f"abs_{stim}"][coherence].append(np.all(mask))

all_turn_angles = defaultdict(dict)
all_start_times = defaultdict(dict)

if ABS_STIM:
    stim_pos = {
        None: (1, 1),
        0: (0, 1),
        180: (2, 1),
        "abs_45": (0, 0),
        "abs_90": (1, 0),
        "abs_135": (2, 0),
    }
else:
    stim_pos = {
        None: (1, 1),
        0: (0, 1),
        45: (0, 2),
        90: (1, 2),
        135: (2, 2),
        180: (2, 1),
        -45: (0, 0),
        -90: (1, 0),
        -135: (2, 0),
    }

stim_keys = list(stim_pos.keys())

for coherence in COHERENCES:
    for stim in stim_keys:
        stim_val = stim_sym_to_int(stim)

        all_turn_angles[stim][coherence] = np.concatenate(turn_angles__trials[stim][coherence])
        all_start_times[stim][coherence] = np.concatenate(start_times__trials[stim][coherence])

if "three_class_figures" in FIGS_TO_SHOW:
    if not ABS_STIM:
        raise NotImplementedError("This figure is only implemented for ABS_STIM=True")

    fig_tcf, axes_tcf = plt.subplots(3, 2, figsize=(4, 7), sharex=True, sharey=True)

    n_bins = 20
    domain_size = 20

    bin_edges = np.linspace(0, domain_size, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_size = domain_size / n_bins

    # total number of bouts for the None stimulus
    n_bouts = len(all_start_times[None][100])
    baseline = n_bouts / n_bins

    for stim in stim_keys:
        print(stim)
        stim_val = stim_sym_to_int(stim)

        axes_tcf[stim_pos[stim]].axvspan(STIM_START, STIM_END, color="lightgray", alpha=0.5)

        frequency_means_left = np.zeros((N_COHERENCE_LEVELS, n_bins))
        frequency_std_errs_left = np.zeros((N_COHERENCE_LEVELS, n_bins))

        frequency_means_right = np.zeros((N_COHERENCE_LEVELS, n_bins))
        frequency_std_errs_right = np.zeros((N_COHERENCE_LEVELS, n_bins))

        frequency_means_forward = np.zeros((N_COHERENCE_LEVELS, n_bins))
        frequency_std_errs_forward = np.zeros((N_COHERENCE_LEVELS, n_bins))

        frequency_means_total = np.zeros((N_COHERENCE_LEVELS, n_bins))
        frequency_std_errs_total = np.zeros((N_COHERENCE_LEVELS, n_bins))

        intersections = {
            None: (-10.3, 10.3),
            0: (-10.3, 10.3),
            45: (-10.3, 10.3),
            90: (-10.3, 10.3),
            135: (-10.3, 10.3),
            180: (-10.3, 10.3),
        }

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            for bin in range(n_bins):
                mask = (all_start_times[stim][coherence] > bin_edges[bin]) & (
                    all_start_times[stim][coherence] <= bin_edges[bin + 1]
                )

                turn_angles__filtered = all_turn_angles[stim][coherence][mask]

                left_mask = turn_angles__filtered < intersections[stim_val][0]
                right_mask = turn_angles__filtered > intersections[stim_val][1]
                forward_mask = (~left_mask) & (~right_mask)

                # We want percentages relative to baseline. if equal to baseline, then 1. if half, then 0.5, if double, then 2.

                percentage_left = np.sum(left_mask) / baseline * 100
                percentage_right = np.sum(right_mask) / baseline * 100
                percentage_forward = np.sum(forward_mask) / baseline * 100
                percentage_total = np.sum(mask) / baseline * 100

                frequency_means_left[coherence_idx, bin] = percentage_left
                frequency_means_right[coherence_idx, bin] = percentage_right
                frequency_means_forward[coherence_idx, bin] = percentage_forward
                frequency_means_total[coherence_idx, bin] = percentage_total

                # error is 0
                frequency_std_errs_left[coherence_idx, bin] = 0
                frequency_std_errs_right[coherence_idx, bin] = 0
                frequency_std_errs_forward[coherence_idx, bin] = 0
                frequency_std_errs_total[coherence_idx, bin] = 0

                if ABS_STIM and (stim_val in [45, 90, 135]):
                    frequency_means_left[coherence_idx, bin] /= 2
                    frequency_std_errs_left[coherence_idx, bin] /= 2

                    frequency_means_right[coherence_idx, bin] /= 2
                    frequency_std_errs_right[coherence_idx, bin] /= 2

                    frequency_means_forward[coherence_idx, bin] /= 2
                    frequency_std_errs_forward[coherence_idx, bin] /= 2

                    frequency_means_total[coherence_idx, bin] /= 2
                    frequency_std_errs_total[coherence_idx, bin] /= 2

        # plot as line plot. left is blue, right is red, forward is black. Coherence translates into opacity of those colors.

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            print(stim)
            opacity = coherence / 100

            axes_tcf[stim_pos[stim]].errorbar(
                bin_centers[1:-1],
                frequency_means_left[coherence_idx][1:-1],
                yerr=frequency_std_errs_left[coherence_idx][1:-1],
                linewidth=3,
                linestyle="solid",
                color="blue",
                alpha=opacity,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=2,
            )

            axes_tcf[stim_pos[stim]].errorbar(
                bin_centers[1:-1],
                frequency_means_right[coherence_idx][1:-1],
                yerr=frequency_std_errs_right[coherence_idx][1:-1],
                linewidth=3,
                linestyle="solid",
                color="red",
                alpha=opacity,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=2,
            )

            axes_tcf[stim_pos[stim]].errorbar(
                bin_centers[1:-1],
                frequency_means_forward[coherence_idx][1:-1],
                yerr=frequency_std_errs_forward[coherence_idx][1:-1],
                linewidth=3,
                linestyle="solid",
                color="black",
                alpha=opacity,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=2,
            )

            # total in pink
            axes_tcf[stim_pos[stim]].errorbar(
                bin_centers[1:-1],
                frequency_means_total[coherence_idx][1:-1],
                yerr=frequency_std_errs_total[coherence_idx][1:-1],
                linewidth=3,
                linestyle="solid",
                color="pink",
                alpha=opacity,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=2,
            )

            axes_tcf[stim_pos[stim]].set_xlim(0, 20)
            axes_tcf[stim_pos[stim]].set_ylim(0, 200)

            axes_tcf[stim_pos[stim]].set_xticks([0, 5, 10, 15, 20])
            axes_tcf[stim_pos[stim]].set_xticklabels([0, 5, 10, 15, 20])
            # ticks for y axis at 0, 50, 100, 150, 200
            axes_tcf[stim_pos[stim]].set_yticks([0, 50, 100, 150, 200])

            axes_tcf[stim_pos[stim]].set_xlabel("Time (s)")
            axes_tcf[stim_pos[stim]].set_ylabel("Percentage of bouts")


if "turn_angle_histogram" in FIGS_TO_SHOW:
    if ABS_STIM:
        turn_angle_histogram, axes = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
    else:
        turn_angle_histogram, axes = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)

    # We want to plot the histogram of turn angles for each coherence level, for each stimulus direction.
    # These will be distributed between -180 and 180 degrees, and bins will be 10 degrees wide.
    # Use np.histogram to compute the histogram, and then plot the histogram using a line plot.

    # For each stimulus direction, plot the histogram for each coherence level, using a different color for each coherence level.

    n_bins = 180

    for stim in stim_keys:
        stim_val = stim_sym_to_int(stim)

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            start_times__this = np.array(all_start_times[stim][coherence])
            turn_angles__this = np.array(all_turn_angles[stim][coherence])

            mask = (start_times__this > (STIM_START + 1)) & (start_times__this <= (STIM_END - 6))
            turn_angles__filtered = turn_angles__this[mask]

            hist, bin_edges = np.histogram(turn_angles__filtered, bins=n_bins, range=(-180, 180), density=False)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            if ABS_STIM and stim_val in [45, 90, 135]:
                hist = hist / 2

            if stim_val is None:
                color = "gray"
            else:
                color = plt.get_cmap("tab10")(coherence_idx)

            axes[stim_pos[stim]].plot(bin_centers, hist, color=color, linewidth=1, linestyle="solid", alpha=0.7)

            axes[stim_pos[stim]].set_xlim(-135, 135)
            axes[stim_pos[stim]].set_xticks([-90, 0, 90])
            axes[stim_pos[stim]].set_xticklabels([-90, 0, 90])

            axes[stim_pos[stim]].set_ylabel("Count")

            axes[stim_pos[stim]].set_title(f"Stimulus: {stim_val}º")

            # Make y-axis 0 be at x axis (meaning, no negative values).
            axes[stim_pos[stim]].spines["bottom"].set_position("zero")

            # print a number in the corner indicating the total count of turn angles
            # in the histogram. print in the same color as the histogram and separated.

            n_bouts = len(turn_angles__filtered)

            if ABS_STIM and stim_val in [45, 90, 135]:
                n_bouts /= 2

            # plot vertical line at the maximum of the histogram
            max_idx = np.argmax(hist)
            max_angle = bin_centers[max_idx]

            axes[stim_pos[stim]].axvline(
                max_angle,
                color=color,
                linewidth=1,
                linestyle="dashed",
                alpha=0.5,
            )

            axes[stim_pos[stim]].text(
                # change position depeding on coherence level, so the text doesn't overlap
                0.95,
                0.95 - (0.1 * coherence_idx),
                n_bouts,
                horizontalalignment="right",
                verticalalignment="top",
                transform=axes[stim_pos[stim]].transAxes,
                color=color,
            )

# Fig. 1b.
# Bouting frequency as a function of time, for all coherence levels.

if "1b" in FIGS_TO_SHOW:
    if ABS_STIM:
        fig1b, axes1b = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
    else:
        fig1b, axes1b = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)

    axes1b[0, 0].set_ylim(0.5, 1.5)

    n_bins = 20
    domain_size = 20

    bin_edges = np.linspace(0, domain_size, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_size = domain_size / n_bins

    for stim in stim_keys:
        stim_val = stim_sym_to_int(stim)

        axes1b[stim_pos[stim]].axvspan(STIM_START, STIM_END, color="lightgray", alpha=0.5)

        frequency_means = np.zeros((N_COHERENCE_LEVELS, n_bins))
        frequency_std_errs = np.zeros((N_COHERENCE_LEVELS, n_bins))

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            for bin in range(n_bins):
                frequencies_per_trial = []

                for (trial_idx, is_complete) in enumerate(is_complete__trials[stim][coherence]):
                    # if not is_complete:
                    #     continue

                    start_times = start_times__trials[stim][coherence][trial_idx]
                    mask = (start_times > bin_edges[bin]) & (start_times <= bin_edges[bin + 1])
                    frequencies_per_trial.append(np.sum(mask) / bin_size)

                frequency_means[coherence_idx, bin] = np.nanmean(frequencies_per_trial)
                frequency_std_errs[coherence_idx, bin] = np.nanstd(frequencies_per_trial) / np.sqrt(
                    len(frequencies_per_trial)
                )

        cmap = plt.get_cmap("tab10")

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            if stim_val is None:
                color = "gray"
            else:
                color = cmap(coherence_idx)

            axes1b[stim_pos[stim]].errorbar(
                bin_centers[1:-1],
                frequency_means[coherence_idx, 1:-1],
                yerr=frequency_std_errs[coherence_idx, 1:-1],
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            # Mark the mean frequency during stimulus with a dashed line.
            axes1b[stim_pos[stim]].axhline(
                np.nanmean(frequency_means[coherence_idx, STIM_START + 2 : STIM_END]),
                color=color,
                linewidth=1,
                linestyle="dashed",
                alpha=0.5,
            )

            # axes1b[stim_pos[stim]].set_xlim(0, 20)

            # axes1b[stim_pos[stim]].set_xticks([0, 5, 10, 15, 20])
            # axes1b[stim_pos[stim]].set_xticklabels([0, 5, 10, 15, 20])

            axes1b[stim_pos[stim]].set_xlabel("Time (s)")
            axes1b[stim_pos[stim]].set_ylabel("Bouts / s")

# Fig. 1c.
# Time-binned accuracy as a function of time, for all coherence levels.

if "1c" in FIGS_TO_SHOW:
    if ABS_STIM:
        fig1ca, axes1ca = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
        fig1cb, axes1cb = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
    else:
        fig1ca, axes1ca = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)
        fig1cb, axes1cb = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)

    axes1ca[0, 0].set_ylim(-20, 40)
    axes1cb[0, 0].set_ylim(30, 85)

    start_times__null_stim = all_start_times[None]
    turn_angles__null_stim = all_turn_angles[None]

    num_bins = 20
    bin_edges = np.linspace(0, 20, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for stim in stim_keys:
        stim_val = stim_sym_to_int(stim)

        ax1ca = axes1ca[stim_pos[stim]]
        ax1ca.axvspan(STIM_START, STIM_END, color="lightgray", alpha=0.5)

        ax1cb = axes1cb[stim_pos[stim]]
        ax1cb.axvspan(STIM_START, STIM_END, color="lightgray", alpha=0.5)

        start_times__this_stim = all_start_times[stim]
        turn_angles__this_stim = all_turn_angles[stim]

        if stim_val is None:
            stim__rad = np.deg2rad(0)
        else:
            stim__rad = np.deg2rad(stim_val)

        total_dot_product_mean = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_dot_product_std_err = np.zeros((N_COHERENCE_LEVELS, num_bins))

        total_correct_bouts_mean = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_std_err = np.zeros((N_COHERENCE_LEVELS, num_bins))

        total_bouts = np.zeros((N_COHERENCE_LEVELS, num_bins))

        null_turn__rad = np.deg2rad(turn_angles__null_stim[100])
        null_turn_vec = np.array([np.cos(null_turn__rad), np.sin(null_turn__rad)])

        null_turn_dp = np.cos(stim__rad - null_turn__rad)
        null_dp_mean = np.nanmean(null_turn_dp)

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            for bin in range(num_bins):
                mask = (start_times__this_stim[coherence] > bin_edges[bin]) & (
                    start_times__this_stim[coherence] <= bin_edges[bin + 1]
                )

                turn__deg = turn_angles__this_stim[coherence][mask]

                turn__rad = np.deg2rad(turn__deg)
                turn_vec = np.array([np.cos(turn__rad), np.sin(turn__rad)])

                turn_dp = np.cos(stim__rad - turn__rad)
                normalized_turn_dp = turn_dp - null_dp_mean

                normalized_turn_dp = normalized_turn_dp * 100

                total_dot_product_mean[coherence_idx, bin] = np.nanmean(normalized_turn_dp)
                total_dot_product_std_err[coherence_idx, bin] = np.std(normalized_turn_dp) / np.sqrt(
                    len(normalized_turn_dp)
                )

                if stim_val is None:
                    total_correct_bout_mask = turn__deg > 0
                elif stim_val == 0:
                    total_correct_bout_mask = np.abs(turn__deg) < CATEGORICAL_THRESHOLDS
                elif stim_val == 180:
                    total_correct_bout_mask = np.abs(turn__deg) >= CATEGORICAL_THRESHOLDS
                elif stim_val > 0:
                    total_correct_bout_mask = turn__deg > CATEGORICAL_THRESHOLDS
                elif stim_val < 0:
                    total_correct_bout_mask = turn__deg <= CATEGORICAL_THRESHOLDS

                # Map false to 0 and true to 100

                total_correct_bout_mask = total_correct_bout_mask * 100

                total_correct_bouts_mean[coherence_idx, bin] = np.nanmean(total_correct_bout_mask)
                total_correct_bouts_std_err[coherence_idx, bin] = np.std(total_correct_bout_mask) / np.sqrt(
                    len(total_correct_bout_mask)
                )

            total_bouts[coherence_idx, bin] = len(turn__rad)

        cmap = plt.get_cmap("tab10")

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            if stim_val is None:
                color = "gray"
            else:
                color = cmap(coherence_idx)

            ax1ca.axhline(
                np.nanmean(total_dot_product_mean[coherence_idx, STIM_START:STIM_END]),
                color=color,
                linewidth=1,
                linestyle="dashed",
                alpha=0.5,
            )

            ax1ca.errorbar(
                bin_centers,
                total_dot_product_mean[coherence_idx],
                yerr=total_dot_product_std_err[coherence_idx],
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            ax1cb.axhline(
                np.nanmean(total_correct_bouts_mean[coherence_idx, STIM_START:STIM_END]),
                color=color,
                linewidth=1,
                linestyle="dashed",
                alpha=0.5,
            )

            ax1cb.errorbar(
                bin_centers,
                total_correct_bouts_mean[coherence_idx],
                yerr=total_correct_bouts_std_err[coherence_idx],
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

# Fig. 1d.
# Accuracy over consecutive bouts, for all coherence levels.

if "1d" in FIGS_TO_SHOW:
    if ABS_STIM:
        fig1da, axes1da = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
        fig1db, axes1db = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
    else:
        fig1da, axes1da = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)
        fig1db, axes1db = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)

    axes1da[0, 0].set_ylim(-20, 40)
    axes1db[0, 0].set_ylim(30, 85)

    start_times__null_stim = all_start_times[None]
    turn_angles__null_stim = all_turn_angles[None]

    num_bins = 3

    for stim in stim_keys:
        stim_val = stim_sym_to_int(stim)

        stim_showing_duration = 3
        end_shift = 6
        stim_gap = STIM_END - STIM_START - stim_showing_duration - end_shift

        ax1da = axes1da[stim_pos[stim]]
        ax1da.axvspan(STIM_START, STIM_START + stim_showing_duration, color="lightgray", alpha=0.5)
        ax1da.axvspan(STIM_END - end_shift - (stim_gap / 7), STIM_END - end_shift, color="lightgray", alpha=0.5)

        ax1db = axes1db[stim_pos[stim]]
        ax1db.axvspan(STIM_START, STIM_START + stim_showing_duration, color="lightgray", alpha=0.5)
        ax1db.axvspan(STIM_END - end_shift - (stim_gap / 7), STIM_END - end_shift, color="lightgray", alpha=0.5)

        # Show x axis from (START_TIME - 1) to (END_TIME + 4) seconds.
        ax1da.set_xlim(STIM_START - 1, STIM_END + 4 - end_shift)
        ax1db.set_xlim(STIM_START - 1, STIM_END + 4 - end_shift)

        start_times__trials__this_stim = start_times__trials[stim]
        turn_angles__trials__this_stim = turn_angles__trials[stim]

        if stim_val is None:
            stim__rad = np.deg2rad(0)
            stim_vec = np.expand_dims(np.array([np.cos(stim__rad), np.sin(stim__rad)]), axis=1)
        else:
            stim__rad = np.deg2rad(stim_val)
            stim_vec = np.expand_dims(np.array([np.cos(stim__rad), np.sin(stim__rad)]), axis=1)

        null_turn__rad = np.deg2rad(turn_angles__null_stim[100])
        null_turn_vec = np.array([np.cos(null_turn__rad), np.sin(null_turn__rad)])

        null_turn_dp = np.cos(stim__rad - null_turn__rad)
        null_dp_mean = np.nanmean(null_turn_dp)

        # Let's find, for coherence level 100, all the first bouts that start after STIM_START.

        total_dot_product_mean_during = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_dot_product_std_dev_during = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_dot_product_count_during = np.zeros((N_COHERENCE_LEVELS, num_bins))

        total_dot_product_mean_after = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_dot_product_std_dev_after = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_dot_product_count_after = np.zeros((N_COHERENCE_LEVELS, num_bins))

        total_correct_bouts_mean_during = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_std_dev_during = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_count_during = np.zeros((N_COHERENCE_LEVELS, num_bins))

        total_correct_bouts_mean_after = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_std_dev_after = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_count_after = np.zeros((N_COHERENCE_LEVELS, num_bins))

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            during__deg = defaultdict(list)
            after__deg = defaultdict(list)

            for (trail_idx, start_times) in enumerate(start_times__trials__this_stim[coherence]):
                # Find the indices of the first three bouts that start after STIM_START and the first three bouts that start after STIM_END.
                # The indices should be relative to the start_times array.

                during_idxs = np.where((start_times > (STIM_START + 0)) & (start_times <= STIM_END))[0][:num_bins]
                after_idxs = np.where(start_times >= STIM_END)[0][:num_bins]

                # print(start_times)
                # print(during_idxs)
                # print(turn_angles__trials__this_stim[coherence][trail_idx])
                # print()

                for (bout_pos, bout_idx) in enumerate(during_idxs):
                    during__deg[bout_pos].append(turn_angles__trials__this_stim[coherence][trail_idx][bout_idx])

                for (bout_pos, bout_idx) in enumerate(after_idxs):
                    after__deg[bout_pos].append(turn_angles__trials__this_stim[coherence][trail_idx][bout_idx])

            for bout_pos in range(3):
                turn__rad = np.deg2rad(during__deg[bout_pos])

                turn_dp = np.cos(stim__rad - turn__rad)
                normalized_turn_dp = turn_dp - null_dp_mean

                normalized_turn_dp = normalized_turn_dp * 100

                total_dot_product_mean_during[coherence_idx, bout_pos] = np.nanmean(normalized_turn_dp)
                total_dot_product_std_dev_during[coherence_idx, bout_pos] = np.nanstd(normalized_turn_dp)
                total_dot_product_count_during[coherence_idx, bout_pos] = len(normalized_turn_dp)

                turn__rad = np.deg2rad(after__deg[bout_pos])

                turn_dp = np.cos(stim__rad - turn__rad)
                normalized_turn_dp = turn_dp - null_dp_mean

                normalized_turn_dp = normalized_turn_dp * 100

                total_dot_product_mean_after[coherence_idx, bout_pos] = np.nanmean(normalized_turn_dp)
                total_dot_product_std_dev_after[coherence_idx, bout_pos] = np.nanstd(normalized_turn_dp)
                total_dot_product_count_after[coherence_idx, bout_pos] = len(normalized_turn_dp)

                if stim_val is None:
                    total_correct_bout_mask_during = np.array(during__deg[bout_pos]) > 0
                    total_correct_bout_mask_after = np.array(after__deg[bout_pos]) > 0
                elif stim_val == 0:
                    total_correct_bout_mask_during = np.abs(np.array(during__deg[bout_pos])) < CATEGORICAL_THRESHOLDS
                    total_correct_bout_mask_after = np.abs(np.array(after__deg[bout_pos])) < CATEGORICAL_THRESHOLDS
                elif stim_val == 180:
                    total_correct_bout_mask_during = np.abs(np.array(during__deg[bout_pos])) > CATEGORICAL_THRESHOLDS
                    total_correct_bout_mask_after = np.abs(np.array(after__deg[bout_pos])) > CATEGORICAL_THRESHOLDS
                elif stim_val > 0:
                    total_correct_bout_mask_during = np.array(during__deg[bout_pos]) > CATEGORICAL_THRESHOLDS
                    total_correct_bout_mask_after = np.array(after__deg[bout_pos]) > CATEGORICAL_THRESHOLDS
                elif stim_val < 0:
                    total_correct_bout_mask_during = np.array(during__deg[bout_pos]) < CATEGORICAL_THRESHOLDS
                    total_correct_bout_mask_after = np.array(after__deg[bout_pos]) < CATEGORICAL_THRESHOLDS

                # Map false to 0 and true to 100

                total_correct_bout_mask_during = total_correct_bout_mask_during * 100
                total_correct_bout_mask_after = total_correct_bout_mask_after * 100

                total_correct_bouts_mean_during[coherence_idx, bout_pos] = np.nanmean(total_correct_bout_mask_during)
                total_correct_bouts_std_dev_during[coherence_idx, bout_pos] = np.nanstd(total_correct_bout_mask_during)
                total_correct_bouts_count_during[coherence_idx, bout_pos] = len(total_correct_bout_mask_during)

                total_correct_bouts_mean_after[coherence_idx, bout_pos] = np.nanmean(total_correct_bout_mask_after)
                total_correct_bouts_std_dev_after[coherence_idx, bout_pos] = np.nanstd(total_correct_bout_mask_after)
                total_correct_bouts_count_after[coherence_idx, bout_pos] = len(total_correct_bout_mask_after)

        cmap = plt.get_cmap("tab10")

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            # Arbitrarily position the first three bouts during stimulus at seconds STIM_START + 1, STIM_START + 2, STIM_START + 3,
            # and the first three bouts after stimulus at seconds STIM_END + 1, STIM_END + 2, STIM_END + 3. Only these six ticks
            # will be shown on the x-axis.

            if stim_val is None:
                color = "gray"
            else:
                color = cmap(coherence_idx)

            ax1da.plot(
                [STIM_START - 0.5 + num_bins, STIM_END + 0.5 - end_shift],
                [
                    total_dot_product_mean_during[coherence_idx, 2],
                    total_dot_product_mean_after[coherence_idx, 0],
                ],
                linewidth=0.6,
                linestyle="dashed",
                color=color,
                alpha=0.2,
            )

            ax1da.errorbar(
                STIM_START + 0.5 + np.arange(num_bins),
                total_dot_product_mean_during[coherence_idx],
                yerr=total_dot_product_std_dev_during[coherence_idx]
                / np.sqrt(total_dot_product_count_during[coherence_idx]),
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            ax1da.errorbar(
                STIM_END + 0.5 + np.arange(num_bins) - end_shift,
                total_dot_product_mean_after[coherence_idx],
                yerr=total_dot_product_std_dev_after[coherence_idx]
                / np.sqrt(total_dot_product_count_after[coherence_idx]),
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            ax1db.plot(
                [STIM_START - 0.5 + num_bins, STIM_END + 0.5 - end_shift],
                [
                    total_correct_bouts_mean_during[coherence_idx, 2],
                    total_correct_bouts_mean_after[coherence_idx, 0],
                ],
                linewidth=0.6,
                linestyle="dashed",
                color=color,
                alpha=0.2,
            )

            ax1db.errorbar(
                STIM_START + 0.5 + np.arange(num_bins),
                total_correct_bouts_mean_during[coherence_idx],
                yerr=total_correct_bouts_std_dev_during[coherence_idx]
                / np.sqrt(total_correct_bouts_count_during[coherence_idx]),
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            ax1db.errorbar(
                STIM_END + 0.5 + np.arange(num_bins) - end_shift,
                total_correct_bouts_mean_after[coherence_idx],
                yerr=total_correct_bouts_std_dev_after[coherence_idx]
                / np.sqrt(total_correct_bouts_count_after[coherence_idx]),
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            # Ticks.
            ax1da.set_xticks(
                [
                    STIM_START + 0.5,
                    STIM_START + 1.5,
                    STIM_START + 2.5,
                    STIM_END + 0.5 - end_shift,
                    STIM_END + 1.5 - end_shift,
                    STIM_END + 2.5 - end_shift,
                ]
            )
            ax1da.set_xticklabels(["1º", "2º", "3º", "1º", "2º", "3º"])
            ax1db.set_xticks(
                [
                    STIM_START + 0.5,
                    STIM_START + 1.5,
                    STIM_START + 2.5,
                    STIM_END + 0.5 - end_shift,
                    STIM_END + 1.5 - end_shift,
                    STIM_END + 2.5 - end_shift,
                ]
            )
            ax1db.set_xticklabels(["1º", "2º", "3º", "1º", "2º", "3º"])

# Fig. 1e.
# Accuracy of the first bout during the stimulus and the first bout after
# the stimulus end as a function of delay, for all coherence levels.

if "1e" in FIGS_TO_SHOW:
    if ABS_STIM:
        fig1ea, axes1ea = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
        fig1eb, axes1eb = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
    else:
        fig1ea, axes1ea = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)
        fig1eb, axes1eb = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)

    axes1ea[0, 0].set_ylim(-20, 40)
    axes1eb[0, 0].set_ylim(30, 85)

    start_times__null_stim = all_start_times[None]
    turn_angles__null_stim = all_turn_angles[None]

    stim_showing_duration = 2
    end_shift = 7
    stim_gap = STIM_END - STIM_START - stim_showing_duration - end_shift
    num_bins = 4

    bin_edges_during = np.linspace(STIM_START, STIM_START + stim_showing_duration, num_bins + 1)
    bin_edges_after = np.linspace(STIM_END, STIM_END + stim_showing_duration, num_bins + 1)

    bin_centers_during = (bin_edges_during[:-1] + bin_edges_during[1:]) / 2
    bin_centers_after = (bin_edges_after[:-1] + bin_edges_after[1:]) / 2

    for stim in stim_keys:
        stim_val = stim_sym_to_int(stim)

        ax1ea = axes1ea[stim_pos[stim]]
        ax1ea.axvspan(STIM_START, STIM_START + stim_showing_duration, color="lightgray", alpha=0.5)
        ax1ea.axvspan(STIM_END - end_shift - (stim_gap / 7), STIM_END - end_shift, color="lightgray", alpha=0.5)

        ax1eb = axes1eb[stim_pos[stim]]
        ax1eb.axvspan(STIM_START, STIM_START + stim_showing_duration, color="lightgray", alpha=0.5)
        ax1eb.axvspan(STIM_END - end_shift - (stim_gap / 7), STIM_END - end_shift, color="lightgray", alpha=0.5)

        # same as Fig. 1c, but we only keep the first bouts during and after the stimulus.

        start_times__trials__this_stim = start_times__trials[stim]
        turn_angles__trials__this_stim = turn_angles__trials[stim]

        if stim_val is None:
            stim__rad = np.deg2rad(0)
        else:
            stim__rad = np.deg2rad(stim_val)

        null_turn__rad = np.deg2rad(turn_angles__null_stim[100])
        null_turn_vec = np.array([np.cos(null_turn__rad), np.sin(null_turn__rad)])

        null_turn_dp = np.cos(stim__rad - null_turn__rad)
        null_dp_mean = np.nanmean(null_turn_dp)

        total_dot_product_mean_during = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_dot_product_std_dev_during = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_dot_product_count_during = np.zeros((N_COHERENCE_LEVELS, num_bins))

        total_dot_product_mean_after = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_dot_product_std_dev_after = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_dot_product_count_after = np.zeros((N_COHERENCE_LEVELS, num_bins))

        total_correct_bouts_mean_during = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_std_dev_during = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_count_during = np.zeros((N_COHERENCE_LEVELS, num_bins))

        total_correct_bouts_mean_after = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_std_dev_after = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_count_after = np.zeros((N_COHERENCE_LEVELS, num_bins))

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            times_during = []
            angles_during = []

            times_after = []
            angles_after = []

            for (trail_idx, start_times) in enumerate(start_times__trials__this_stim[coherence]):
                during_idxs = np.where((start_times > (STIM_START + 0)) & (start_times <= STIM_END))[0][:1]
                after_idxs = np.where(start_times > STIM_END)[0][:1]

                for (_bout_pos, bout_idx) in enumerate(during_idxs):
                    times_during.append(start_times[bout_idx])
                    angles_during.append(turn_angles__trials__this_stim[coherence][trail_idx][bout_idx])

                for (_bout_pos, bout_idx) in enumerate(after_idxs):
                    times_after.append(start_times[bout_idx])
                    angles_after.append(turn_angles__trials__this_stim[coherence][trail_idx][bout_idx])

            times_during = np.array(times_during)
            angles_during = np.array(angles_during)

            times_after = np.array(times_after)
            angles_after = np.array(angles_after)

            for bin in range(num_bins):
                mask_during = (times_during >= bin_edges_during[bin]) & (times_during < bin_edges_during[bin + 1])

                angle_during__deg = angles_during[mask_during]

                angle_during__rad = np.deg2rad(angle_during__deg)
                angle_during_vec = np.array([np.cos(angle_during__rad), np.sin(angle_during__rad)])

                turn_dp_during = np.cos(stim__rad - angle_during__rad)
                normalized_turn_dp_during = turn_dp_during - null_dp_mean

                normalized_turn_dp_during = normalized_turn_dp_during * 100

                total_dot_product_mean_during[coherence_idx, bin] = np.nanmean(normalized_turn_dp_during)
                total_dot_product_std_dev_during[coherence_idx, bin] = np.nanstd(normalized_turn_dp_during)
                total_dot_product_count_during[coherence_idx, bin] = len(normalized_turn_dp_during)

                mask_after = (times_after >= bin_edges_after[bin]) & (times_after < bin_edges_after[bin + 1])

                angle_after__deg = angles_after[mask_after]

                angle_after__rad = np.deg2rad(angle_after__deg)
                angle_after_vec = np.array([np.cos(angle_after__rad), np.sin(angle_after__rad)])

                turn_dp_after = np.cos(stim__rad - angle_after__rad)
                normalized_turn_dp_after = turn_dp_after - null_dp_mean

                normalized_turn_dp_after = normalized_turn_dp_after * 100

                total_dot_product_mean_after[coherence_idx, bin] = np.nanmean(normalized_turn_dp_after)
                total_dot_product_std_dev_after[coherence_idx, bin] = np.nanstd(normalized_turn_dp_after)
                total_dot_product_count_after[coherence_idx, bin] = len(normalized_turn_dp_after)

                if stim_val is None:
                    total_correct_bout_mask_during = angle_during__deg > 0
                elif stim_val == 0:
                    total_correct_bout_mask_during = np.abs(angle_during__deg) < CATEGORICAL_THRESHOLDS
                elif stim_val == 180:
                    total_correct_bout_mask_during = np.abs(angle_during__deg) > CATEGORICAL_THRESHOLDS
                elif stim_val > 0:
                    total_correct_bout_mask_during = angle_during__deg > CATEGORICAL_THRESHOLDS
                elif stim_val < 0:
                    total_correct_bout_mask_during = angle_during__deg < -CATEGORICAL_THRESHOLDS

                total_correct_bout_mask_during = total_correct_bout_mask_during * 100

                total_correct_bouts_mean_during[coherence_idx, bin] = np.nanmean(total_correct_bout_mask_during)
                total_correct_bouts_std_dev_during[coherence_idx, bin] = np.nanstd(total_correct_bout_mask_during)
                total_correct_bouts_count_during[coherence_idx, bin] = len(total_correct_bout_mask_during)

                if stim_val is None:
                    total_correct_bout_mask_after = angle_after__deg > 0
                elif stim_val == 0:
                    total_correct_bout_mask_after = np.abs(angle_after__deg) < CATEGORICAL_THRESHOLDS
                elif stim_val == 180:
                    total_correct_bout_mask_after = np.abs(angle_after__deg) > CATEGORICAL_THRESHOLDS
                elif stim_val > 0:
                    total_correct_bout_mask_after = angle_after__deg > CATEGORICAL_THRESHOLDS
                elif stim_val < 0:
                    total_correct_bout_mask_after = angle_after__deg < -CATEGORICAL_THRESHOLDS

                total_correct_bout_mask_after = total_correct_bout_mask_after * 100

                total_correct_bouts_mean_after[coherence_idx, bin] = np.nanmean(total_correct_bout_mask_after)
                total_correct_bouts_std_dev_after[coherence_idx, bin] = np.nanstd(total_correct_bout_mask_after)
                total_correct_bouts_count_after[coherence_idx, bin] = len(total_correct_bout_mask_after)

        cmap = plt.get_cmap("tab10")

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            if stim_val is None:
                color = "gray"
            else:
                color = cmap(coherence_idx)

            ax1ea.plot(
                [bin_centers_during[-1], bin_centers_after[0] - end_shift],
                [total_dot_product_mean_during[coherence_idx, -1], total_dot_product_mean_after[coherence_idx, 0]],
                linewidth=0.6,
                linestyle="dashed",
                color=color,
                alpha=0.2,
            )

            ax1ea.errorbar(
                bin_centers_during,
                total_dot_product_mean_during[coherence_idx],
                yerr=total_dot_product_std_dev_during[coherence_idx]
                / np.sqrt(total_dot_product_count_during[coherence_idx]),
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            ax1ea.errorbar(
                bin_centers_after - end_shift,
                total_dot_product_mean_after[coherence_idx],
                yerr=total_dot_product_std_dev_after[coherence_idx]
                / np.sqrt(total_dot_product_count_after[coherence_idx]),
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            ax1eb.plot(
                [bin_centers_during[-1], bin_centers_after[0] - end_shift],
                [total_correct_bouts_mean_during[coherence_idx, -1], total_correct_bouts_mean_after[coherence_idx, 0]],
                linewidth=0.6,
                linestyle="dashed",
                color=color,
                alpha=0.2,
            )

            ax1eb.errorbar(
                bin_centers_during,
                total_correct_bouts_mean_during[coherence_idx],
                yerr=total_correct_bouts_std_dev_during[coherence_idx]
                / np.sqrt(total_correct_bouts_count_during[coherence_idx]),
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            ax1eb.errorbar(
                bin_centers_after - end_shift,
                total_correct_bouts_mean_after[coherence_idx],
                yerr=total_correct_bouts_std_dev_after[coherence_idx]
                / np.sqrt(total_correct_bouts_count_after[coherence_idx]),
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

        # Define ticks manually.
        ax1ea.set_xticks([STIM_START, STIM_START + stim_showing_duration, STIM_END - end_shift])
        ax1ea.set_xticklabels([STIM_START, STIM_START + stim_showing_duration, STIM_END])
        ax1eb.set_xticks([STIM_START, STIM_START + stim_showing_duration, STIM_END - end_shift])
        ax1eb.set_xticklabels([STIM_START, STIM_START + stim_showing_duration, STIM_END])

# Fig 1f.
# Probability of swimming in the same direction as a function of
# inter-bout interval, for all coherence levels.

if "1f" in FIGS_TO_SHOW:
    if ABS_STIM:
        fig1fa, axes1fa = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
        fig1fb, axes1fb = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
    else:
        fig1fa, axes1fa = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)
        fig1fb, axes1fb = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)

    axes1fa[0, 0].set_ylim(40, 100)
    axes1fb[0, 0].set_ylim(40, 100)

    for stim in stim_keys:
        stim_val = stim_sym_to_int(stim)

        ax1fa = axes1fa[stim_pos[stim]]
        ax1fb = axes1fb[stim_pos[stim]]

        start_times__this_stim = start_times__trials[stim]
        turn_angles__this_stim = turn_angles__trials[stim]

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            prob_rr = []
            prob_ll = []
            ibi_rr = []
            ibi_ll = []

            dps = []
            ibi_tot = []

            for (trial_idx, start_times) in enumerate(start_times__this_stim[coherence]):
                mask = (start_times > (STIM_START)) & (start_times <= STIM_END)
                turn_angles = turn_angles__this_stim[coherence][trial_idx]

                start_times__during = start_times[mask]
                turn_angles__during = turn_angles[mask]

                for (bout_idx, start_time) in enumerate(start_times__during):
                    if bout_idx == 0:
                        continue

                    inter_bout_interval = start_time - start_times__during[bout_idx - 1]

                    if turn_angles__during[bout_idx - 1] > 0:
                        if turn_angles__during[bout_idx] > 0:
                            prob_rr.append(100)
                        else:
                            prob_rr.append(0)

                        ibi_rr.append(inter_bout_interval)

                    else:
                        if turn_angles__during[bout_idx] < 0:
                            prob_ll.append(100)
                        else:
                            prob_ll.append(0)

                        ibi_ll.append(inter_bout_interval)

                    angle_first_bout__rad = np.deg2rad(turn_angles__during[bout_idx - 1])
                    angle_second_bout__rad = np.deg2rad(turn_angles__during[bout_idx])
                    dot_product = np.cos(angle_first_bout__rad - angle_second_bout__rad) * 100

                    dps.append(dot_product)
                    ibi_tot.append(inter_bout_interval)

            # bin the probabilities into 7 bins according to their times: 0-0.2 s, 0.2-0.4 s, 0.4-0.6 s, 0.6-0.8 s, 0.8-1.0 s, 1.0-1.2 s, 1.2+ s.

            prob_rr = np.array(prob_rr)
            prob_ll = np.array(prob_ll)
            ibi_rr = np.array(ibi_rr)
            ibi_ll = np.array(ibi_ll)
            dps = np.array(dps)
            ibi_tot = np.array(ibi_tot)

            num_bins = 6
            bin_size = 0.3

            prob_rr_mean = np.zeros(num_bins)
            prob_rr_std_dev = np.zeros(num_bins)
            prob_rr_count = np.zeros(num_bins)

            prob_ll_mean = np.zeros(num_bins)
            prob_ll_std_dev = np.zeros(num_bins)
            prob_ll_count = np.zeros(num_bins)

            prob_same_mean = np.zeros(num_bins)
            prob_same_std_dev = np.zeros(num_bins)
            prob_same_count = np.zeros(num_bins)

            dps_mean = np.zeros(num_bins)
            dps_std_dev = np.zeros(num_bins)
            dps_count = np.zeros(num_bins)

            for bin in range(num_bins):
                if bin == num_bins - 1:
                    mask_rr = ibi_rr >= bin_size * bin
                    mask_ll = ibi_ll >= bin_size * bin
                    mask_dps = ibi_tot >= bin_size * bin
                else:
                    mask_rr = (ibi_rr >= bin_size * bin) & (ibi_rr < bin_size * (bin + 1))
                    mask_ll = (ibi_ll >= bin_size * bin) & (ibi_ll < bin_size * (bin + 1))
                    mask_dps = (ibi_tot >= bin_size * bin) & (ibi_tot < bin_size * (bin + 1))

                prob_rr_mean[bin] = np.nanmean(prob_rr[mask_rr])
                prob_rr_std_dev[bin] = np.nanstd(prob_rr[mask_rr])
                prob_rr_count[bin] = len(prob_rr[mask_rr])

                prob_ll_mean[bin] = np.nanmean(prob_ll[mask_ll])
                prob_ll_std_dev[bin] = np.nanstd(prob_ll[mask_ll])
                prob_ll_count[bin] = len(prob_ll[mask_ll])

                prob_same_mean[bin] = np.nanmean(np.concatenate((prob_rr[mask_rr], prob_ll[mask_ll])))
                prob_same_std_dev[bin] = np.nanstd(np.concatenate((prob_rr[mask_rr], prob_ll[mask_ll])))
                prob_same_count[bin] = len(np.concatenate((prob_rr[mask_rr], prob_ll[mask_ll])))

                dps_mean[bin] = np.nanmean(dps[mask_dps])
                dps_std_dev[bin] = np.nanstd(dps[mask_dps])
                dps_count[bin] = len(dps[mask_dps])

            cmap = plt.get_cmap("tab10")

            if stim_val is None:
                color = "gray"
            else:
                color = cmap(coherence_idx)

            ax1fa.plot(
                bin_size * np.arange(num_bins) + (bin_size / 2),
                dps_mean,
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            # Define ticks manually.
            # bin_size * np.arange(num_bins) + (bin_size / 2) are the bin centers, and therefore the x-ticks.
            ax1fa.set_xticks(bin_size * np.arange(num_bins) + (bin_size / 2))
            ax1fa.set_xticklabels(
                [f"{bin_size * bin:.1f}-{bin_size * (bin + 1):.1f}" for bin in range(num_bins - 1)]
                + [f"{bin_size * (num_bins - 1):.1f}+"]
            )
            # rotate 60 degrees and make smaller.
            ax1fa.tick_params(axis="x", labelrotation=60, labelsize=6)

            ax1fb.axhline(50, color="gray", linestyle="dotted", linewidth=1, alpha=0.2)

            ax1fb.plot(
                bin_size * np.arange(num_bins) + (bin_size / 2),
                prob_rr_mean,
                linewidth=0.6,
                linestyle="--",
                color=color,
                alpha=0.5,
            )

            ax1fb.plot(
                bin_size * np.arange(num_bins) + (bin_size / 2),
                prob_ll_mean,
                linewidth=0.6,
                linestyle="-.",
                color=color,
                alpha=0.5,
            )

            ax1fb.errorbar(
                bin_size * np.arange(num_bins) + (bin_size / 2),
                prob_same_mean,
                yerr=prob_same_std_dev / np.sqrt(prob_same_count),
                linewidth=1,
                linestyle="solid",
                color=color,
                alpha=1.0,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=3,
            )

            # Define ticks manually.
            # bin_size * np.arange(num_bins) + (bin_size / 2) are the bin centers, and therefore the x-ticks.
            ax1fb.set_xticks(bin_size * np.arange(num_bins) + (bin_size / 2))
            ax1fb.set_xticklabels(
                [f"{bin_size * bin:.1f}-{bin_size * (bin + 1):.1f}" for bin in range(num_bins - 1)]
                + [f"{bin_size * (num_bins - 1):.1f}+"]
            )
            # rotate 60 degrees and make smaller.
            ax1fb.tick_params(axis="x", labelrotation=60, labelsize=6)

plt.show()

# save all figures as svgs so that they can be edited later.

if SAVE_FIGS:
    pass
    # fig_tcf.savefig("fig_tcf.svg", bbox_inches="tight")
    # turn_angle_histogram.savefig("turn_angle_histogram.svg", bbox_inches="tight")
    # fig1b.savefig("fig1b.svg", bbox_inches="tight")
    # fig1ca.savefig("fig1ca.svg", bbox_inches="tight")
    # fig1cb.savefig("fig1cb.svg", bbox_inches="tight")
    # fig1da.savefig("fig1da.svg", bbox_inches="tight")
    # fig1db.savefig("fig1db.svg", bbox_inches="tight")
    # fig1ea.savefig("fig1ea.svg", bbox_inches="tight")
    # fig1eb.savefig("fig1eb.svg", bbox_inches="tight")
    # fig1fa.savefig("fig1fa.svg", bbox_inches="tight")
    # fig1fb.savefig("fig1fb.svg", bbox_inches="tight")
