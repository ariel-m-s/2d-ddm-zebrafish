from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from load_data import extract_bouts

SAVE_FIGS = True

FIGS_TO_SHOW = [
    "bout-categories",
    "angle-histogram",
    "bahl-engert-1b",
    "bahl-engert-1c",
    "bahl-engert-1d",
    "bahl-engert-1e",
    "bahl-engert-1f",
]

EXPERIMENT_IDX = 2

if EXPERIMENT_IDX == 0:
    N_COHERENCE_LEVELS = 1
    COHERENCES = [1]
elif EXPERIMENT_IDX == 1:
    N_COHERENCE_LEVELS = 2
    COHERENCES = [0.25, 1]
elif EXPERIMENT_IDX == 2:
    N_COHERENCE_LEVELS = 5
    COHERENCES = [0.25, 0.5, 1]
else:
    raise ValueError("Invalid EXPERIMENT_IDX")

DIRECTORY_NAME = "../behavior/free_swimming_8fish_random_dot_kinematogram_data/org"
# DATA_DIR = "../2d-ddm-zebrafish/simulated_data"

angles__trials, times__trials, angles__all, times__all = extract_bouts(DIRECTORY_NAME, EXPERIMENT_IDX)

DECISION_SEPARATION = 10.3

STIM_START = 5
STIM_END = 15

STIM_POS = {
    None: (1, 0),
    0: (0, 0),
    180: (2, 0),
    "abs_45": (0, 1),
    "abs_90": (1, 1),
    "abs_135": (2, 1),
}

STIM_KEYS = list(STIM_POS.keys())

if "bout-categories" in FIGS_TO_SHOW:
    fig_tcf, axes_tcf = plt.subplots(3, 2, figsize=(4, 7), sharex=True, sharey=True)

    n_bins = 20
    domain_size = 20

    bin_edges = np.linspace(0, domain_size, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_size = domain_size / n_bins

    # total number of bouts for the None stimulus
    n_bouts = len(times__all[None][1])
    baseline = n_bouts / n_bins

    for stim in STIM_KEYS:
        print(stim)

        axes_tcf[STIM_POS[stim]].axvspan(STIM_START, STIM_END, color="lightgray", alpha=0.5)

        frequency_means_left = np.zeros((N_COHERENCE_LEVELS, n_bins))
        frequency_std_errs_left = np.zeros((N_COHERENCE_LEVELS, n_bins))

        frequency_means_right = np.zeros((N_COHERENCE_LEVELS, n_bins))
        frequency_std_errs_right = np.zeros((N_COHERENCE_LEVELS, n_bins))

        frequency_means_forward = np.zeros((N_COHERENCE_LEVELS, n_bins))
        frequency_std_errs_forward = np.zeros((N_COHERENCE_LEVELS, n_bins))

        frequency_means_total = np.zeros((N_COHERENCE_LEVELS, n_bins))
        frequency_std_errs_total = np.zeros((N_COHERENCE_LEVELS, n_bins))

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            for bin in range(n_bins):
                mask = (times__all[stim][coherence] > bin_edges[bin]) & (
                    times__all[stim][coherence] <= bin_edges[bin + 1]
                )

                turn_angles__filtered = angles__all[stim][coherence][mask]

                left_mask = turn_angles__filtered < DECISION_SEPARATION
                right_mask = turn_angles__filtered > DECISION_SEPARATION
                forward_mask = (~left_mask) & (~right_mask)

                # We want percentages relative to baseline. if equal to baseline, then 1. if half, then 0.5, if double, then 2.

                percentage_left = np.sum(left_mask) / baseline * 1
                percentage_right = np.sum(right_mask) / baseline * 1
                percentage_forward = np.sum(forward_mask) / baseline * 1
                percentage_total = np.sum(mask) / baseline * 1

                frequency_means_left[coherence_idx, bin] = percentage_left
                frequency_means_right[coherence_idx, bin] = percentage_right
                frequency_means_forward[coherence_idx, bin] = percentage_forward
                frequency_means_total[coherence_idx, bin] = percentage_total

                # error is 0
                frequency_std_errs_left[coherence_idx, bin] = 0
                frequency_std_errs_right[coherence_idx, bin] = 0
                frequency_std_errs_forward[coherence_idx, bin] = 0
                frequency_std_errs_total[coherence_idx, bin] = 0

                if stim in ["abs_45", "abs_90", "abs_135"]:
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
            opacity = coherence / 1

            axes_tcf[STIM_POS[stim]].errorbar(
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

            axes_tcf[STIM_POS[stim]].errorbar(
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

            axes_tcf[STIM_POS[stim]].errorbar(
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
            axes_tcf[STIM_POS[stim]].errorbar(
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

            axes_tcf[STIM_POS[stim]].set_xlim(0, 20)

            axes_tcf[STIM_POS[stim]].set_xticks([0, 5, 10, 15, 20])
            axes_tcf[STIM_POS[stim]].set_xticklabels([0, 5, 10, 15, 20])

            axes_tcf[STIM_POS[stim]].set_xlabel("Time (s)")
            axes_tcf[STIM_POS[stim]].set_ylabel("Percentage of bouts")


if "fig_angle_histogram" in FIGS_TO_SHOW:
    fig_angle_histogram, axes = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)

    # We want to plot the histogram of turn angles for each coherence level, for each stimulus direction.
    # These will be distributed between -180 and 180 degrees, and bins will be 10 degrees wide.
    # Use np.histogram to compute the histogram, and then plot the histogram using a line plot.

    # For each stimulus direction, plot the histogram for each coherence level, using a different color for each coherence level.

    n_bins = 180

    for stim in STIM_KEYS:
        for (coherence_idx, coherence) in enumerate(COHERENCES):
            start_times__this = np.array(times__all[stim][coherence])
            turn_angles__this = np.array(angles__all[stim][coherence])

            mask = (start_times__this > (STIM_START + 1)) & (start_times__this <= (STIM_END - 6))
            turn_angles__filtered = turn_angles__this[mask]

            hist, bin_edges = np.histogram(turn_angles__filtered, bins=n_bins, range=(-180, 180), density=False)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            if stim in ["abs_45", "abs_90", "abs_135"]:
                hist = hist / 2

            if stim is None:
                color = "gray"
            else:
                color = plt.get_cmap("tab10")(coherence_idx)

            axes[STIM_POS[stim]].plot(bin_centers, hist, color=color, linewidth=1, linestyle="solid", alpha=0.7)

            axes[STIM_POS[stim]].set_xlim(-135, 135)
            axes[STIM_POS[stim]].set_xticks([-90, 0, 90])
            axes[STIM_POS[stim]].set_xticklabels([-90, 0, 90])

            axes[STIM_POS[stim]].set_ylabel("Count")

            axes[STIM_POS[stim]].set_title(f"Stimulus: {stim}º")

            # Make y-axis 0 be at x axis (meaning, no negative values).
            axes[STIM_POS[stim]].spines["bottom"].set_position("zero")

            # print a number in the corner indicating the total count of turn angles
            # in the histogram. print in the same color as the histogram and separated.

            n_bouts = len(turn_angles__filtered)

            if stim in ["abs_45", "abs_90", "abs_135"]:
                n_bouts /= 2

            # plot vertical line at the maximum of the histogram
            max_idx = np.argmax(hist)
            max_angle = bin_centers[max_idx]

            axes[STIM_POS[stim]].axvline(
                max_angle,
                color=color,
                linewidth=1,
                linestyle="dashed",
                alpha=0.5,
            )

            axes[STIM_POS[stim]].text(
                # change position depeding on coherence level, so the text doesn't overlap
                0.95,
                0.95 - (0.1 * coherence_idx),
                n_bouts,
                horizontalalignment="right",
                verticalalignment="top",
                transform=axes[STIM_POS[stim]].transAxes,
                color=color,
            )

# Fig. bahl-engert-1b.
# Bouting frequency as a function of time, for all coherence levels.

if "bahl-engert-1b" in FIGS_TO_SHOW:
    fig_be_1b, axes1b = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)

    n_bins = 20
    domain_size = 20

    bin_edges = np.linspace(0, domain_size, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_size = domain_size / n_bins

    for stim in STIM_KEYS:
        axes1b[STIM_POS[stim]].axvspan(STIM_START, STIM_END, color="lightgray", alpha=0.5)

        frequency_means = np.zeros((N_COHERENCE_LEVELS, n_bins))
        frequency_std_errs = np.zeros((N_COHERENCE_LEVELS, n_bins))

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            for bin in range(n_bins):
                frequencies_per_trial = []

                for trial_idx in range(len(times__trials[stim][coherence])):

                    start_times = times__trials[stim][coherence][trial_idx]
                    mask = (start_times > bin_edges[bin]) & (start_times <= bin_edges[bin + 1])
                    frequencies_per_trial.append(np.sum(mask) / bin_size)

                frequency_means[coherence_idx, bin] = np.nanmean(frequencies_per_trial)
                frequency_std_errs[coherence_idx, bin] = np.nanstd(frequencies_per_trial) / np.sqrt(
                    len(frequencies_per_trial)
                )

        cmap = plt.get_cmap("tab10")

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            if stim is None:
                color = "gray"
            else:
                color = cmap(coherence_idx)

            axes1b[STIM_POS[stim]].errorbar(
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
            axes1b[STIM_POS[stim]].axhline(
                np.nanmean(frequency_means[coherence_idx, STIM_START + 2 : STIM_END]),
                color=color,
                linewidth=1,
                linestyle="dashed",
                alpha=0.5,
            )

            axes1b[STIM_POS[stim]].set_xlabel("Time (s)")
            axes1b[STIM_POS[stim]].set_ylabel("Bouts / s")

# Fig. bahl-engert-1c.
# Time-binned accuracy as a function of time, for all coherence levels.

if "bahl-engert-1c" in FIGS_TO_SHOW:
    fig_be_1c, axes1c = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)

    start_times__null_stim = times__all[None]
    turn_angles__null_stim = angles__all[None]

    num_bins = 20
    bin_edges = np.linspace(0, 20, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for stim in STIM_KEYS:
        ax1c = axes1c[STIM_POS[stim]]
        ax1c.axvspan(STIM_START, STIM_END, color="lightgray", alpha=0.5)

        start_times__this_stim = times__all[stim]
        turn_angles__this_stim = angles__all[stim]

        total_correct_bouts_mean = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_std_err = np.zeros((N_COHERENCE_LEVELS, num_bins))

        total_bouts = np.zeros((N_COHERENCE_LEVELS, num_bins))

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            for bin in range(num_bins):
                mask = (start_times__this_stim[coherence] > bin_edges[bin]) & (
                    start_times__this_stim[coherence] <= bin_edges[bin + 1]
                )

                turn__deg = turn_angles__this_stim[coherence][mask]

                if stim is None:
                    total_correct_bout_mask = turn__deg > 0
                elif stim == 0:
                    total_correct_bout_mask = np.abs(turn__deg) < DECISION_SEPARATION
                elif stim == 180:
                    total_correct_bout_mask = np.abs(turn__deg) > DECISION_SEPARATION
                else:
                    total_correct_bout_mask = turn__deg > DECISION_SEPARATION

                # Map false to 0 and true to 1

                total_correct_bout_mask = total_correct_bout_mask * 1

                total_correct_bouts_mean[coherence_idx, bin] = np.nanmean(total_correct_bout_mask)
                total_correct_bouts_std_err[coherence_idx, bin] = np.std(total_correct_bout_mask) / np.sqrt(
                    len(total_correct_bout_mask)
                )

            total_bouts[coherence_idx, bin] = len(turn__deg)

        cmap = plt.get_cmap("tab10")

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            if stim is None:
                color = "gray"
            else:
                color = cmap(coherence_idx)

            ax1c.axhline(
                np.nanmean(total_correct_bouts_mean[coherence_idx, STIM_START:STIM_END]),
                color=color,
                linewidth=1,
                linestyle="dashed",
                alpha=0.5,
            )

            ax1c.errorbar(
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

# Fig. bahl-engert-1d.
# Accuracy over consecutive bouts, for all coherence levels.

if "bahl-engert-1d" in FIGS_TO_SHOW:
    fig_be_1d, axes1d = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)

    start_times__null_stim = times__all[None]
    turn_angles__null_stim = angles__all[None]

    num_bins = 3

    for stim in STIM_KEYS:

        stim_showing_duration = 3
        end_shift = 6
        stim_gap = STIM_END - STIM_START - stim_showing_duration - end_shift

        ax1d = axes1d[STIM_POS[stim]]
        ax1d.axvspan(STIM_START, STIM_START + stim_showing_duration, color="lightgray", alpha=0.5)
        ax1d.axvspan(STIM_END - end_shift - (stim_gap / 7), STIM_END - end_shift, color="lightgray", alpha=0.5)

        # Show x axis from (START_TIME - 1) to (END_TIME + 4) seconds.
        ax1d.set_xlim(STIM_START - 1, STIM_END + 4 - end_shift)

        times__trials__this_stim = times__trials[stim]
        angles__trials__this_stim = angles__trials[stim]

        # Let's find, for coherence level 1, all the first bouts that start after STIM_START.

        total_correct_bouts_mean_during = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_std_dev_during = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_count_during = np.zeros((N_COHERENCE_LEVELS, num_bins))

        total_correct_bouts_mean_after = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_std_dev_after = np.zeros((N_COHERENCE_LEVELS, num_bins))
        total_correct_bouts_count_after = np.zeros((N_COHERENCE_LEVELS, num_bins))

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            during__deg = defaultdict(list)
            after__deg = defaultdict(list)

            for (trail_idx, start_times) in enumerate(times__trials__this_stim[coherence]):
                # Find the indices of the first three bouts that start after STIM_START and the first three bouts that start after STIM_END.
                # The indices should be relative to the start_times array.

                during_idxs = np.where((start_times > (STIM_START + 0)) & (start_times <= STIM_END))[0][:num_bins]
                after_idxs = np.where(start_times >= STIM_END)[0][:num_bins]

                # print(start_times)
                # print(during_idxs)
                # print(angles__trials__this_stim[coherence][trail_idx])
                # print()

                for (bout_pos, bout_idx) in enumerate(during_idxs):
                    during__deg[bout_pos].append(angles__trials__this_stim[coherence][trail_idx][bout_idx])

                for (bout_pos, bout_idx) in enumerate(after_idxs):
                    after__deg[bout_pos].append(angles__trials__this_stim[coherence][trail_idx][bout_idx])

            for bout_pos in range(3):
                if stim is None:
                    total_correct_bout_mask_during = np.array(during__deg[bout_pos]) > 0
                    total_correct_bout_mask_after = np.array(after__deg[bout_pos]) > 0
                elif stim == 0:
                    total_correct_bout_mask_during = np.abs(np.array(during__deg[bout_pos])) < DECISION_SEPARATION
                    total_correct_bout_mask_after = np.abs(np.array(after__deg[bout_pos])) < DECISION_SEPARATION
                elif stim == 180:
                    total_correct_bout_mask_during = np.abs(np.array(during__deg[bout_pos])) > DECISION_SEPARATION
                    total_correct_bout_mask_after = np.abs(np.array(after__deg[bout_pos])) > DECISION_SEPARATION
                else:
                    total_correct_bout_mask_during = np.array(during__deg[bout_pos]) > DECISION_SEPARATION
                    total_correct_bout_mask_after = np.array(after__deg[bout_pos]) > DECISION_SEPARATION

                # Map false to 0 and true to 1

                total_correct_bout_mask_during = total_correct_bout_mask_during * 1
                total_correct_bout_mask_after = total_correct_bout_mask_after * 1

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

            if stim is None:
                color = "gray"
            else:
                color = cmap(coherence_idx)

            ax1d.plot(
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

            ax1d.errorbar(
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

            ax1d.errorbar(
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
            ax1d.set_xticks(
                [
                    STIM_START + 0.5,
                    STIM_START + 1.5,
                    STIM_START + 2.5,
                    STIM_END + 0.5 - end_shift,
                    STIM_END + 1.5 - end_shift,
                    STIM_END + 2.5 - end_shift,
                ]
            )
            ax1d.set_xticklabels(["1º", "2º", "3º", "1º", "2º", "3º"])

# Fig. bahl-engert-1e.
# Accuracy of the first bout during the stimulus and the first bout after
# the stimulus end as a function of delay, for all coherence levels.

if "bahl-engert-1e" in FIGS_TO_SHOW:
    fig_be_1e, axes1e = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)

    start_times__null_stim = times__all[None]
    turn_angles__null_stim = angles__all[None]

    stim_showing_duration = 2
    end_shift = 7
    stim_gap = STIM_END - STIM_START - stim_showing_duration - end_shift
    num_bins = 4

    bin_edges_during = np.linspace(STIM_START, STIM_START + stim_showing_duration, num_bins + 1)
    bin_edges_after = np.linspace(STIM_END, STIM_END + stim_showing_duration, num_bins + 1)

    bin_centers_during = (bin_edges_during[:-1] + bin_edges_during[1:]) / 2
    bin_centers_after = (bin_edges_after[:-1] + bin_edges_after[1:]) / 2

    for stim in STIM_KEYS:

        ax1e = axes1e[STIM_POS[stim]]
        ax1e.axvspan(STIM_START, STIM_START + stim_showing_duration, color="lightgray", alpha=0.5)
        ax1e.axvspan(STIM_END - end_shift - (stim_gap / 7), STIM_END - end_shift, color="lightgray", alpha=0.5)

        # same as Fig. bahl-engert-1c, but we only keep the first bouts during and after the stimulus.

        times__trials__this_stim = times__trials[stim]
        angles__trials__this_stim = angles__trials[stim]

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

            for (trail_idx, start_times) in enumerate(times__trials__this_stim[coherence]):
                during_idxs = np.where((start_times > (STIM_START + 0)) & (start_times <= STIM_END))[0][:1]
                after_idxs = np.where(start_times > STIM_END)[0][:1]

                for (_bout_pos, bout_idx) in enumerate(during_idxs):
                    times_during.append(start_times[bout_idx])
                    angles_during.append(angles__trials__this_stim[coherence][trail_idx][bout_idx])

                for (_bout_pos, bout_idx) in enumerate(after_idxs):
                    times_after.append(start_times[bout_idx])
                    angles_after.append(angles__trials__this_stim[coherence][trail_idx][bout_idx])

            times_during = np.array(times_during)
            angles_during = np.array(angles_during)

            times_after = np.array(times_after)
            angles_after = np.array(angles_after)

            for bin in range(num_bins):
                mask_during = (times_during >= bin_edges_during[bin]) & (times_during < bin_edges_during[bin + 1])

                angle_during__deg = angles_during[mask_during]

                mask_after = (times_after >= bin_edges_after[bin]) & (times_after < bin_edges_after[bin + 1])

                angle_after__deg = angles_after[mask_after]

                if stim is None:
                    total_correct_bout_mask_during = angle_during__deg > 0
                elif stim == 0:
                    total_correct_bout_mask_during = np.abs(angle_during__deg) < DECISION_SEPARATION
                elif stim == 180:
                    total_correct_bout_mask_during = np.abs(angle_during__deg) > DECISION_SEPARATION
                else:
                    total_correct_bout_mask_during = angle_during__deg > DECISION_SEPARATION

                total_correct_bout_mask_during = total_correct_bout_mask_during * 1

                total_correct_bouts_mean_during[coherence_idx, bin] = np.nanmean(total_correct_bout_mask_during)
                total_correct_bouts_std_dev_during[coherence_idx, bin] = np.nanstd(total_correct_bout_mask_during)
                total_correct_bouts_count_during[coherence_idx, bin] = len(total_correct_bout_mask_during)

                if stim is None:
                    total_correct_bout_mask_after = angle_after__deg > 0
                elif stim == 0:
                    total_correct_bout_mask_after = np.abs(angle_after__deg) < DECISION_SEPARATION
                elif stim == 180:
                    total_correct_bout_mask_after = np.abs(angle_after__deg) > DECISION_SEPARATION
                else:
                    total_correct_bout_mask_after = angle_after__deg > DECISION_SEPARATION

                total_correct_bout_mask_after = total_correct_bout_mask_after * 1

                total_correct_bouts_mean_after[coherence_idx, bin] = np.nanmean(total_correct_bout_mask_after)
                total_correct_bouts_std_dev_after[coherence_idx, bin] = np.nanstd(total_correct_bout_mask_after)
                total_correct_bouts_count_after[coherence_idx, bin] = len(total_correct_bout_mask_after)

        cmap = plt.get_cmap("tab10")

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            if stim is None:
                color = "gray"
            else:
                color = cmap(coherence_idx)

            ax1e.plot(
                [bin_centers_during[-1], bin_centers_after[0] - end_shift],
                [total_correct_bouts_mean_during[coherence_idx, -1], total_correct_bouts_mean_after[coherence_idx, 0]],
                linewidth=0.6,
                linestyle="dashed",
                color=color,
                alpha=0.2,
            )

            ax1e.errorbar(
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

            ax1e.errorbar(
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
        ax1e.set_xticks([STIM_START, STIM_START + stim_showing_duration, STIM_END - end_shift])
        ax1e.set_xticklabels([STIM_START, STIM_START + stim_showing_duration, STIM_END])

# Fig bahl-engert-1f.
# Probability of swimming in the same direction as a function of
# inter-bout interval, for all coherence levels.

if "bahl-engert-1f" in FIGS_TO_SHOW:
    fig_be_1f, axes1f = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)

    for stim in STIM_KEYS:

        ax1f = axes1f[STIM_POS[stim]]

        start_times__this_stim = times__trials[stim]
        turn_angles__this_stim = angles__trials[stim]

        for (coherence_idx, coherence) in enumerate(COHERENCES):
            prob_rr = []
            prob_ll = []
            inter_bout_interval_rr = []
            inter_bout_interval_ll = []

            inter_bout_interval_tot = []

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
                            prob_rr.append(1)
                        else:
                            prob_rr.append(0)

                        inter_bout_interval_rr.append(inter_bout_interval)

                    else:
                        if turn_angles__during[bout_idx] < 0:
                            prob_ll.append(1)
                        else:
                            prob_ll.append(0)

                        inter_bout_interval_ll.append(inter_bout_interval)

                    angle_first_bout__rad = np.deg2rad(turn_angles__during[bout_idx - 1])
                    angle_second_bout__rad = np.deg2rad(turn_angles__during[bout_idx])

                    inter_bout_interval_tot.append(inter_bout_interval)

            # bin the probabilities into 7 bins according to their times: 0-0.2 s, 0.2-0.4 s, 0.4-0.6 s, 0.6-0.8 s, 0.8-1.0 s, 1.0-1.2 s, 1.2+ s.

            prob_rr = np.array(prob_rr)
            prob_ll = np.array(prob_ll)
            inter_bout_interval_rr = np.array(inter_bout_interval_rr)
            inter_bout_interval_ll = np.array(inter_bout_interval_ll)
            inter_bout_interval_tot = np.array(inter_bout_interval_tot)

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

            for bin in range(num_bins):
                if bin == num_bins - 1:
                    mask_rr = inter_bout_interval_rr >= bin_size * bin
                    mask_ll = inter_bout_interval_ll >= bin_size * bin
                else:
                    mask_rr = (inter_bout_interval_rr >= bin_size * bin) & (
                        inter_bout_interval_rr < bin_size * (bin + 1)
                    )
                    mask_ll = (inter_bout_interval_ll >= bin_size * bin) & (
                        inter_bout_interval_ll < bin_size * (bin + 1)
                    )

                prob_rr_mean[bin] = np.nanmean(prob_rr[mask_rr])
                prob_rr_std_dev[bin] = np.nanstd(prob_rr[mask_rr])
                prob_rr_count[bin] = len(prob_rr[mask_rr])

                prob_ll_mean[bin] = np.nanmean(prob_ll[mask_ll])
                prob_ll_std_dev[bin] = np.nanstd(prob_ll[mask_ll])
                prob_ll_count[bin] = len(prob_ll[mask_ll])

                prob_same_mean[bin] = np.nanmean(np.concatenate((prob_rr[mask_rr], prob_ll[mask_ll])))
                prob_same_std_dev[bin] = np.nanstd(np.concatenate((prob_rr[mask_rr], prob_ll[mask_ll])))
                prob_same_count[bin] = len(np.concatenate((prob_rr[mask_rr], prob_ll[mask_ll])))

            cmap = plt.get_cmap("tab10")

            if stim is None:
                color = "gray"
            else:
                color = cmap(coherence_idx)

            ax1f.axhline(0.5, color="gray", linestyle="dotted", linewidth=1, alpha=0.2)

            ax1f.plot(
                bin_size * np.arange(num_bins) + (bin_size / 2),
                prob_rr_mean,
                linewidth=0.6,
                linestyle="--",
                color=color,
                alpha=0.5,
            )

            ax1f.plot(
                bin_size * np.arange(num_bins) + (bin_size / 2),
                prob_ll_mean,
                linewidth=0.6,
                linestyle="-.",
                color=color,
                alpha=0.5,
            )

            ax1f.errorbar(
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
            ax1f.set_xticks(bin_size * np.arange(num_bins) + (bin_size / 2))
            ax1f.set_xticklabels(
                [f"{bin_size * bin:.1f}-{bin_size * (bin + 1):.1f}" for bin in range(num_bins - 1)]
                + [f"{bin_size * (num_bins - 1):.1f}+"]
            )
            # rotate 60 degrees and make smaller.
            ax1f.tick_params(axis="x", labelrotation=60, labelsize=6)

plt.show()

# save all figures as svgs so that they can be edited later.

if SAVE_FIGS:
    fig_tcf.savefig("fig_tcf.svg", bbox_inches="tight")
    fig_angle_histogram.savefig("fig_angle_histogram.svg", bbox_inches="tight")
    fig_be_1b.savefig("fig_be_1b.svg", bbox_inches="tight")
    fig_be_1c.savefig("fig_be_1c.svg", bbox_inches="tight")
    fig_be_1c.savefig("fig_be_1c.svg", bbox_inches="tight")
    fig_be_1d.savefig("fig_be_1d.svg", bbox_inches="tight")
    fig_be_1e.savefig("fig_be_1e.svg", bbox_inches="tight")
    fig_be_1f.savefig("fig_be_1f.svg", bbox_inches="tight")
