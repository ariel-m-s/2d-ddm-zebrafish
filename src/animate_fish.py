import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from experiment import STIM_IDS_EXPERIMENT_0
from load_data import get_trial_data, load_trials

FISH_SIZE = 4 / 60  # 4 mm
CIRCLE_RADIUS = 1.0
DISH_UNITS_IN_MM = 120
STIM_START = 5
STIM_END = 15


def animate_trajectory(fish_pos_x, fish_pos_y, fish_orient, stim_orient__deg, fps):
    if stim_orient__deg in [-45, -90, -135, 45, 90, 135]:
        stim_orient__deg = -stim_orient__deg
    # Create a figure and axes
    fig, ax = plt.subplots()

    # make the plot axis always square
    ax.set_aspect("equal", adjustable="box")

    # make it big
    fig.set_size_inches(10, 8)

    # Set the axis limits
    ax.set_xlim((-CIRCLE_RADIUS * 1.5, CIRCLE_RADIUS * 1.5))
    ax.set_ylim((-CIRCLE_RADIUS * 1.5, CIRCLE_RADIUS * 1.5))

    # Create a circle with radius 12 cm
    circle = plt.Circle((0, 0), radius=CIRCLE_RADIUS, fill=False)
    ax.add_patch(circle)

    # Create a line
    (trajectory,) = ax.plot([], [], "r-", linewidth=1, color="lightblue")

    # Create a dot and a line to show orientation
    (fish_head,) = ax.plot([], [], "ro", markersize=4)
    (fish_tail,) = ax.plot([], [], "r-")

    # Stimulus dot and line. This is an arrow that shows the direction of the stimulus.
    # It should be in front of the fish. It respresents what the fish sees.
    # make arrow head larger.
    # Put these above the rest of the object so that they are always visible.
    (stim_head,) = ax.plot([], [], "b>", markersize=4)
    (stim_tail,) = ax.plot([], [], "b-", linewidth=1, linestyle="dashed")

    # in the corner of the plot, show the time
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # Write stimulus in corner
    if stim_orient__deg is not None:
        ax.text(0.02, 0.90, f"Stimulus: {-stim_orient__deg}°", transform=ax.transAxes)
    else:
        ax.text(0.02, 0.90, f"Stimulus: None", transform=ax.transAxes)

    # Add available distance to the corner
    available_distance_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)
    radius_text = ax.text(0.02, 0.80, "", transform=ax.transAxes)

    def update(frame):
        trajectory.set_data(fish_pos_x[: frame + 1], fish_pos_y[: frame + 1])

        time = frame / fps

        time_text.set_text(f"Time: {time:.2f} s")

        fish_head_x = fish_pos_x[frame]
        fish_head_y = fish_pos_y[frame]

        fish_head.set_data(fish_head_x, fish_head_y)

        fish_orient__deg = fish_orient[frame]
        fish_orient__rad = np.deg2rad(fish_orient__deg)
        fish_orient_x = np.cos(fish_orient__rad)
        fish_orient_y = np.sin(fish_orient__rad)
        fish_orient_line_x = fish_head_x - FISH_SIZE * fish_orient_x
        fish_orient_line_y = fish_head_y - FISH_SIZE * fish_orient_y

        available_distance = np.sqrt(
            ((fish_head_x - np.cos(fish_orient__rad)) ** 2 + (fish_head_y - np.sin(fish_orient__rad)) ** 2)
        )
        radius = np.sqrt(fish_head_x**2 + fish_head_y**2)

        available_distance_text.set_text(f"Available distance: {available_distance:.2f}")
        radius_text.set_text(f"Radius: {radius:.2f}")

        fish_tail.set_data([fish_head_x, fish_orient_line_x], [fish_head_y, fish_orient_line_y])

        if stim_orient__deg is not None:
            stim_orient__rad = np.deg2rad(stim_orient__deg)
            stim_centroid_x = fish_head_x + 4 * FISH_SIZE * fish_orient_x
            stim_centroid_y = fish_head_y + 4 * FISH_SIZE * fish_orient_y
            stim_orient_x = np.cos(stim_orient__rad + fish_orient__rad)
            stim_orient_y = np.sin(stim_orient__rad + fish_orient__rad)
            stim_head_x = stim_centroid_x + 1 * FISH_SIZE * stim_orient_x
            stim_head_y = stim_centroid_y + 1 * FISH_SIZE * stim_orient_y
            stim_tail_x = stim_centroid_x - 1 * FISH_SIZE * stim_orient_x
            stim_tail_y = stim_centroid_y - 1 * FISH_SIZE * stim_orient_y

            if STIM_START <= time <= STIM_END:
                stim_head.set_data(stim_head_x, stim_head_y)
                # marker orientation
                stim_head.set_marker((3, 0, np.rad2deg(stim_orient__rad + fish_orient__rad) - 90))
                stim_tail.set_data([stim_head_x, stim_tail_x], [stim_head_y, stim_tail_y])
            else:
                stim_head.set_data([], [])
                stim_tail.set_data([], [])

        return trajectory, fish_head, fish_tail, stim_head, stim_tail, time_text, available_distance_text, radius_text

    # Create an animation
    anim = animation.FuncAnimation(fig, update, frames=len(fish_pos_x), interval=(1000 / fps), blit=True)

    return anim


if __name__ == "__main__":
    # Load the trials.
    trials = load_trials("../behavior/free_swimming_8fish_random_dot_kinematogram_data/org/trials.csv")

    for _ in range(100):
        trial_id = np.random.choice(trials.index)

        # Load the data.
        all_data = get_trial_data(
            data_path="../behavior/free_swimming_8fish_random_dot_kinematogram_data/org", trial_id=trial_id
        )

        # Let's choose one stimulus. Can be None, 0, ±45, ±90, ±135 and 180.
        stim = np.random.choice([-135, -90, -45, 0, 45, 90, 135, 180, None])
        stim_id = STIM_IDS_EXPERIMENT_0[(stim, 1) if stim is not None else stim]

        # Open the data based on camera timestamp.
        raw_data = all_data[f"raw_stimulus_{stim_id}"]

        # Extract the data we need.
        fish_pos_x = raw_data["fish_position_x"]
        fish_pos_y = raw_data["fish_position_y"]
        fish_orient = raw_data["fish_accumulated_orientation"]
        fps = raw_data["camera_fps"][0]

        # Plot trajectory.
        anim = animate_trajectory(fish_pos_x, fish_pos_y, fish_orient, stim, fps)
        plt.show()
