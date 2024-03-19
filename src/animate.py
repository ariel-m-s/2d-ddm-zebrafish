"""
This script animates the two-dimensional simulator model using matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

from constants import (
    BOUT_DURATION,
    GAMMA,
    M1,
    M2,
    REFRACTORY_PERIOD,
    S1,
    S2,
    SIGMA,
    TIME_STEP,
    Z,
)
from model import Simulator

M_det = np.sqrt(M1 * M2)
M_det_inv = 1 / M_det
R1 = 1 / M1
R2 = 1 / M2

initial_stim = np.array([0, 0])

refractory_period_n_steps = int(REFRACTORY_PERIOD / TIME_STEP)
bout_duration_n_steps = int(BOUT_DURATION / TIME_STEP)

simulator = Simulator()

figs, axs = plt.subplots(1, 2, figsize=(12, 6))

simulator_ax = axs[0]
distr_ax = axs[1]

simulator_ax.set_aspect("equal", adjustable="box")

# rename x,y axes to x_1 and x_2 (with subscripts)
simulator_ax.set_xlabel("$x_1$")
simulator_ax.set_ylabel("$x_2$")

# set ticks only at 0, 0.5 and 1
simulator_ax.set_xticks([-1, 0, 1])
simulator_ax.set_yticks([-1, 0, 1])

distr_ax.set_xlim(-180, 180)
distr_ax.set_ylim(0, 0.2)

figure_bound = 2 * max(R1, R2)
simulator_ax.set_xlim(-figure_bound, figure_bound)
simulator_ax.set_ylim(-figure_bound, figure_bound)

# DECISION VARIABLE TARGET
drift = np.array([S1 * initial_stim[0], S2 * initial_stim[1]])

target_mean = drift / GAMMA
target_std = [SIGMA, SIGMA] / np.sqrt(2 * GAMMA)

print(target_std)

target_ellipse_1std = Ellipse(
    xy=target_mean,
    width=1 * target_std[0],
    height=1 * target_std[1],
    angle=0,
    alpha=0.05,
    color="g",
)
target_ellipse_2std = Ellipse(
    xy=target_mean,
    width=2 * target_std[0],
    height=2 * target_std[1],
    angle=0,
    alpha=0.05,
    color="green",
)
target_ellipse_3std = Ellipse(
    xy=target_mean,
    width=3 * target_std[0],
    height=3 * target_std[1],
    angle=0,
    alpha=0.05,
    color="green",
    # dashed green border
    ls="dashed",
    fill=False,
    lw=1,
)

# simulator_ax.add_artist(target_ellipse_1std)
# simulator_ax.add_artist(target_ellipse_2std)
# simulator_ax.add_artist(target_ellipse_3std)

circle_stim_25coh = plt.Circle((0, 0), 0.25, color="black", fill=False, ls="dashed", alpha=0.1)
circle_stim_50coh = plt.Circle((0, 0), 0.5, color="black", fill=False, ls="dashed", alpha=0.2)
circle_stim_100coh = plt.Circle((0, 0), 1, color="black", fill=False, ls="dashed", alpha=0.4)

# simulator_ax.add_artist(circle_stim_25coh)
# simulator_ax.add_artist(circle_stim_50coh)
# simulator_ax.add_artist(circle_stim_100coh)

ellipse_decision_boundary = Ellipse(
    xy=(-Z[0] * R1, -Z[1] * R2),
    width=2 * R1,
    height=2 * R2,
    angle=0,
    alpha=0.6,
    color="black",
    fill=False,
    ls="solid",
)

simulator_ax.add_artist(ellipse_decision_boundary)

# paths start with a bunch of zeros
max_path_length = 100
path_x = [0] * max_path_length
path_y = [0] * max_path_length

path_line = simulator_ax.plot([], [], lw=1, alpha=0.3, color="darkgoldenrod")[0]
path_tip = simulator_ax.scatter([], [], s=20, color="darkgoldenrod")

stim_pos = simulator_ax.scatter([0], [0], s=5, color="purple")
stim_line = simulator_ax.plot([0, 0], [0, 0], color="purple", ls="dashed", lw=1, alpha=0.5)[0]
stim_text = simulator_ax.text(0.05, 0.95, "", transform=simulator_ax.transAxes, ha="left", va="top")

timer = simulator_ax.text(0.95, 0.95, "".format(0, 0), transform=simulator_ax.transAxes, ha="right", va="top")

bouting_indicator_color = "#FFFF00"

bouting_indicator = distr_ax.plot([0, 0], [0, 0], color=bouting_indicator_color, lw=4, alpha=0.5)[0]

n_last_bout = -np.inf

stim_x = 0
stim_y = 0


def automatic_update(n):
    # global stim_x
    # global stim_y

    # stim_angle = np.arctan2(stim_x, stim_y)
    # stim_radius = np.sqrt(stim_x**2 + stim_y**2)

    # # every 0.5 seconds, change the angle
    # if n % (0.5 / TIME_STEP) == 0:
    #     stim_angle = -stim_angle

    # # orbit the origin
    # stim_angle = stim_angle + (TIME_STEP * GAMMA)

    # stim_x = np.sin(stim_angle) * stim_radius
    # stim_y = np.cos(stim_angle) * stim_radius
    pass


def update(n):
    global stim_x
    global stim_y
    global n_last_bout

    automatic_update(n)

    stim_angle = np.rad2deg(np.arctan2(stim_x, stim_y))
    stim_radius = np.sqrt(stim_x**2 + stim_y**2)

    stim_pos.set_offsets([stim_x, stim_y])
    stim_line.set_data([0, stim_x], [0, stim_y])
    stim_text.set_text("c = {:.2f}\nθ = {:.2f}°".format(stim_radius, stim_angle))

    c = np.array([stim_x, stim_y])
    decision = simulator.step(c=c)
    simulator_x, simulator_y = simulator.x__flat

    path_x.append(simulator_x)
    path_y.append(simulator_y)

    if len(path_x) > max_path_length:
        path_x.pop(0)
        path_y.pop(0)

    max_line_elements = 100
    path_line.set_data(path_x[-max_line_elements:], path_y[-max_line_elements:])
    path_tip.set_offsets([simulator_x, simulator_y])

    time = (n + 1) * TIME_STEP
    timer.set_text("t = {:.2f}".format(time))

    n_since_last_bout = n - n_last_bout

    if decision is not None:
        # If initiating bout
        path_tip.set_color(bouting_indicator_color)
        path_tip.set_edgecolor("black")
        path_tip.set_sizes([200])

        bouting_indicator.set_color(bouting_indicator_color)

        n_last_bout = n

    elif (bout_duration_n_steps != refractory_period_n_steps) and (n_since_last_bout == (bout_duration_n_steps + 1)):
        # If exiting bout
        path_tip.set_color("none")
        path_tip.set_edgecolor("lightgrey")
        bouting_indicator.set_color("lightgrey")

    elif n_since_last_bout == (refractory_period_n_steps + 1):
        # If exiting refractory period
        path_tip.set_sizes([30])
        path_tip.set_color("darkgoldenrod")
        path_tip.set_edgecolor("none")
        bouting_indicator.set_data([0, 0], [0, 0])


def on_move(event):
    global stim_x
    global stim_y

    if event.button != 1:
        return

    if event.inaxes == simulator_ax:
        stim_x = event.xdata
        stim_y = event.ydata


def on_key_press(event):
    global stim_x
    global stim_y

    step = 0.25

    if event.key == "left":
        stim_x = stim_x - step
        stim_y = stim_y
    elif event.key == "right":
        stim_x = stim_x + step
        stim_y = stim_y
    elif event.key == "up":
        stim_x = stim_x
        stim_y = stim_y + step
    elif event.key == "down":
        stim_x = stim_x
        stim_y = stim_y - step


def on_right_click(event):
    global stim_x
    global stim_y

    if event.button != 3:
        return

    stim_x = 0
    stim_y = 0


figs.canvas.mpl_connect("button_press_event", on_move)
figs.canvas.mpl_connect("motion_notify_event", on_move)
figs.canvas.mpl_connect("key_press_event", on_key_press)
figs.canvas.mpl_connect("button_press_event", on_right_click)


grid_1 = np.linspace(-figure_bound, figure_bound, 1000)

# feature_vectors = boutfield([(y, x) for x in grid_1 for y in grid_1])

# params = np.apply_along_axis(lambda fv: gmm.main_norm_params(*fv), 1, feature_vectors)

# mus = params[:, 0]

# mus are the values we want to plot

# reshape mus into a 2d array
# mus = mus.reshape((len(grid_1), len(grid_1)))

# rotate matrix by -90 degrees
# mus = np.rot90(mus, k=1)

# plot the heatmap. The values are angles from -max_angle to max_angle. Select colormap accordingly.

max_angle = 90
cmap = "bwr"

# simulator_ax.imshow(
#     mus,
#     cmap=cmap,
#     vmin=-max_angle,
#     vmax=max_angle,
#     extent=[-figure_bound, figure_bound, -figure_bound, figure_bound],
#     alpha=0.8,
# )

colorbar_pos = 0.115
colorbar_width = 0.01

distr_ax.imshow(
    np.linspace(-max_angle, max_angle, 1000).reshape(1, -1),
    cmap=cmap,
    vmin=-max_angle,
    vmax=max_angle,
    extent=[-max_angle, max_angle, colorbar_pos, colorbar_pos + colorbar_width],
    aspect="auto",
)

DURATION = 20
N_FRAMES = int(DURATION / TIME_STEP)

# for i in range(N_FRAMES):
#     update(i)

# we have path_x and path_y. These are the x and y components
# of samples of a bivariate normal distribution. Let's
# find the distribution. Let's fit a bivariate normal distribution
# to these samples and plot it.

# fit a bivariate normal distribution to the samples
# of the bivariate normal distribution. Print the parameters.

# exclude the first 5000 samples
# path_x = np.array(path_x[5000:])
# path_y = np.array(path_y[5000:])

# mean = np.array([np.mean(path_x), np.mean(path_y)])
# cov = np.cov(path_x, path_y)

# # print stds
# stds = np.sqrt(np.diag(cov))
# print(stds)

# concatenate path_x and path_y and find std
# of the samples. This is the std of the bivariate
# normal distribution.

# samples = np.concatenate([path_x, path_y])
# std = np.std(samples)

# print(std)

MODE = "save_image"
MODE = "animate"

if MODE == "save_image":
    # don't show animation, just run it and save the last frame as svg
    for i in range(N_FRAMES):
        update(i)

    # path_x = np.array(path_x[5000:])
    # path_y = np.array(path_y[5000:])
    # samples = np.concatenate([path_x, path_y])
    # std = np.std(samples)
    # print(std)

    plt.savefig("simulator.svg", format="svg")

elif MODE == "animate":
    anim = animation.FuncAnimation(
        figs,
        update,
        N_FRAMES,
        interval=TIME_STEP * 1000,
        repeat=False,
    )
    plt.show()
