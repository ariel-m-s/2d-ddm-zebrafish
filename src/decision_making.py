from enum import Enum

import numpy as np

import constants


class DecisionCategory(Enum):
    LEFT = -1
    FORWARD = 0
    RIGHT = +1


def get_alpha(x: np.array) -> np.array:
    return np.arctan2(x[1], x[0])


def is_subthreshold(x: np.array, M: np.array, z: np.array) -> np.array:
    ellipse = np.matmul(M, x) + z
    return np.linalg.norm(ellipse, axis=0) < 1


def decision_category__alpha(alpha: np.array, phi: float) -> np.array:
    left = alpha < -phi
    right = alpha > phi

    decisions = np.full(alpha.shape, DecisionCategory.FORWARD)
    decisions[left] = DecisionCategory.LEFT
    decisions[right] = DecisionCategory.RIGHT

    if decisions.shape == (1,):
        return decisions.item()

    return decisions


def decision_category__x(x: np.array, phi: float) -> np.array:
    alpha = get_alpha(x)
    return decision_category__alpha(alpha, phi)


def sample_bout_angles(decision_categories: np.array) -> np.array:
    left = decision_categories == DecisionCategory.LEFT
    right = decision_categories == DecisionCategory.RIGHT
    forward = decision_categories == DecisionCategory.FORWARD

    n_left = np.sum(left)
    n_right = np.sum(right)
    n_forward = np.sum(forward)

    angles = np.zeros(decision_categories.shape)

    angles[left] = constants.LEFT_ANGLE_DISTRIBUTION.rvs(n_left)
    angles[right] = constants.RIGHT_ANGLE_DISTRIBUTION.rvs(n_right)
    angles[forward] = constants.FORWARD_ANGLE_DISTRIBUTION.rvs(n_forward)

    return angles


def boutfield(x: np.array, alpha: np.array, M: np.array, z: np.array) -> np.array:
    x = np.array(x)

    subthreshold = is_subthreshold(x, M, z)
    decisions = decision_category__alpha(alpha, constants.PHI)

    n_samples = x.shape[1]
    field = np.empty((3, n_samples))

    field[0, decisions == DecisionCategory.LEFT] = constants.MAX_BOUT_RATE
    field[1, decisions == DecisionCategory.FORWARD] = constants.MAX_BOUT_RATE
    field[2, decisions == DecisionCategory.RIGHT] = constants.MAX_BOUT_RATE

    field[:, subthreshold] = 0

    return field
