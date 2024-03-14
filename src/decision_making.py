"""
Functions for the decision-making part of the model.
"""

from enum import Enum

import numpy as np

import constants


class DecisionCategory(Enum):
    """
    Enum class for the decision categories.

    Attributes:
        LEFT: The decision category for a left bout.
        FORWARD: The decision category for a forward bout.
        RIGHT: The decision category for a right bout.
    """

    LEFT = -1
    FORWARD = 0
    RIGHT = +1


def get_alpha(x: np.array) -> np.array:
    """
    [Equation 7]

    Calculate the angle of the decision variable (in radians).

    Args:
        x: The decision variable.

    Returns:
        The angle of the decision variable (in radians).
    """
    return np.arctan2(x[1], x[0])


def is_subthreshold(x: np.array, M: np.array, z: np.array) -> np.array:
    """
    [Equation 19]

    Check if the decision variable is inside the decision boundary. The
    decision boundary is defined by the equation of an ellipse, given by
    the motor gain (M) and the response bias (z).

    Args:
        x: The decision variable.
        M: The motor gain (the matrix that defines the ellipse).
        z: The response bias (the center of the ellipse).

    Returns:
        A boolean array indicating if the decision variable is inside the
        decision boundary. True if the decision variable is inside the
        decision boundary, False otherwise. The shape of the array is the
        same as the shape of the decision variable.
    """
    ellipse = np.matmul(M, x) + z
    return np.linalg.norm(ellipse, axis=0) < 1


def decision_category__alpha(alpha: np.array, phi: float) -> np.array:
    """
    [Equation 21]

    Determine the decision category based on the angle of the decision
    variable (alpha).

    The decision categories are defined by the angle at which the decision
    categories are separated (phi). The decision categories are LEFT,
    FORWARD, and RIGHT.

    Args:
        alpha: The angle of the decision variable (in radians).
        phi: The angle at which the decision categories are separated (in radians).

    Returns:
        The decision category for each decision variable. The decision
        categories are LEFT, FORWARD, and RIGHT. The shape of the array
        is the same as the shape of the decision variable.
    """
    # The decision categories are separated by the angle phi.
    left = alpha < -phi
    right = alpha > phi

    # The possible decision categories are LEFT, FORWARD, and RIGHT.
    decisions = np.full(alpha.shape, DecisionCategory.FORWARD)
    decisions[left] = DecisionCategory.LEFT
    decisions[right] = DecisionCategory.RIGHT

    # If there is only one decision,
    # return the decision as a scalar, not an array.
    if decisions.shape == (1,):
        return decisions.item()

    # If there are multiple decisions, return the decisions as an array.
    return decisions


def decision_category__x(x: np.array, phi: float) -> np.array:
    """
    [Equation 21]

    Determine the decision category based on the decision variable (x).

    Args:
        x: The decision variable.
        phi: The angle at which the decision categories are separated (in radians).

    Returns:
        The decision category for each decision variable.
    """
    # Calculate the angle of the decision variable.
    alpha = get_alpha(x)

    # Determine the decision category based on the angle of the decision variable.
    return decision_category__alpha(alpha, phi)


def sample_bout_angles(decision_categories: np.array) -> np.array:
    """
    Sample the angles for the bouts based on the decision categories.

    Args:
        decision_categories: The decision categories for the bouts.

    Returns:
        The angles for the bouts.
    """
    # Filter by decision category.
    left = decision_categories == DecisionCategory.LEFT
    right = decision_categories == DecisionCategory.RIGHT
    forward = decision_categories == DecisionCategory.FORWARD

    # Determine the number of bouts for each decision category.
    n_left = np.sum(left)
    n_right = np.sum(right)
    n_forward = np.sum(forward)

    # Initialize the array for the angles.
    angles = np.zeros(decision_categories.shape)

    # Sample the angles for the bouts based on the decision categories.
    angles[left] = constants.LEFT_ANGLE_DISTRIBUTION.rvs(n_left)
    angles[right] = constants.RIGHT_ANGLE_DISTRIBUTION.rvs(n_right)
    angles[forward] = constants.FORWARD_ANGLE_DISTRIBUTION.rvs(n_forward)

    return angles


def boutfield(x: np.array, alpha: np.array, M: np.array, z: np.array) -> np.array:
    """
    Calculate the bout type and rate based on the decision variable.

    Args:
        x: The decision variable.
        alpha: The angle of the decision variable (in radians).
        M: The motor gain (the matrix that defines the ellipse).
        z: The response bias (the center of the ellipse).

    Returns:
        The bout type and rate based on the decision variable. The shape
        of the array is (3, n_samples), where n_samples is the number of
        samples of the decision variable. The first row is the bout rate
        for a left bout, the second row is the bout rate for a forward
        bout, and the third row is the bout rate for a right bout.
    """
    # Determine if the decision variable is inside the decision boundary.
    subthreshold = is_subthreshold(x, M, z)

    # Determine the decision category based on the angle of the decision variable.
    decisions = decision_category__alpha(alpha, constants.PHI)

    # Initialize the array for the bout field.
    n_samples = x.shape[1]
    field = np.empty((3, n_samples))

    # Set the bout rate based on the decision category.
    field[0, decisions == DecisionCategory.LEFT] = constants.MAX_BOUT_RATE
    field[1, decisions == DecisionCategory.FORWARD] = constants.MAX_BOUT_RATE
    field[2, decisions == DecisionCategory.RIGHT] = constants.MAX_BOUT_RATE

    # Set the bout rate to zero if the decision variable is inside the decision boundary.
    field[:, subthreshold] = 0

    return field
