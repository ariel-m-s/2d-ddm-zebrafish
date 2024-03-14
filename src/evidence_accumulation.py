import numpy as np


def drift(c: np.array, S: np.array, xi: np.array) -> np.array:
    """
    [Equation 17]

    Calculate the drift. The drift is the product of the coherence (c) and
    the sensory gain (S) plus the stimulus bias (xi).

    Args:
        c: The coherence (stimulus).
        S: The sensory gain.
        xi: The stimulus bias.

    Returns:
        The drift.
    """
    return np.matmul(S, c) + xi


def dx(
    x: np.array, gamma: float, c: np.array, S: np.array, xi: np.array, epsilon: np.array, time_step: float
) -> np.array:
    """
    [Equation 5]

    Calculate the change in the decision variable (dx) at each time step.

    Args:
        x: The decision variable.
        gamma: The leak.
        c: The coherence (stimulus).
        S: The sensory gain.
        xi: The stimulus bias.
        epsilon: The diffusion.
        time_step: The time step.

    Returns:
        The change in the decision variable.
    """
    # Calculate the drift.
    d = drift(c, S, xi)

    # Calculate the increment, which is proportional to the time step.
    increment = (d - gamma * x) * time_step

    # Calculate the diffusion,
    # which is proportional to the square root of the time step.
    noise = epsilon * np.sqrt(time_step)

    return increment + noise


def x_cov(sigma: np.array, gamma: float, delta_t: float = np.inf) -> np.array:
    """
    [Equation 8]

    Calculate the covariance of the decision variable.

    Args:
        sigma: The standard deviation of the diffusion.
        gamma: The leak.
        delta_t: The time that has elapsed since the last steady-state.

    Returns:
        The covariance of the decision variable.
    """
    # Calculate the asymptote (steady-state) of the covariance.
    asymptote = sigma**2 / (2 * gamma)

    # If the time that has elapsed since the last steady-state is infinite,
    # return the asymptoate. The decision variable has reached a steady-state.
    if delta_t == np.inf:
        return asymptote

    # If the time that has elapsed since the last steady-state is not infinite,
    # calculate the value of the decision variable at a given time. The decision
    # variable has not reached a steady-state.
    exponential = np.exp(-2 * gamma * delta_t)
    return asymptote * np.sqrt(1 - exponential)


def x_mean(drift: np.array, gamma: float, delta_t: float = np.inf) -> np.array:
    """
    [Equation 9]

    Calculate the expected value of the decision variable.

    Args:
        drift: The drift.
        gamma: The leak.
        delta_t: The time that has elapsed since the last steady-state.

    Returns:
        The expected value of the decision variable.

    """
    # Calculate the asymptote (steady-state) of the expected value.
    asymptote = drift / gamma

    # If the time that has elapsed since the last steady-state is infinite,
    # return the asymptote. The decision variable has reached a steady-state.
    if delta_t == np.inf:
        return asymptote

    # If the time that has elapsed since the last steady-state is not infinite,
    # calculate the value of the decision variable at a given time. The decision
    # variable has not reached a steady-state.
    exponential = np.exp(-gamma * delta_t)
    return asymptote * (1 - exponential)


def get_c(coherence: float, theta__degrees: float = None) -> np.array:
    """
    [Equations 13 and 14]

    Get the stimulus vector (c) from the stimulus strength (coherence) and
    the direction (theta).

    Args:
        coherence: The stimulus strength (ranging from 0 to 1, not 0% to 100%).
        theta__degrees: The direction of the stimulus (in degrees).

    Returns:
        The stimulus vector (c).
    """
    if coherence != 0:
        theta__radians = np.deg2rad(theta__degrees)
        c = coherence * np.array([np.sin(theta__radians), np.cos(theta__radians)])
    else:
        c = np.zeros((2,))

    return c.reshape(-1, 1)
