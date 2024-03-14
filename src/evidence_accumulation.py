import numpy as np


def drift(c: np.array, S: np.array, xi: np.array) -> np.array:
    return np.matmul(S, c) + xi


def dx(
    x: np.array, c: np.array, S: np.array, xi: np.array, gamma: float, time_step: float, epsilon: np.array
) -> np.array:
    d = drift(c, S, xi)

    increment = (d - gamma * x) * time_step
    noise = epsilon * np.sqrt(time_step)

    return increment + noise


def x_cov(sigma: np.array, gamma: float, delta_t: float = np.inf) -> np.array:
    asymptote = sigma**2 / (2 * gamma)

    if delta_t == np.inf:
        return np.zeros_like(asymptote)

    exponential = np.exp(-2 * gamma * delta_t)
    return asymptote * np.sqrt(1 - exponential)


def x_mean(drift: np.array, gamma: float, delta_t: float = np.inf) -> np.array:
    asymptote = drift / gamma

    if delta_t == np.inf:
        return asymptote

    exponential = np.exp(-gamma * delta_t)
    return asymptote * (1 - exponential)


def get_c(coherence: float, theta__degrees: float = None) -> np.array:
    if coherence != 0:
        theta__radians = np.deg2rad(theta__degrees)
        c = coherence * np.array([np.sin(theta__radians), np.cos(theta__radians)])
    else:
        c = np.zeros((2,))

    return c.reshape(-1, 1)
