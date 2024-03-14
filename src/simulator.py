import numpy as np

import constants
from decision_making import decision_category__x, is_subthreshold
from evidence_accumulation import dx


class Simulator:
    def __init__(
        self,
        refractory_period: float = constants.REFRACTORY_PERIOD,
        bout_duration: float = constants.BOUT_DURATION,
        time_step: float = constants.TIME_STEP,
        gamma: float = constants.GAMMA,
        gamma_reset: float = constants.GAMMA_RESET,
        M: np.array = constants.M,
        z: np.array = constants.Z,
        S: np.array = constants.S,
        xi: float = constants.XI,
        phi: float = constants.PHI,
        sigma: float = constants.SIGMA,
    ):
        self.refractory_period = refractory_period
        self.bout_duration = bout_duration

        self.time_step = time_step

        self.gamma = gamma
        self.gamma_reset = gamma_reset

        self.M = M
        self.z = z

        self.S = S
        self.xi = xi

        self.phi = phi

        self.epsilon_mean = np.zeros((2,)).reshape(-1, 1)
        self.epsilon_cov = constants.SIGMA**2

        self.x = np.zeros((2,)).reshape(-1, 1)
        self.t = 0

        self.decisions = np.ndarray((2, 0))
        self.time_since_last_decision = np.inf

    @property
    def x__flat(self) -> np.array:
        return self.x.flatten()

    @property
    def epsilon(self) -> np.array:
        return np.random.multivariate_normal(self.epsilon_mean.flatten(), self.epsilon_cov).reshape(-1, 1)

    @property
    def is_refractory(self) -> bool:
        return self.time_since_last_decision <= self.refractory_period

    @property
    def is_subthreshold(self) -> bool:
        return is_subthreshold(self.x, self.M, self.z)

    def decide(self) -> np.array:
        if self.is_refractory or self.is_subthreshold:
            return None

        category = decision_category__x(x=self.x, phi=self.phi)
        return np.array([self.t, category]).reshape(-1, 1)

    def step(self, c: np.array) -> np.array:
        c = c.reshape(-1, 1)

        self.x += dx(
            x=self.x, c=c, S=self.S, xi=self.xi, gamma=self.gamma, time_step=self.time_step, epsilon=self.epsilon
        )

        self.t += self.time_step

        decision = self.decide()

        if decision is None:
            self.time_since_last_decision += self.time_step

        else:
            self.decisions = np.hstack((self.decisions, decision))
            self.time_since_last_decision = 0

        return decision


if __name__ == "__main__":
    np.random.seed(0)

    simulator = Simulator()

    while simulator.t < 20:
        c = np.array([0, 0])
        simulator.step(c=c)

    print(simulator.decisions)
