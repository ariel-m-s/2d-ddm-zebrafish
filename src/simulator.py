"""
This module contains the Simulator class, which simulates the two-dimensional
drift-diffusion model. The simulator can be used to simulate an agent's evidence
accumulation process and decision-making behavior.
"""

import numpy as np

import constants
from decision_making import decision_category__x, get_alpha, is_subthreshold
from evidence_accumulation import dx


class Simulator:
    """
    A simulator for the two-dimensional drift-diffusion model. The simulator
    can be used to simulate an agent's evidence accumulation process and
    decision-making behavior.
    """

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
        """
        Initialize the simulator.

        Args:
            refractory_period: The refractory period (in seconds).
            bout_duration: The duration of a bout (in seconds).
            time_step: The time step for the simulation (in seconds).
            gamma: The leak (in 1/s).
            gamma_reset: The leak when a decision is made (in 1/s).
            M: The motor gain (the matrix that defines the ellipse).
            z: The response bias (the center of the ellipse).
            S: The sensory gain (in 1/s).
            xi: The stimulus bias (in 1/s).
            phi: The decision category separation angle (in radians).
            sigma: The standard deviation of the diffusion.
        """
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
        """
        The decision variable as a flattened array. This property is useful
        for plotting or animating the decision variable.

        Returns:
            The decision variable as a flattened array.

        Example:
            >>> simulator = Simulator()
            >>> simulator.x__flat
            array([0., 0.])

            >>> simulator.x += np.array([[1], [2]])
            >>> simulator.x__flat
            array([1., 2.])
        """
        return self.x.flatten()

    @property
    def epsilon(self) -> np.array:
        """
        The diffusion (not scaled by the square root of the time step).

        Returns:
            The diffusion.

        Example:
            >>> simulator = Simulator()
            >>> simulator.epsilon
            array([[0.], [0.]])
        """
        return np.random.multivariate_normal(self.epsilon_mean.flatten(), self.epsilon_cov).reshape(-1, 1)

    @property
    def is_refractory(self) -> bool:
        """
        Check if the agent is in a refractory period. The agent is in a
        refractory period if the time since the last decision is less than
        the refractory period. The refractory period is the time that the
        agent needs to wait before making another decision.

        Returns:
            True if the agent is in a refractory period; False otherwise.

        Example:
            >>> simulator = Simulator()
            >>> simulator.is_refractory
            False

            >>> simulator.time_since_last_decision = 0
            >>> simulator.is_refractory
            True

            >>> simulator.time_since_last_decision = simulator.refractory_period
            >>> simulator.is_refractory
            True

            >>> simulator.time_since_last_decision = simulator.refractory_period + 1
            >>> simulator.is_refractory
            False

            >>> simulator.time_since_last_decision = np.inf
            >>> simulator.is_refractory
            False
        """
        return self.time_since_last_decision <= self.refractory_period

    @property
    def is_subthreshold(self) -> bool:
        """
        Check if the decision variable is inside the decision boundary. The
        decision boundary is defined by the equation of an ellipse, given by
        the motor gain (M) and the response bias (z).

        Returns:
            True if the decision variable is inside the decision boundary;
            False otherwise.
        """
        return is_subthreshold(self.x, self.M, self.z)

    def decide(self) -> np.array:
        """
        Make a decision. The decision is made based on the angle of the
        decision variable (x) and the decision category separation angle (phi).
        The decision categories are LEFT, FORWARD, and RIGHT. The decision is
        made only if the agent is not in a refractory period and the decision
        variable is not inside the decision boundary.

        Returns:
            The decision. The decision is a tuple containing the time (in
            seconds) and the decision category. The decision category is an
            integer that represents the decision category. The decision is
            None if the agent is in a refractory period or the decision
            variable is inside the decision boundary.

        Example:
            >>> simulator = Simulator()
            >>> simulator.decide()
            None
        """
        # If the agent is in a refractory period or the decision variable is
        # inside the decision boundary, do not make a decision. Return None.
        if self.is_refractory or self.is_subthreshold:
            return None

        # Determine the decision category based on the angle of the decision
        # variable (x). The decision categories are LEFT, FORWARD, and RIGHT.
        category = decision_category__x(x=self.x, phi=self.phi)

        # Return the decision. The decision is a tuple containing the time (in
        # seconds) and the decision category. The decision category is an
        # integer that represents the decision category.
        return np.array([self.t, category]).reshape(-1, 1)

    def step(self, c: np.array) -> np.array:
        """
        Simulate one time step. The agent accumulates evidence and makes a
        decision based on the stimulus vector (c).

        Args:
            c: The stimulus vector. The stimulus vector is a two-dimensional
            array that represents the stimulus in Cartesian coordinates. The
            first element is the x-component of the stimulus, and the second
            element is the y-component of the stimulus. The stimulus vector
            is used to calculate the drift, which is the rate of change of
            the decision variable.

        Returns:
            The decision.

        Example:
            >>> simulator = Simulator()
            >>> c = np.array([0, 0])
            >>> simulator.step(c=c)
            None
        """
        # Reshape the stimulus vector (c) to a column vector (2x1) if it is
        # not already in that shape.
        c = c.reshape(-1, 1)

        # alpha = np.rad2deg(get_alpha(self.x))
        # theta = np.rad2deg(get_alpha(c))
        # print(f"alpha: {alpha}")
        # print(f"theta: {theta}")

        # Update the decision variable (x). The decision variable is the
        # agent's internal representation of the accumulated evidence.
        # [Equation 5]
        self.x += dx(
            x=self.x, c=c, gamma=self.gamma, S=self.S, xi=self.xi, epsilon=self.epsilon, time_step=self.time_step
        )

        # Update the time. The time is the cumulative time since the start
        # of the simulation.
        self.t += self.time_step

        # Make a decision.
        decision = self.decide()

        # If a decision is not made, increment the time since the last
        # decision by the time step.
        if decision is None:
            self.time_since_last_decision += self.time_step

        # If a decision is made, append the decision to the decisions array
        # and reset the time since the last decision to zero.
        else:
            self.decisions = np.hstack((self.decisions, decision))
            self.time_since_last_decision = 0

        return decision


if __name__ == "__main__":
    # Example usage of the Simulator class.

    np.random.seed(0)

    simulator = Simulator()

    while simulator.t < 20:
        c = np.array([0, 0])
        simulator.step(c=c)

    print(simulator.decisions)
