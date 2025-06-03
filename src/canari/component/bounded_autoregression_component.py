from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class BoundedAutoregression(BaseComponent):
    """
    `BoundedAutoregression` class, inheriting from Canari's `BaseComponent`.
    It models residuals following a univariate AR(1) process with optional constrains defined by
    a coefficient (gamma), which scales the standard deviation of the stationary AR.

    Parameters:
        std_error ([float]): Known standard deviation of the process noise.
        phi ([float]): Known autoregressive coefficient.
        gamma (Optional[float]): Coefficient to scale the standard deviation of the stationary AR.
                                    If none, no constraint is applied.
        mu_states (Optional[list[float]]): Initial mean of the hidden state. Defaults:
            initialized to zeros.
        var_states (Optional[list[float]]): Initial variance of the hidden state. Defaults:
            initialized to zeros.

    Behavior:
        - Adds 1 extra state, i.e. `X^{BAR}`, if gamma is provided. Otherwise no constraint is applied.

    References:
        Xin, Z.  and Goulet, J.-A. (2024). `Enhancing structural anomaly detection using a bounded autoregressive component
        <https://www.sciencedirect.com/science/article/pii/S0888327024001778>`_.
        Mechanical Systems and Signal Processing. Volume 212, pp.111279.

    Examples:
        >>> from canari.component import BoundedAutoregression
        >>> # with gamma
        >>> bar = BoundedAutoregression(std_error=1, phi=0.75, gamma=0.5)
        >>> # without gamma
        >>> bar_1 = BoundedAutoregression(std_error=1, phi=0.75)
    """

    def __init__(
        self,
        std_error: float,
        phi: float,
        gamma: Optional[float] = None,
        mu_states: Optional[list[float]] = None,
        var_states: Optional[list[float]] = None,
    ):
        self.std_error = std_error
        self.phi = phi
        self.gamma = gamma
        self._mu_states = mu_states
        self._var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "bounded autoregression"

    def initialize_num_states(self):
        self._num_states = 2
        if self.gamma is None:
            self._num_states = 1

    def initialize_states_name(self):
        self._states_name = ["autoregression", "bounded autoregression"]
        if self.gamma is None:
            self._states_name = ["autoregression"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[0, self.phi], [0, 0]])
        if self.gamma is None:
            self._transition_matrix = np.array([[self.phi]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1, 0]])
        if self.gamma is None:
            self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.array([[self.std_error**2, 0], [0, 0]])
        if self.gamma is None:
            self._process_noise_matrix = np.array([[self.std_error**2]])

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the autoregression component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the autoregression component."
            )
