from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent
import canari.common as common


class KernalRegression(BaseComponent):
    """
    `KernalRegression` class, inheriting from Canari's `BaseComponent`.
    It models a recurrent behavior with a fixed period using Kernel Regression.

    Args:
        period (float): Length of one full cycle of the periodic component
          (number of time steps).
        std_error (Optional[float]): Standard deviation of the process noise. Defaults to 0.0.
        std_error_cp (Optional[float]): Standard deviation of the process noise for control points. 
                                        Defaults to 0.0.
        mu_states (Optional[list[float]]): Initial mean of the hidden state. Defaults:
            initialized to zeros.
        var_states (Optional[list[float]]): Initial variance of the hidden state. Defaults:
            initialized to zeros.

    Examples:
        >>> from canari.component import Periodic
        >>> # With known parameters and default mu_states and var_states
        >>> periodic = Periodic(std_error=0.1, period=52)
        >>> # With known mu_states and var_states
        >>> periodic = Periodic(mu_states=[0., 0.], var_states=[1., 1.], std_error=0.1, period=52)
        >>> periodic.states_name
        ['periodic 1', 'periodic 2']
        >>> periodic.mu_states
        >>> periodic.var_states
        >>> periodic.transition_matrix
        >>> periodic.observation_matrix
        >>> periodic.process_noise_matrix
    """

    def __init__(
        self,
        period: float,
        num_control_point: Optional[int] = 10,
        kernel_length: Optional[float] = 0.5,
        std_error: Optional[float] = 0.0,
        std_error_cp: Optional[float] = 0.0,
        mu_states: Optional[list[float]] = None,
        var_states: Optional[list[float]] = None,
    ):
        self.std_error = std_error
        self.std_error_cp = std_error_cp
        self.period = period
        self._mu_states = mu_states
        self._var_states = var_states
        self.num_control_point = num_control_point
        self.kernel_length = kernel_length
        self.time_control_point = np.linspace(0, period, num_control_point)
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "kernel regression"

    def initialize_num_states(self):
        self._num_states = self.num_control_point + 1

    def initialize_states_name(self):
        name = ["kernel regression"]
        for i in range(self.num_control_point):
            name.append(f"kernel regression cp_{i}")
        self._states_name = name

    def initialize_transition_matrix(self):
        """
        .
        """
        self._transition_matrix = common.create_block_diag(
            np.array([[0]]), np.eye(self.num_control_point)
            )
        self._transition_matrix[0,1:] = 1

    def initialize_observation_matrix(self):
        self._observation_matrix = np.concatenate(
                (np.array([[1]]), np.zeros((1, self.num_control_point))), axis=1
            )

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = common.create_block_diag(
            self.std_error**2 * np.array([[1]]), self.std_error_cp**2 * np.eye(self.num_control_point)
            )

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the kernel regression component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the kernel regression component."
            )
