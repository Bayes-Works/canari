from typing import Optional, Callable, Tuple
import numpy as np
from canari.component.base_component import BaseComponent


PredictFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


class Auxiliary(BaseComponent):
    """
    `Auxiliary` class, inheriting from Canari's `BaseComponent`.
    Wraps an external one-step-ahead predictor (e.g. DeepAR, Chronos2)
    so it can plug into the canari state-space model the same way
    :class:`~canari.component.lstm_component.LstmNetwork` does.

    The user provides a ``predict_fn`` with signature
    ``(mu_input, var_input) -> (mu_pred, var_pred)``. The input window
    is the same one used by :class:`LstmNetwork`: concatenation of the
    past ``look_back_len`` component outputs and ``num_features`` input
    covariates, built via :func:`~canari.common.prepare_lstm_input`.
    ``mu_pred`` and ``var_pred`` must be 1-D arrays of size 1
    (one-step-ahead mean and variance).

    Args:
        predict_fn (Callable): User-supplied one-step-ahead predictor.
        std_error (Optional[float]): Process noise std in the SSM. Defaults to 0.0.
        look_back_len (Optional[int]): Size of the rolling output history. Defaults to 1.
        num_features (Optional[int]): Number of input covariate features. Defaults to 1.
        mu_states (Optional[list[float]]): Initial mean of the hidden state.
        var_states (Optional[list[float]]): Initial variance of the hidden state.

    Examples:
        >>> from canari.component import Auxiliary
        >>> def predict_fn(mu_in, var_in):
        ...     return np.array([0.0]), np.array([1.0])
        >>> aux = Auxiliary(predict_fn=predict_fn, std_error=0.1, look_back_len=5)
    """

    def __init__(
        self,
        predict_fn: PredictFn,
        std_error: Optional[float] = 0.0,
        look_back_len: Optional[int] = 1,
        num_features: Optional[int] = 1,
        mu_states: Optional[list] = None,
        var_states: Optional[list] = None,
    ):
        self.predict_fn = predict_fn
        self.std_error = std_error
        self.look_back_len = look_back_len
        self.num_features = num_features
        self._mu_states = mu_states
        self._var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "auxiliary"

    def initialize_num_states(self):
        self._num_states = 1

    def initialize_states_name(self):
        self._states_name = ["auxiliary"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[0]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.array([[self.std_error**2]])

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError("Incorrect mu_states dimension for the auxiliary component.")

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError("Incorrect var_states dimension for the auxiliary component.")
