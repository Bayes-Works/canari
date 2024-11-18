from typing import Optional
import numpy as np
from src.base_component import BaseComponent
import pytagi
from pytagi.nn import LSTM, Linear
from pytagi.nn import Sequential


class LstmNetwork(BaseComponent):
    """
    LSTM component
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        num_layer: Optional[int] = 1,
        num_hidden_unit: Optional[int] = 50,
        look_back_len: int = 1,
        num_features: int = 1,
        input_features: Optional[str] = None,
        num_output: Optional[int] = 1,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
        device: Optional[str] = "cpu",
        num_thread: Optional[int] = 1,
    ):
        self.std_error = std_error
        self.num_layer = num_layer
        self.num_hidden_unit = num_hidden_unit
        self.look_back_len = look_back_len
        self.num_features = num_features
        self.input_features = input_features
        self.num_output = num_output
        self.mu_states = mu_states
        self.var_states = var_states
        self.device = device
        self.num_thread = num_thread
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "lstm"

    def initialize_num_states(self):
        self._num_states = 1

    def initialize_states_name(self):
        self._states_name = ["lstm"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[0]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.array([[self.std_error**2]])

    def initialize_mu_states(self):
        if self.mu_states is None:
            self.mu_states = np.zeros((self.num_states, 1))
        elif len(self.mu_states) == self.num_states:
            self.mu_states = np.atleast_2d(self.mu_states).T
        else:
            raise ValueError(f"Incorrect mu_states dimension for the lstm component.")

    def initialize_var_states(self):
        if self.var_states is None:
            self.var_states = np.zeros((self.num_states, 1))
        elif len(self.var_states) == self.num_states:
            self.var_states = np.atleast_2d(self.var_states).T
        else:
            raise ValueError(f"Incorrect var_states dimension for the lstm component.")

    def initialize_lstm_network(self) -> Sequential:
        pytagi.manual_seed(2)
        layers = []
        if isinstance(self.num_hidden_unit, int):
            self.num_hidden_unit = [self.num_hidden_unit] * self.num_layer
        # First layer
        layers.append(
            LSTM(self.num_features + self.look_back_len - 1, self.num_hidden_unit[0], 1)
        )
        for i in range(1, self.num_layer):
            layers.append(LSTM(self.num_hidden_unit[i], self.num_hidden_unit[i], 1))
        # Last layer
        layers.append(Linear(self.num_hidden_unit[-1], self.num_output, 1))
        # Initialize lstm network
        lstm_network = Sequential(*layers)
        if self.device == "cpu":
            lstm_network.set_threads(self.num_thread)
        elif self.device == "cuda":
            lstm_network.to_device("cuda")

        return lstm_network
