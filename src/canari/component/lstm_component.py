from typing import Optional
import numpy as np
import pytagi
from pytagi.nn import Sequential, LSTM, Linear, SLSTM, SLinear
from canari.component.base_component import BaseComponent


class LstmNetwork(BaseComponent):
    """
    `LstmNetwork` class, inheriting from Canari's `BaseComponent`.
    This component configures a Bayesian LSTM neural network from `pyTAGI` library, and
    it can be used in the same way as a traditional LSTM from PyTorch, e.g. to model
    recurrent patterns such as seasonal and periodic patterns.

    Args:
        std_error (Optional[float]): Standard deviation of the process noise. Defaults to 0.0.
        num_layer (Optional[int]): Number of LSTM layers. Defaults to 1.
        num_hidden_unit (Optional[int] or list[int]): Number of hidden units per LSTM
            layer. If an integer is provided, it is used for all layers. Defaults to 50.
        look_back_len (Optional[int]): Number of past LSTM's outputs used as input features.
            Defaults to 1.
        num_features (Optional[int]): Number of input features. Defaults to 1.
        num_output (Optional[int]): Number of output features predicted by the network.
                                    Defaults to 1.
        device (Optional[str]): Device used for computation, either "cpu" or "cuda".
                                Defaults to "cpu".
        num_thread (Optional[int]): Number of CPU threads for computation. Defaults to 1.
        manual_seed (Optional[int]): Initial seed for reproducing random number generation,
                                    i.e. intializing LSTM's weights and biases.
                                    Defaults to None (random initialization).
        gain_weight (Optional[int]): Scaling factor for weight initialization. Defaults to 1.
        gain_bias (Optional[int]): Scaling factor for bias initialization. Defaults to 1.
        load_lstm_net (Optional[str]): Path to a saved LSTM network file containing pretrained
                                        LSTM's weights and biases. Defaults to None.
        mu_states (Optional[list[float]]): Initial mean of the hidden state. Defaults:
            initialized to zeros.
        var_states (Optional[list[float]]): Initial variance of the hidden state. Defaults:
            initialized to zeros.
        smoother (Optional[bool]): if True using Smooth LSTM (SLSTM), if False LSTM is used.
                                    Defaults to True.
        infer_len (Optional[int]): Length of the window before the training set to be inferred
                                    when SLSTM is used. Defaults to 1.
        model_noise (Optional[bool]): if True, using the AGVI method to model heteroscedastic noise
                                    (See references). Defaults to False.

    References:
        Vuong, V.D., Nguyen, L.H. and Goulet, J.-A. (2025). `Coupling LSTM neural networks and
        state-space models through analytically tractable inference
        <https://www.sciencedirect.com/science/article/pii/S0169207024000335>`_.
        International Journal of Forecasting. Volume 41, Issue 1, Pages 128-140.

        Deka, B., Nguyen, L.H. and Goulet, J.-A. (2024). `Analytically tractable heteroscedastic
        uncertainty quantification in Bayesian neural networks for regression tasks
        <https://www.sciencedirect.com/science/article/pii/S0925231223013061>`_.
        Neurocomputing. Volume 572, pp.127183.

    Examples:
        >>> from canari.component import LstmNetwork
        >>> lstm = LstmNetwork(
        ...     std_error=0.1,
        ...     num_layer=2,
        ...     num_hidden_unit=[32, 16],
        ...     look_back_len=5,
        ...     num_features=3,
        ...     device="cpu",
        ...     manual_seed=1,
        ... )
        >>> lstm.states_name
        ['lstm']
        >>> lstm.mu_states
        array([[0.]])
        >>> lstm.var_states
        array([[0.]])
        >>> lstm.transition_matrix
        array([[0.]])
        >>> lstm.observation_matrix
        array([[1.]])
        >>> lstm.process_noise_matrix
        array([[0.01]])
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        num_layer: Optional[int] = 1,
        num_hidden_unit: Optional[int] = 50,
        look_back_len: Optional[int] = 1,
        infer_len: Optional[int] = 1,
        num_features: Optional[int] = 1,
        num_output: Optional[int] = 1,
        device: Optional[str] = "cpu",
        num_thread: Optional[int] = 1,
        manual_seed: Optional[int] = None,
        gain_weight: Optional[int] = 1,
        gain_bias: Optional[int] = 1,
        load_lstm_net: Optional[str] = None,
        mu_states: Optional[list[float]] = None,
        var_states: Optional[list[float]] = None,
        smoother: Optional[bool] = True,
        model_noise: Optional[bool] = False,
    ):
        self.std_error = std_error
        self.num_layer = num_layer
        self.num_hidden_unit = num_hidden_unit
        self.look_back_len = look_back_len
        self.infer_len = infer_len
        self.num_features = num_features
        self.device = device
        self.num_thread = num_thread
        self.manual_seed = manual_seed
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.load_lstm_net = load_lstm_net
        self._mu_states = mu_states
        self._var_states = var_states
        self.smoother = smoother
        self.model_noise = model_noise
        self.num_output = 2 * num_output if self.model_noise else num_output
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "lstm"

    def initialize_num_states(self):
        self._num_states = 2 if self.model_noise else 1

    def initialize_states_name(self):
        if self.model_noise:
            self._states_name = ["lstm", "heteroscedastic noise"]
        else:
            self._states_name = ["lstm"]

    def initialize_transition_matrix(self):
        if self.model_noise:
            self._transition_matrix = np.array([[0, 0], [0, 0]])
        else:
            self._transition_matrix = np.array([[0]])

    def initialize_observation_matrix(self):
        if self.model_noise:
            self._observation_matrix = np.array([[1, 1]])
        else:
            self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        if self.model_noise:
            self._process_noise_matrix = np.array([[self.std_error**2, 0], [0, 0]])
        else:
            self._process_noise_matrix = np.array([[self.std_error**2]])

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self.num_states, 1))
        elif len(self._mu_states) == self.num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(f"Incorrect mu_states dimension for the lstm component.")

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self.num_states, 1))
        elif len(self._var_states) == self.num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(f"Incorrect var_states dimension for the lstm component.")

    def initialize_lstm_network(self) -> Sequential:
        """
        Builds and returns the LSTM network as a :class:`pytagi.Sequential` instance.

        The network consists of:

        - One or multiple LSTM layers, each with specified hidden units.
        - A final Linear layer mapping the LSTM's output to the desired output size.

        The first LSTM layer input size is determined by `num_features + look_back_len - 1`.

        Returns:
            Sequential: a :class:`pytagi.Sequential` instance representing the LSTM network.
        """

        if self.manual_seed:
            pytagi.manual_seed(self.manual_seed)

        layers = []
        if isinstance(self.num_hidden_unit, int):
            self.num_hidden_unit = [self.num_hidden_unit] * self.num_layer
        if self.smoother:
            layers.append(
                SLSTM(
                    self.num_features + self.look_back_len - 1,
                    self.num_hidden_unit[0],
                    1,
                    gain_weight=self.gain_weight,
                    gain_bias=self.gain_bias,
                )
            )
            for i in range(1, self.num_layer):
                layers.append(
                    SLSTM(self.num_hidden_unit[i], self.num_hidden_unit[i], 1)
                )
            # Last layer
            layers.append(
                SLinear(
                    self.num_hidden_unit[-1],
                    self.num_output,
                    1,
                    gain_weight=self.gain_weight,
                    gain_bias=self.gain_bias,
                )
            )
        else:
            layers.append(
                LSTM(
                    self.num_features + self.look_back_len - 1,
                    self.num_hidden_unit[0],
                    1,
                    gain_weight=self.gain_weight,
                    gain_bias=self.gain_bias,
                )
            )
            for i in range(1, self.num_layer):
                layers.append(LSTM(self.num_hidden_unit[i], self.num_hidden_unit[i], 1))
            # Last layer
            layers.append(
                Linear(
                    self.num_hidden_unit[-1],
                    self.num_output,
                    1,
                    gain_weight=self.gain_weight,
                    gain_bias=self.gain_bias,
                )
            )
        # Initialize lstm network
        lstm_network = Sequential(*layers)
        lstm_network.lstm_look_back_len = self.look_back_len
        lstm_network.lstm_infer_len = self.infer_len
        lstm_network.model_noise = self.model_noise
        lstm_network.num_samples = 1  # dummy intialization until otherwise specified
        if self.device == "cpu":
            lstm_network.set_threads(self.num_thread)
        elif self.device == "cuda":
            # TODO: remove this warning when SLSTM supports GPU
            if self.smoother:
                print(
                    "Warning: pytagi SLSTM does not support GPU yet. Resetting to CPU."
                )
                lstm_network.set_threads(self.num_thread)
            else:
                lstm_network.to_device("cuda")

        if self.smoother:
            lstm_network.smooth = True
        else:
            lstm_network.smooth = False

        if self.load_lstm_net:
            lstm_network.load(filename=self.load_lstm_net)

        return lstm_network
