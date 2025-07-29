from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """
    `BaseComponent` abstract class. It defines the required
    attributes and methods for any other Canari's component.
    Subclasses must implement all abstract methods including:

    - Initialize the component's name
    - Initialize the component's number of hidden states
    - Initialize hidden states' names
    - Initialize the mean values for hidden states
    - Initialize the covariance matrix for hidden states
    - Initialize the component's transition matrix
    - Initialize the component's observation matrix
    - Initialize the component's process noise matrix

    """

    def __init__(self):
        self._component_name = None
        self._num_states = None
        self._states_name = None
        self._transition_matrix = None
        self._observation_matrix = None
        self._process_noise_matrix = None

        if not hasattr(self, "_mu_states"):
            self._mu_states = None
        if not hasattr(self, "_var_states"):
            self._var_states = None

        self.initialize_component_name()
        self.initialize_num_states()
        self.initialize_states_name()
        self.initialize_transition_matrix()
        self.initialize_observation_matrix()
        self.initialize_process_noise_matrix()
        self.initialize_mu_states()
        self.initialize_var_states()

    @property
    def component_name(self):
        """
        str: Name of the component.
        """
        return self._component_name

    @property
    def num_states(self):
        """
        int: Number of hidden states in the component.
        """
        return self._num_states

    @property
    def states_name(self):
        """
        list[str]: Names of each hidden state in the component.
        """
        return self._states_name

    @property
    def transition_matrix(self):
        """
        np.ndarray: Transition matrix for the component as an 2D array.
        """
        return self._transition_matrix

    @property
    def observation_matrix(self):
        """
        np.ndarray: Observation matrix for the component as an 2D array.
        """
        return self._observation_matrix

    @property
    def process_noise_matrix(self):
        """
        np.ndarray: Process noise covariance matrix as an 2D array.
        """
        return self._process_noise_matrix

    @property
    def mu_states(self):
        """
        np.ndarray: mean values as an 2D array.
        """
        return self._mu_states

    @property
    def var_states(self):
        """
        np.ndarray: covariance matrix as an 2D array.
        """
        return self._var_states

    @abstractmethod
    def initialize_component_name(self):
        """
        Initialize the component's name. str.
        """

    @abstractmethod
    def initialize_num_states(self):
        """
        Initialize the number of hidden states. int.
        """

    @abstractmethod
    def initialize_states_name(self):
        """
        Initialize the names of all hidden states. list[str].
        """

    @abstractmethod
    def initialize_mu_states(self):
        """
        Initialize the mean of the hidden states. Output an 2D array.
        """

    @abstractmethod
    def initialize_var_states(self):
        """
        Initialize the covariance matrix of the hidden states. Output an 2D array.
        """

    @abstractmethod
    def initialize_transition_matrix(self):
        """
        Initialize the transition matrix. Output an 2D array.
        """

    @abstractmethod
    def initialize_observation_matrix(self):
        """
        Initialize the observation matrix. Output an 2D array.
        """

    @abstractmethod
    def initialize_process_noise_matrix(self):
        """
        Initialize the process noise covariance matrix. Output an 2D array.
        """
