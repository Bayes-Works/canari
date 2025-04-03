from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """
    Base component
    """

    def __init__(self):
        self._component_name = None
        self._num_states = None
        self._states_name = None
        self._transition_matrix = None
        self._observation_matrix = None
        self._process_noise_matrix = None

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
        """Getter for the component's name."""
        return self._component_name

    @property
    def num_states(self):
        """Getter for the number of hidden states."""
        return self._num_states

    @property
    def states_name(self):
        """Getter for the hidden states's name."""
        return self._states_name

    @property
    def transition_matrix(self):
        """Getter for the transition matrix."""
        return self._transition_matrix

    @property
    def observation_matrix(self):
        """Getter for the observation matrix."""
        return self._observation_matrix

    @property
    def process_noise_matrix(self):
        """Getter for the process noise matrix."""
        return self._process_noise_matrix

    @abstractmethod
    def initialize_component_name(self):
        """Initialize the component's name."""
        pass

    @abstractmethod
    def initialize_num_states(self):
        """Initialize the number of hidden states."""
        pass

    @abstractmethod
    def initialize_states_name(self):
        """Initialize the hidden states' names."""
        pass

    @abstractmethod
    def initialize_transition_matrix(self):
        """Initialize the transition matrix."""
        pass

    @abstractmethod
    def initialize_observation_matrix(self):
        """Initialize the observation matrix."""
        pass

    @abstractmethod
    def initialize_process_noise_matrix(self):
        """Initialize the process noise matrix."""
        pass

    @abstractmethod
    def initialize_mu_states(self):
        """Initialize the means for the hidden states."""
        pass

    @abstractmethod
    def initialize_var_states(self):
        """Initialize the variances for the hidden states."""
        pass
