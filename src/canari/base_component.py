"""
BaseComponent for Canari

This module defines the `BaseComponent` abstract class. It defines the required
structure for any Canari's component.

Subclasses must implement all abstract methods.
"""

from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """
    Abstract base class for all Canari's components.

    This class defines the interface and shared attributes for all components.

    Attributes:
        _component_name (str): Unique identifier string for the component.
        _num_states (int): Number of hidden states in the component.
        _states_name (list of str): Names of each hidden state.
        _transition_matrix (np.ndarray): Transition matrix for the component.
        _observation_matrix (np.ndarray): Observation matrix for the component.
        _process_noise_matrix (np.ndarray): Process noise covariance matrix.
        _mu_states (np.ndarray): mean values for hidden states.
        _var_states (np.ndarray): covariance matrix for hidden states.

    Properties:
        component_name (str): Name of the component.
        num_states (int): Number of hidden states.
        states_name (list[str]): Names of the hidden states.
        transition_matrix (np.ndarray): Transition matrix.
        observation_matrix (np.ndarray): Observation matrix.
        process_noise_matrix (np.ndarray): Process noise matrix.
        mu_states (np.ndarray): mean vector for hidden states
        var_states (np.ndarray): covariance matrix for hidden states


    Examples:
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
        list[str]: Names of each hidden state.
        """
        return self._states_name

    @property
    def transition_matrix(self):
        """
        np.ndarray: Transition matrix for the component.
        """
        return self._transition_matrix

    @property
    def observation_matrix(self):
        """
        np.ndarray: Observation matrix for the component.
        """
        return self._observation_matrix

    @property
    def process_noise_matrix(self):
        """
        np.ndarray: Process noise covariance matrix.
        """
        return self._process_noise_matrix

    @property
    def mu_states(self):
        """
        np.ndarray: mean values.
        """
        return self._mu_states

    @property
    def var_states(self):
        """
        np.ndarray: covariance matrix.
        """
        return self._var_states

    @abstractmethod
    def initialize_component_name(self):
        """
        Abstract method to initialize the component's name.
        """

    @abstractmethod
    def initialize_num_states(self):
        """
        Abstract method to initialize the number of hidden states.
        """

    @abstractmethod
    def initialize_states_name(self):
        """
        Abstract method to initialize the names of all hidden states.
        Must provide `self._states_name` with a list of strings.
        """

    @abstractmethod
    def initialize_mu_states(self):
        """
        Abstract method to initialize the mean of the hidden states.
        Must define or validate the initial `mu_states` as a 2D NumPy ndarray.
        """

    @abstractmethod
    def initialize_var_states(self):
        """
        Abstract method to initialize the variance of the hidden states.
        Must define or validate the initial `var_states` as a 2D NumPy ndarray.
        """

    @abstractmethod
    def initialize_transition_matrix(self):
        """
        Abstract method to initialize the transition matrix.
        Must define `self._transition_matrix` as a 2D NumPy ndarray.
        """

    @abstractmethod
    def initialize_observation_matrix(self):
        """
        Abstract method to initialize the observation matrix.
        Must define `self._observation_matrix` as a 2D NumPy ndarray.
        """

    @abstractmethod
    def initialize_process_noise_matrix(self):
        """
        Abstract method to initialize the process noise covariance matrix.
        Must define `self._process_noise_matrix` as a 2D NumPy ndarray.
        """

    @property
    def mu_states(self):
        """
        np.ndarray: mean vector for hidden states
        """
        return self._mu_states

    @property
    def var_states(self):
        """
        np.ndarray: covariance matrix for hidden states
        """
        return self._var_states
