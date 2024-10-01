import numpy as np
from typing import Optional
from typing import List

# base component class
class baseComponent:
    """
    Base class for state-space components.

    Attributes:
        mu (Optional[np.ndarray]): Mean of the hidden states.
        var (Optional[np.ndarray]): Variance of the hidden states.
        A (Optional[np.ndarray]):  Transition matrix.
        Q (Optional[np.ndarray]): Process noise covariance matrix.
        C (Optional[np.ndarray]): Observation matrix.
        num_states (int): Number of hidden states.
        num_param (int): Number of parameters in the component.
        component_name (str): Name of the component.
        parameter_name (str): Name of the parameter.
        states_name (List[str]): Names of the each hidden state in the component.
    """
    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        var: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        C: Optional[np.ndarray] = None,  
        sigma: Optional[float] = 0.0,
        num_states: int = 0,    
        num_param: int = 0, 
        component_name: Optional[str] = None,
        parameter_name: Optional[List[str]] = None,  
        states_name: Optional[List[str]] = None,
    ) -> None:
        # Initialize mu and var based on num_states if not provided
        if mu is not None:
            self.mu = np.atleast_2d(mu).T
            if self.mu.shape[0] != num_states:
                raise ValueError(f"Incorrect mu dimension.")
        else:
            self.mu = np.zeros((num_states, 1))

        if var is not None:
            self.var = np.atleast_2d(var).T
            if self.var.shape[0] != num_states:
                raise ValueError(f"Incorrect var dimension.")
        else:
            self.var = np.zeros((num_states, 1))

        self.A = A if A is not None else np.zeros((num_states, num_states))  
        self.Q = Q if Q is not None else np.zeros((num_states, num_states))  
        self.C = C if C is not None else np.zeros((num_states))  
        
        self.sigma = sigma
        self.num_states = num_states
        self.num_param = num_param
        self.component_name = component_name
        self.parameter_name = parameter_name
        self.states_name = states_name

    
    # @staticmethod
    # def init_states(self, data):
    #     self.mu = np.zeros(self.num_states)
    #     self.var = np.zeros(self.num_states)


    

    

    

