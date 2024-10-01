import numpy as np
from typing import Optional
from base_component import baseComponent


class Autoregression_deterministic(baseComponent):
    def __init__(
        self,
        phi_AR : Optional[float] = 0.0,
        sigma_AR: Optional[float] = 0.0,
        **kwargs
    ) -> None:
        self.phi_AR = phi_AR
        self.sigma_AR = sigma_AR
        super().__init__(
            num_states = 1, 
            num_param = 2,
            component_name = 'autoregression deterministic',
            parameter_name = ['phi_AR','sigma_AR'],
            A = np.array([[self.phi_AR]]),
            Q = np.array([[self.sigma_AR**2]]),
            C = np.array([[1]]),
            states_name = ['AR_hidden state'],
            **kwargs 
            )


    
    

    

    

