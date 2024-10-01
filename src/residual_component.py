import numpy as np
from typing import Optional
from base_component import baseComponent


class Autoregression(baseComponent):
    def __init__(
        self,
        phi : Optional[float] = 0.0,
        **kwargs
    ) -> None:
        self.phi = phi
        sigma = kwargs.get('sigma', 0.0)
        super().__init__(
            num_states = 1, 
            num_param = 2,
            component_name = 'autoregression',
            parameter_name = ['phi','sigma'],
            A = np.array([[self.phi]]),
            Q = np.array([[sigma**2]]),
            C = np.array([[1]]),
            states_name = ['AR'],
            **kwargs 
            )


    
    

    

    

