import numpy as np
from typing import Optional
from base_component import baseComponent

class Periodic(baseComponent):
    def __init__(
        self,
        period: float = 0.0,
        **kwargs
    ) -> None:
        self.period = period
        sigma = kwargs.get('sigma', 0.0)
        w = 2*np.pi/period
        super().__init__(
            num_states = 2, 
            num_param = 1,
            component_name = 'periodic',
            parameter_name = ['period','sigma'],
            A = np.array([[np.cos(w), np.sin(w)], [-np.sin(w), np.cos(w)]]),
            Q = sigma**2 * np.array([[1, 0],[0, 1]]),
            C = np.array([[1, 0]]),
            states_name = ['periodic #1', 'periodic #2'],
            **kwargs 
            )

    

    

    

