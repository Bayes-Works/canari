import numpy as np
from typing import Optional
from base_component import baseComponent

# Local level component
class Local_level(baseComponent):
    def __init__(
        self,
        **kwargs 
    ) -> None:
        sigma = kwargs.get('sigma',0.0)
        super().__init__(
            num_states = 1, 
            num_param = 1,
            component_name = 'local level',
            parameter_name = ['sigma'],
            A = np.array([[1]]),
            Q = np.array([[sigma**2]]),
            C = np.array([[1]]),
            states_name = ['level'],
            **kwargs  
            )


# Local trend component
class Local_trend(baseComponent):
    def __init__(
        self,
        **kwargs 
    ) -> None:
        sigma = kwargs.get('sigma', 0.0)
        super().__init__(
            num_states = 2, 
            num_param = 1,
            component_name = 'local trend',
            parameter_name = ['sigma'],
            A = np.array([[1, 1], [0, 1]]),
            Q = sigma**2 * np.array([[1/3, 1/2],[1/2, 1/3]]),
            C = np.array([[1,0]]),
            states_name = ['level','trend'],
            **kwargs 
            )
        
# Local acceleration component
class Local_acceleration(baseComponent):
    def __init__(
        self,
        **kwargs 
    ) -> None:
        sigma = kwargs.get('sigma', 0.0)
        super().__init__(
            num_states = 3, 
            num_param = 1,
            component_name = 'local acceleration',
            parameter_name = ['sigma'],
            A = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]]),
            Q = sigma**2 * np.array([[1/20, 1/8, 1/6],[1/8, 1/3, 1/2], [1/6, 1/2, 1]]),
            C = np.array([[1, 0, 0]]),
            states_name = ['level','trend','acceralation'],
            **kwargs 
            )


    
    

    

    

