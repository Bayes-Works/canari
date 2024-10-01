import numpy as np
from typing import Optional
from base_component import baseComponent

# Local level component
class Local_level(baseComponent):
    def __init__(
        self,
        sigma_w : Optional[float] = 0.0,
        **kwargs 
    ) -> None:
        self.sigma_w = sigma_w
        super().__init__(
            num_states = 1, 
            num_param = 1,
            component_name = 'local level',
            parameter_name = ['sigma_w'],
            A = np.array([[1]]),
            Q = np.array([[self.sigma_w**2]]),
            C = np.array([[1]]),
            states_name = ['level'],
            **kwargs  
            )


# Local trend component
class Local_trend(baseComponent):
    def __init__(
        self,
        sigma_w : Optional[float] = 0.0,
        **kwargs 
    ) -> None:
        self.sigma_w = sigma_w
        super().__init__(
            num_states = 2, 
            num_param = 1,
            component_name = 'local trend',
            parameter_name = ['sigma_w'],
            A = np.array([[1, 1], [0, 1]]),
            Q = self.sigma_w**2 * np.array([[1/3, 1/2],[1/2, 1/3]]),
            C = np.array([[1,0]]),
            states_name = ['level','trend'],
            **kwargs 
            )
        
# Local acceleration component
class Local_acceleration(baseComponent):
    def __init__(
        self,
        sigma_w : Optional[float] = 0.0,
        **kwargs 
    ) -> None:
        self.sigma_w = sigma_w
        super().__init__(
            num_states = 3, 
            num_param = 1,
            component_name = 'local acceleration',
            parameter_name = ['sigma_w'],
            A = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]]),
            Q = self.sigma_w**2 * np.array([[1/20, 1/8, 1/6],[1/8, 1/3, 1/2], [1/6, 1/2, 1]]),
            C = np.array([[1, 0, 0]]),
            states_name = ['level','trend','acceralation'],
            **kwargs 
            )


    
    

    

    

