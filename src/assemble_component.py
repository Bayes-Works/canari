import numpy as np
from typing import Optional, List
from base_component import baseComponent
from common import block_diag


class Assemble_component(baseComponent):
    def __init__(self, components: List[baseComponent]) -> None:
        if not components:
            raise ValueError("At least one component must be provided for assembly.")
        
        # Extract matrices from each component
        A_matrices = [component.A for component in components]
        Q_matrices = [component.Q for component in components]
        C_matrices = [component.C for component in components]
        
        # Assemble A, Q, C
        assembled_A = block_diag(*A_matrices)
        assembled_Q = block_diag(*Q_matrices)
        assembled_C = np.array([])
        for component in components:
            assembled_C = np.concatenate((assembled_C,component.C[0,:]),axis=0)
        
        # Assemble mu_x and var_x
        assembled_mu_x = np.vstack([component.mu_x for component in components])
        assembled_var_x = np.vstack([component.var_x for component in components])
        
        # Combine component names, parameter names, and states names
        assembled_component_names = ', '.join([component.component_name for component in components])
        assembled_parameter_names = [param_name for component in components for param_name in component.parameter_name]
        assembled_states_names = [state for component in components for state in component.states_name]
        
        # Sum the number of states and parameters
        total_num_states = sum(component.num_states for component in components)
        total_num_param = sum(component.num_param for component in components)
        



    
    

    

    

