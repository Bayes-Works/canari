"""
This module defines one component "exponential".
"""

from typing import Optional, Tuple
import numpy as np
from canari.component.base_component import BaseComponent
from canari.common import GMA
from canari import common


class Exponential(BaseComponent):
    """
    `Exponential` class, inheriting from Canari's `BaseComponent`.
    It models exponential growth with a locally constant speed over time (linear level), which simulates the abscissa scale,
    and a constant amplitude, which simulates the ordinate scale.

    Args:
        std_error (Optional[float]): Standard deviation of the process noise. Defaults: initialized to 0.
        mu_states (Optional[list[float]]): Initial mean of the hidden state. Defaults:
            initialized to zeros.
        var_states (Optional[list[float]]): Initial variance of the hidden state. Defaults:
            initialized to zeros.

    Examples:
        >>> from canari.component import Exponential
        >>> # With known mu_states and var_states
        >>> exponential = Exponential(mu_states=[0, 0.15, 10, 0, 0], var_states=[0.04, 0.01, 1, 0, 0], std_error=0.3)
        >>> # With default mu_states and var_states
        >>> exponential = Exponential(std_error=0.2)
        >>> exponential.component_name
        'exp'
        >>> exponential.states_name
        ['latent level', 'latent trend', 'exp scale factor', 'exp', 'scaled exp']
        >>> exponential.transition_matrix
        array([ [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],)
        >>> exponential.observation_matrix
        array([[0, 0, 0, 0, 1]])
        >>> exponential.process_noise_matrix
        >>> exponential.mu_states
        >>> exponential.var_states
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[list[float]] = None,
        var_states: Optional[list[float]] = None,
    ):
        self.std_error = std_error
        self._mu_states = mu_states
        self._var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "exp"

    def initialize_num_states(self):
        self._num_states = 5

    def initialize_states_name(self):
        self._states_name = [
            "latent level",  # latent level
            "latent trend",  # latent trend
            "exp scale factor",  # exp scale factor
            "exp",  # exp
            "scaled exp",  # scaled exp
        ]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[0, 0, 0, 0, 1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = self.std_error**2 * np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
            self._mu_states[0] = -self._mu_states[1]
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the exponential component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the exponential component."
            )

    def _update_exp_and_scaled_exp(
        self, mu_states, var_states, var_states_behind, method
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply forward path exponential moment transformations.

        Updates prior state means and variances based on the exponential model.
        The modification is applied after that `latent level`, `latent trend` and `exp scale factor`
        are updated by the transition matrix.
        After that,the closed form solutions to compute the prior distribution of `exp`
        from `latent level` and `latent trend`.
        GMA is also applied to `exp scale factor` and `exp` to get the prior distribution
        of `scaled exp`.
        These are used during the forward pass when exponential components are present.

        Args:
            mu_states_prior (np.ndarray): Prior mean vector of the states.
            var_states_prior (np.ndarray): Prior variance-covariance matrix of the states.
            var_states (np.ndarray): Variance-covariance matrix before the linear update
                                        of the states

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Updated (mu_states_prior, var_states_prior, mu_obs_predict, var_obs_predict).
        """

        model = self.model
        latent_level_index = model.get_states_index("latent level")
        latent_trend_index = model.get_states_index("latent trend")
        exp_scale_factor_index = model.get_states_index("exp scale factor")
        exp_index = model.get_states_index("exp")
        scaled_exp_index = model.get_states_index("scaled exp")

        mu_ll = np.asarray(mu_states[latent_level_index]).item()
        var_ll = np.asarray(var_states[latent_level_index, latent_level_index]).item()

        mu_states[exp_index] = np.exp(-mu_ll + 0.5 * var_ll) - 1

        var_states[exp_index, exp_index] = np.exp(-2 * mu_ll + var_ll) * (
            np.exp(var_ll) - 1
        )

        var_states[latent_level_index, exp_index] = -var_ll * np.exp(
            -mu_ll + 0.5 * var_ll
        )

        var_states[exp_index, latent_level_index] = var_states[
            latent_level_index, exp_index
        ]

        if method == "forward":
            skip_index = {latent_level_index, latent_trend_index, exp_index}
            var_states[latent_trend_index, exp_index] = -np.exp(
                -mu_ll + 0.5 * var_ll
            ) * (
                var_states_behind[latent_trend_index, latent_trend_index]
                + var_states_behind[latent_level_index, latent_trend_index]
            )
            var_states[exp_index, latent_trend_index] = var_states[
                latent_trend_index, exp_index
            ]
        elif method in {"backward", "smoother"}:
            skip_index = {latent_level_index, exp_index}

        magnitud_normal_space_exponential_space = (
            var_states[exp_index, latent_level_index]
            / var_states[latent_level_index, latent_level_index]
        )
        for other_component_index in range(len(mu_states)):
            if other_component_index in skip_index:
                continue
            cov_other_component_index = (
                magnitud_normal_space_exponential_space
                * var_states[latent_level_index, other_component_index]
            )
            var_states[exp_index, other_component_index] = cov_other_component_index
            var_states[other_component_index, exp_index] = cov_other_component_index

        mu_states, var_states = GMA(
            mu_states,
            var_states,
            index1=exp_scale_factor_index,
            index2=exp_index,
            replace_index=scaled_exp_index,
        ).get_results()

        return (
            mu_states,
            var_states,
        )
    
    def forward(self):
        """
        Apply forward path exponential moment transformations.

        Updates prior state means and variances based on the exponential model.
        The modification is applied after that `latent level`, `latent trend` and `exp scale factor`
        are updated by the transition matrix.
        After that,the closed form solutions to compute the prior distribution of `exp`
        from `latent level` and `latent trend`.
        GMA is also applied to `exp scale factor` and `exp` to get the prior distribution
        of `scaled exp`.
        These are used during the forward pass when exponential components are present.
        """

        model = self.model

        model.mu_states_prior, model.var_states_prior = (
            self._update_exp_and_scaled_exp(
                model.mu_states_prior, model.var_states_prior, model.var_states, "forward"
            )
        )


    def backward(self):
        """
        Backward modification for each component
        """

        model = self.model

        model.mu_states_posterior, model.var_states_posterior = (
            self._update_exp_and_scaled_exp(
                model.mu_states_posterior, model.var_states_posterior, 0, "backward"
            )
        )

    def rts_smoother(self, time_step:int):
        """
        RTS smoother modification for each component
        """

        model = self.model

        (
            model.states.mu_smooth[time_step],
            model.states.var_smooth[time_step]
        ) = self._update_exp_and_scaled_exp(
            model.states.mu_smooth[time_step],
            model.states.var_smooth[time_step],
            0,
            "smoother",
        )        


