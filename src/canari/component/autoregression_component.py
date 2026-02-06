from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent
from canari.common import GMA


class Autoregression(BaseComponent):
    """
    `Autoregression` class, inheriting from Canari's `BaseComponent`.
    It models residuals following a univariate AR(1) process with optional treatement and estimation of
    the autoregressive coefficient (`phi`) and process noise standard deviation (`std_error`) as hidden states.

    Parameters:
        std_error (Optional[float]): Standard deviation of the process noise. Defaults to `None`.
            If None, it will be learned using Approximate Gaussian Variance Inference (AGVI).
        phi (Optional[float]): Autoregressive coefficient. Defaults to `None`.
            If None, it will be learned using Gaussian Multiplicative Approximation (GMA).
        mu_states (Optional[list[float]]): Initial mean of the hidden state. Defaults:
            initialized to zeros.
        var_states (Optional[list[float]]): Initial variance of the hidden state. Defaults:
            initialized to zeros.

    Behavior:
        - Adds 2 extra (dummy) states if `phi` is None.
        - Adds 3 extra (dummy) states if `std_error` is None.

    References:
        Deka, B., Nguyen, L.H. and Goulet, J.-A. (2024). `Analytically Tractable Heteroscedastic
        Uncertainty Quantification in Bayesian Neural Networks for Regression Tasks
        <https://www.sciencedirect.com/science/article/abs/pii/S0925231223013061>`_.
        Neurocomputing. Volume 572, pp.127183.

        Deka, B. and Goulet, J.-A. (2023). `Approximate Gaussian Variance Inference for State-Space
        Models <https://onlinelibrary.wiley.com/doi/full/10.1002/acs.3667>`_.
        International Journal of Adaptive Control and Signal Processing.
        Volume 37, Issue 11, pp. 2934-2962.

    Examples:
        >>> from canari.component import Autoregression
        >>> # With known parameters
        >>> ar = Autoregression(phi=0.9, std_error=0.1)
        >>> # With known mu_states and var_states
        >>> ar = Autoregression(mu_states=[0.1], var_states=[0.1], phi=0.9, std_error=0.1)
        >>> # With parameters to be estimated
        >>> ar = Autoregression()
        >>> ar.component_name
        autoregression
        >>> ar.states_name
        ['autoregression', 'phi', 'phi_autoregression', 'AR_error', 'W2', 'W2bar']
        >>> ar.mu_states
        >>> ar.var_states
        >>> ar.transition_matrix
        >>> ar.observation_matrix
        >>> ar.process_noise_matrix
    """

    def __init__(
        self,
        std_error: Optional[float] = None,
        phi: Optional[float] = None,
        mu_states: Optional[list[float]] = None,
        var_states: Optional[list[float]] = None,
    ):
        self.std_error = (
            std_error  # When std_error is None, use AGVI to learn the process error
        )
        self.phi = phi  # When phi is None, use GMA to learn
        self._mu_states = mu_states
        self._var_states = var_states
        self.mu_W2bar = None
        self.var_W2bar = None
        self.mu_W2_prior = None
        self.var_W2_prior = None

        super().__init__()

    def initialize_component_name(self):
        self._component_name = "autoregression"

    def initialize_num_states(self):
        self._num_states = 1
        if self.phi is None:
            self._num_states += 2
        if self.std_error is None:
            self._num_states += 3

    def initialize_states_name(self):
        self._states_name = ["autoregression"]
        if self.phi is None:
            self._states_name.append("phi")
            self._states_name.append("phi_autoregression")  # phi^{AR} times X^{AR}
        if self.std_error is None:
            self._states_name.append(
                "AR_error"
            )  # Process error of AR (W variable in AGVI)
            self._states_name.append("W2")  # Square of the process error
            self._states_name.append("W2bar")  # Expected value of W2

    def initialize_transition_matrix(self):
        if self.phi is None:
            self._transition_matrix = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
        else:
            self._transition_matrix = np.array([[self.phi]])
        if self.std_error is None:
            self._transition_matrix = np.block(
                [
                    [self._transition_matrix, np.zeros((self._num_states - 3, 3))],
                    [np.zeros((3, self._num_states))],
                ]
            )

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1]])
        if self.phi is None:
            self._observation_matrix = np.hstack(
                (self._observation_matrix, np.zeros((1, 2)))
            )
        if self.std_error is None:
            self._observation_matrix = np.hstack(
                (self._observation_matrix, np.zeros((1, 3)))
            )

    def initialize_process_noise_matrix(self):
        if self.std_error is not None:
            self._process_noise_matrix = np.array([[self.std_error**2]])
        else:
            self._process_noise_matrix = np.array([[self._mu_states[-1]]])
            self.mu_W2bar = self._mu_states[-1]
            self.var_W2bar = self._var_states[-1]

        if self.phi is None:
            self._process_noise_matrix = np.block(
                [[self._process_noise_matrix, np.zeros((1, 2))], [np.zeros((2, 3))]]
            )
        if self.std_error is None:
            self._process_noise_matrix = np.block(
                [
                    [self._process_noise_matrix, np.zeros((self._num_states - 3, 3))],
                    [np.zeros((3, self._num_states))],
                ]
            )

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the autoregression component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the autoregression component."
            )

    def forward(self):
        """
        Forward modification for AR component.
        Apply forward path autoregressive (AR) moment transformations.

        Updates prior state means and variances based on the autoregressive error model,
        propagating uncertainty from W2bar to AR_error. These are used during the forward
        pass when AR components are present.
        """

        model = self.model
        ar_index = model.get_states_index("autoregression")
        ar_error_index = model.get_states_index("AR_error")
        W2_index = model.get_states_index("W2")
        W2bar_index = model.get_states_index("W2bar")

        # Forward path to compute the moments of W
        # # W2bar
        model.mu_states_prior[W2bar_index] = self.mu_W2bar
        model.var_states_prior[W2bar_index, W2bar_index] = self.var_W2bar

        # # From W2bar to W2
        self.mu_W2_prior = self.mu_W2bar
        self.var_W2_prior = 3 * self.var_W2bar + 2 * self.mu_W2bar**2
        model.mu_states_prior[W2_index] = self.mu_W2_prior
        model.var_states_prior[W2_index, W2_index] = self.var_W2_prior

        # # From W2 to W
        model.mu_states_prior[ar_error_index] = 0
        model.var_states_prior[ar_error_index, :] = np.zeros_like(
            model.var_states_prior[ar_error_index, :]
        )
        model.var_states_prior[:, ar_error_index] = np.zeros_like(
            model.var_states_prior[:, ar_error_index]
        )
        model.var_states_prior[ar_error_index, ar_error_index] = self.mu_W2bar
        model.var_states_prior[ar_error_index, ar_index] = self.mu_W2bar
        model.var_states_prior[ar_index, ar_error_index] = self.mu_W2bar

    def backward(self):
        """
        Apply backward AR moment updates during state-space filtering.

        Computes the posterior distribution of W2 and W2bar from AR_error states,
        and adjusts the autoregressive process noise accordingly. Also applies
        GMA transformations when "phi" is involved in the model.
        """

        model = self.model
        if "phi" in self.states_name:
            # GMA operations
            model.mu_states_posterior, model.var_states_posterior = GMA(
                model.mu_states_posterior,
                model.var_states_posterior,
                index1=model.get_states_index("phi"),
                index2=model.get_states_index("autoregression"),
                replace_index=model.get_states_index("phi_autoregression"),
            ).get_results()

        if "AR_error" in self.states_name:
            ar_index = model.get_states_index("autoregression")
            ar_error_index = model.get_states_index("AR_error")
            W2_index = model.get_states_index("W2")
            W2bar_index = model.get_states_index("W2bar")

            # Backward path to update W2 and W2bar
            # # From W to W2
            mu_W2_posterior = (
                model.mu_states_posterior[ar_error_index] ** 2
                + model.var_states_posterior[ar_error_index, ar_error_index]
            )
            var_W2_posterior = (
                2 * model.var_states_posterior[ar_error_index, ar_error_index] ** 2
                + 4
                * model.var_states_posterior[ar_error_index, ar_error_index]
                * model.mu_states_posterior[ar_error_index] ** 2
            )
            model.mu_states_posterior[W2_index] = mu_W2_posterior
            model.var_states_posterior[W2_index, :] = np.zeros_like(
                model.var_states_posterior[W2_index, :]
            )
            model.var_states_posterior[:, W2_index] = np.zeros_like(
                model.var_states_posterior[:, W2_index]
            )
            model.var_states_posterior[W2_index, W2_index] = var_W2_posterior.item()

            # # From W2 to W2bar
            K = self.var_W2bar / self.var_W2_prior
            self.mu_W2bar = self.mu_W2bar + K * (mu_W2_posterior - self.mu_W2_prior)
            self.var_W2bar = self.var_W2bar + K**2 * (
                var_W2_posterior - self.var_W2_prior
            )
            model.mu_states_posterior[W2bar_index] = self.mu_W2bar
            model.var_states_posterior[W2bar_index, :] = np.zeros_like(
                model.var_states_posterior[W2bar_index, :]
            )
            model.var_states_posterior[:, W2bar_index] = np.zeros_like(
                model.var_states_posterior[:, W2bar_index]
            )
            model.var_states_posterior[W2bar_index, W2bar_index] = self.var_W2bar

            model.process_noise_matrix[ar_index, ar_index] = self.mu_W2bar

