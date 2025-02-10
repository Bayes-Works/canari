from typing import Tuple, Optional, Dict
import numpy as np
from src.data_struct import LstmOutputHistory


def create_block_diag(*arrays: np.ndarray) -> np.ndarray:
    """
    Create a block diagonal matrix from the provided arrays.

    Parameters:
        *arrays (np.ndarray): Variable number of 2D arrays to be placed along the diagonal.

    Returns:
        np.ndarray: A block diagonal matrix.
    """

    if not arrays:
        return np.array([[]])
    total_rows = sum(a.shape[0] for a in arrays)
    total_cols = sum(a.shape[1] for a in arrays)
    block_matrix = np.zeros((total_rows, total_cols))
    current_row = 0
    current_col = 0
    for a in arrays:
        rows, cols = a.shape
        block_matrix[
            current_row : current_row + rows, current_col : current_col + cols
        ] = a
        current_row += rows
        current_col += cols
    return block_matrix


def calc_observation(
    mu_states: np.ndarray,
    var_states: np.ndarray,
    observation_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    mu_obs_predicted = observation_matrix @ mu_states
    var_obs_predicted = observation_matrix @ var_states @ observation_matrix.T
    return mu_obs_predicted, var_obs_predicted


def forward(
    mu_states_posterior: np.ndarray,
    var_states_posterior: np.ndarray,
    transition_matrix: np.ndarray,
    process_noise_matrix: np.ndarray,
    observation_matrix: np.ndarray,
    mu_lstm_pred: Optional[np.ndarray] = None,
    var_lstm_pred: Optional[np.ndarray] = None,
    lstm_indice: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    mu_states_prior = transition_matrix @ mu_states_posterior
    var_states_prior = (
        transition_matrix @ var_states_posterior @ transition_matrix.T
        + process_noise_matrix
    )

    if mu_lstm_pred:
        mu_states_prior[lstm_indice] = mu_lstm_pred.item()
        var_states_prior[lstm_indice, lstm_indice] = var_lstm_pred.item()

    mu_obs_predicted, var_obs_predicted = calc_observation(
        mu_states_prior, var_states_prior, observation_matrix
    )
    return mu_obs_predicted, var_obs_predicted, mu_states_prior, var_states_prior


def backward(
    obs: float,
    mu_obs_predicted: np.ndarray,
    var_obs_predicted: np.ndarray,
    var_states_prior: np.ndarray,
    observation_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    # var_obs_predicted = var_obs_predicted + 1e-20
    cov_obs_states = observation_matrix @ var_states_prior
    delta_mu_states = cov_obs_states.T / var_obs_predicted @ (obs - mu_obs_predicted)
    delta_var_states = -cov_obs_states.T / var_obs_predicted @ cov_obs_states
    return delta_mu_states, delta_var_states


def rts_smoother(
    mu_states_prior: np.ndarray,
    var_states_prior: np.ndarray,
    mu_states_smooth: np.ndarray,
    var_states_smooth: np.ndarray,
    mu_states_posterior: np.ndarray,
    var_states_posterior: np.ndarray,
    cross_cov_states: np.ndarray,
    matrix_inversion_tol: Optional[float] = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:

    jcb = cross_cov_states @ np.linalg.pinv(
        var_states_prior, rcond=matrix_inversion_tol
    )
    mu_states_smooth = mu_states_posterior + jcb @ (mu_states_smooth - mu_states_prior)
    var_states_smooth = (
        var_states_posterior + jcb @ (var_states_smooth - var_states_prior) @ jcb.T
    )
    return mu_states_smooth, var_states_smooth


def prepare_lstm_input(
    lstm_output_history: LstmOutputHistory, input_covariates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare input for lstm network, concatenate lstm output history and the input covariates
    """
    mu_lstm_input = np.concatenate((lstm_output_history.mu, input_covariates))
    mu_lstm_input = np.nan_to_num(mu_lstm_input, nan=0.0)
    var_lstm_input = np.concatenate(
        (
            lstm_output_history.var,
            np.zeros(len(input_covariates), dtype=np.float32),
        )
    )
    return mu_lstm_input, var_lstm_input


def pad_matrix(
    matrix: np.ndarray,
    pad_index: int,
    pad_row: Optional[np.ndarray] = None,
    pad_col: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray]:
    """
    Add padding for a matrix
    """

    if pad_row is not None:
        matrix = np.insert(matrix, pad_index, pad_row, axis=0)
    if pad_col is not None:
        matrix = np.insert(matrix, pad_index, pad_col, axis=1)
    return matrix


def gaussian_mixture(
    mu1: np.ndarray,
    var1: np.ndarray,
    coef1: float,
    mu2: np.ndarray,
    var2: np.ndarray,
    coef2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian reduction mixture
    """
    if mu1.ndim == 1:
        mu1 = np.atleast_2d(mu1).T
    if mu2.ndim == 1:
        mu2 = np.atleast_2d(mu2).T
    mu_mixture = mu1 * coef1 + mu2 * coef2
    m1 = mu1 - mu_mixture
    m2 = mu2 - mu_mixture
    var_mixture = coef1 * (var1 + m1 @ m1.T) + coef2 * (var2 + m2 @ m2.T)
    return mu_mixture, var_mixture


def set_default_input_covariates(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Ensure that 'x' is initialized in the data dictionary.
    If 'x' is missing, initialize it with NaNs.
    """

    num_time_steps = len(data["y"])
    if "x" not in data:
        data["x"] = np.full(num_time_steps, np.nan)
    return data


class GMA(object):
    """
    Gaussian Multiplicative Approximation of two variables in a vector
    Input:
    mu: mean vector of all the variables
    var: full covariance matrix of all the variables
    """

    def __init__(
        self,
        mu: np.ndarray,
        var: np.ndarray,
        index1: Optional[int] = None,
        index2: Optional[int] = None,
        replace_index: Optional[int] = None,
    ) -> None:

        self.mu = mu
        self.var = var
        if index1 is not None and index2 is not None and replace_index is not None:
            self.multiply_and_augment(index1, index2)
            self.swap(-1, replace_index)
            self.delete(-1)

    def multiply_and_augment(self, index1, index2):
        """
        The multiplication of two variables is augmented to the last index of the provided vector
        """

        # Augment the dimension of the input matrix
        GMA_mu = np.vstack((self.mu, 0))
        GMA_var = np.append(self.var, np.zeros((1, self.var.shape[1])), axis=0)
        GMA_var = np.append(GMA_var, np.zeros((GMA_var.shape[0], 1)), axis=1)

        # Multiply the two provided indices
        # # Mean for the multiplicated term
        GMA_mu[-1] = self.mu[index1] * self.mu[index2] + self.var[index1][index2]
        # # Variance for the multiplicated term
        GMA_var[-1, -1] = (
            self.var[index1][index1] * self.var[index2][index2]
            + self.var[index1][index2] ** 2
            + 2 * self.mu[index1] * self.mu[index2] * self.var[index1][index2]
            + self.var[index1][index1] * self.mu[index2] ** 2
            + self.var[index2][index2] * self.mu[index1] ** 2
        )
        # # Covariance between the multiplicated term and the existing terms
        for i in range(len(self.mu)):
            cov_i = (
                self.var[i][index1] * self.mu[index2]
                + self.var[i][index2] * self.mu[index1]
            )
            GMA_var[i][-1] = cov_i
            GMA_var[-1][i] = cov_i

        self.mu = GMA_mu
        self.var = GMA_var

    def swap(self, index1, index2):
        """
        Swap the sequence of moments of two variables in the vector
        """

        self.mu[[index1, index2]] = self.mu[[index2, index1]]
        self.var[[index1, index2]] = self.var[[index2, index1]]
        self.var[:, [index1, index2]] = self.var[:, [index2, index1]]

    def delete(self, index):
        """
        Delete the moments of a variables in the vector
        """

        self.mu = np.delete(self.mu, index, axis=0)
        self.var = np.delete(self.var, index, axis=0)
        self.var = np.delete(self.var, index, axis=1)

    def get_results(self):
        return self.mu, self.var


def online_AR_forward_modification(states_name, mu_states, var_states, process_noise_matrix=None, mu_W2bar=None, var_W2bar=None):
    """
    Online AR modification
    """

    ar_index = states_name.index("autoregression")
    mu_W2_prior = None
    var_W2_prior = None
    if "phi" in states_name:
        phi_index = states_name.index("phi")
        # GMA operations
        mu_states, var_states = GMA(mu_states, var_states, 
                                    index1=phi_index, index2=ar_index, replace_index=ar_index).get_results()
        # Cap phi_AR if it is bigger than 1: for numerical stability in BAR later
        mu_states[phi_index] = 0.9999 if mu_states[phi_index] >= 1 else mu_states[phi_index]

    if "AR_error" in states_name:
        ar_error_index = states_name.index("AR_error")
        W2_index = states_name.index("W2")
        W2bar_index = states_name.index("W2bar")
        # Forward path to compute the moments of W
        # # W2bar
        mu_states[W2bar_index] = mu_W2bar
        var_states[W2bar_index, W2bar_index] = var_W2bar
        # # From W2bar to W2
        mu_W2_prior = mu_W2bar
        var_W2_prior = 3 * var_W2bar + 2 * mu_W2bar ** 2
        mu_states[W2_index] = mu_W2_prior
        var_states[W2_index, W2_index] = var_W2_prior
        # # From W2 to W
        mu_states[ar_error_index] = 0
        var_states[ar_error_index, :] = np.zeros_like(var_states[ar_error_index, :])
        var_states[:, ar_error_index] = np.zeros_like(var_states[:, ar_error_index])
        var_states[ar_error_index, ar_error_index] = mu_W2bar
        var_states[ar_error_index, ar_index] = mu_W2bar
        var_states[ar_index, ar_error_index] = mu_W2bar

        # Replace the process error variance in self.process_noise_matrix
        process_noise_matrix[ar_index, ar_index] = mu_W2bar

    return mu_states, var_states, process_noise_matrix, mu_W2_prior, var_W2_prior

def online_AR_backward_modification(states_name, mu_states_posterior, var_states_posterior, mu_W2bar, var_W2bar, mu_W2_prior, var_W2_prior):
    ar_error_index = states_name.index("AR_error")
    W2_index = states_name.index("W2")
    W2bar_index = states_name.index("W2bar")

    # Backward path to update W2 and W2bar
    # # From W to W2
    mu_W2_posterior = mu_states_posterior[ar_error_index] ** 2 + var_states_posterior[ar_error_index, ar_error_index]
    var_W2_posterior = 2 * var_states_posterior[ar_error_index, ar_error_index] ** 2 + 4 * var_states_posterior[ar_error_index, ar_error_index] * mu_states_posterior[ar_error_index] ** 2
    mu_states_posterior[W2_index] = mu_W2_posterior
    var_states_posterior[W2_index, :] = np.zeros_like(var_states_posterior[W2_index, :] )
    var_states_posterior[:, W2_index] = np.zeros_like(var_states_posterior[:, W2_index])
    var_states_posterior[W2_index, W2_index] = var_W2_posterior
    # # From W2 to W2bar
    K = var_W2bar / var_W2_prior
    mu_W2bar = mu_W2bar + K * (mu_W2_posterior - mu_W2_prior)
    var_W2bar = var_W2bar + K**2 * (var_W2_posterior - var_W2_prior)
    mu_states_posterior[W2bar_index] = mu_W2bar
    var_states_posterior[W2bar_index, :] = np.zeros_like(var_states_posterior[W2bar_index, :])
    var_states_posterior[:, W2bar_index] = np.zeros_like(var_states_posterior[:, W2bar_index])
    var_states_posterior[W2bar_index, W2bar_index] = var_W2bar

    return mu_states_posterior, var_states_posterior, mu_W2bar, var_W2bar