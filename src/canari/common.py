"""
Utility functions that are used in mulitple classes.
"""

from typing import Tuple, Optional
import numpy as np
from pytagi import Normalizer
from canari.data_struct import LstmOutputHistory


def create_block_diag(*arrays: np.ndarray) -> np.ndarray:
    """
    Create a block diagonal matrix from the provided arrays.

    Args:
        *arrays (np.ndarray): Variable number of 2D arrays.

    Returns:
        np.ndarray: Block diagonal matrix with input arrays along the diagonal.
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
    """
    Estimate observation mean and variance from hidden states and the observation_matrix.

    Args:
        mu_states (np.ndarray): Mean of the state variables.
        var_states (np.ndarray): Covariance matrix of the state variables.
        observation_matrix (np.ndarray): Observation model matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted mean and variance of observations.
    """
    mu_obs_predict = observation_matrix @ mu_states
    var_obs_predict = observation_matrix @ var_states @ observation_matrix.T
    return mu_obs_predict, var_obs_predict


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
    """
    Perform the prediction step in Kalman filter.

    Args:
        mu_states_posterior (np.ndarray): Posterior hidden states mean vector.
        var_states_posterior (np.ndarray): Posterior hidden state covariance matrix.
        transition_matrix (np.ndarray): Transition matrix.
        process_noise_matrix (np.ndarray): Process noise matrix.
        observation_matrix (np.ndarray): Observation matrix.
        mu_lstm_pred (Optional[np.ndarray]): LSTM predicted mean.
        var_lstm_pred (Optional[np.ndarray]): LSTM predicted variance.
        lstm_indice (Optional[int]): Index to insert LSTM predictions.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Predicted observation mean/var and state prior mean/var.
    """
    mu_states_prior = transition_matrix @ mu_states_posterior
    var_states_prior = (
        transition_matrix @ var_states_posterior @ transition_matrix.T
        + process_noise_matrix
    )
    if mu_lstm_pred is not None:
        mu_states_prior[lstm_indice] = mu_lstm_pred.item()
        var_states_prior[lstm_indice, lstm_indice] = var_lstm_pred.item()

    mu_obs_predict, var_obs_predict = calc_observation(
        mu_states_prior, var_states_prior, observation_matrix
    )
    return (
        np.float32(mu_obs_predict),
        np.float32(var_obs_predict),
        np.float32(mu_states_prior),
        np.float32(var_states_prior),
    )


def backward(
    obs: float,
    mu_obs_predict: np.ndarray,
    var_obs_predict: np.ndarray,
    var_states_prior: np.ndarray,
    observation_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform the backward step in Kalman filter.

    Args:
        obs (float): Observation.
        mu_obs_predict (np.ndarray): Predicted mean.
        var_obs_predict (np.ndarray): Predicted variance.
        var_states_prior (np.ndarray): Prior covariance matrix for hidden states.
        observation_matrix (np.ndarray): Observation matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Delta/Correction for hidden states' mean and covariance matrix.
    """
    cov_obs_states = observation_matrix @ var_states_prior
    delta_mu_states = cov_obs_states.T / var_obs_predict @ (obs - mu_obs_predict)
    delta_var_states = -cov_obs_states.T / var_obs_predict @ cov_obs_states
    return (
        np.float32(delta_mu_states),
        np.float32(delta_var_states),
    )


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
    """
    Rauch-Tung-Striebel (RTS) smoother.

    Args:
        mu_states_prior (np.ndarray): Prior mean vector.
        var_states_prior (np.ndarray): Prior covariance metrix.
        mu_states_smooth (np.ndarray): Smoothed mean vector.
        var_states_smooth (np.ndarray): Smoothed covariance matrix.
        mu_states_posterior (np.ndarray): Posterior mean vector.
        var_states_posterior (np.ndarray): Posterior covariance matrix.
        cross_cov_states (np.ndarray): Cross-covariance matrix between hidden states at two consecutive time steps.
        matrix_inversion_tol (Optional[float]): Regularization tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated smoothed mean and covariance.
    """
    jcb = cross_cov_states @ np.linalg.pinv(
        var_states_prior, rcond=matrix_inversion_tol
    )
    mu_states_smooth = mu_states_posterior + jcb @ (mu_states_smooth - mu_states_prior)
    var_states_smooth = (
        var_states_posterior + jcb @ (var_states_smooth - var_states_prior) @ jcb.T
    )
    return (
        np.float32(mu_states_smooth),
        np.float32(var_states_smooth),
    )


def prepare_lstm_input(
    lstm_output_history: LstmOutputHistory, input_covariates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare LSTM input by concatenating past LSTM outputs with current input covariates.

    Args:
        lstm_output_history (LstmOutputHistory): Historical LSTM mean/variance.
        input_covariates (np.ndarray): Input covariates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: LSTM input mean and variance vectors.
    """
    mu_lstm_input = np.concatenate((lstm_output_history.mu, input_covariates))
    mu_lstm_input = np.nan_to_num(mu_lstm_input, nan=0.0)
    var_lstm_input = np.concatenate(
        (lstm_output_history.var, np.zeros(len(input_covariates), dtype=np.float32))
    )
    return np.float32(mu_lstm_input), np.float32(var_lstm_input)


def pad_matrix(
    matrix: np.ndarray,
    pad_index: int,
    pad_row: Optional[np.ndarray] = None,
    pad_col: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Add a row and/or column padding to the matrix.

    Args:
        matrix (np.ndarray): Matrix to pad.
        pad_index (int): Index to insert the padding.
        pad_row (Optional[np.ndarray]): Row vector to insert.
        pad_col (Optional[np.ndarray]): Column vector to insert.

    Returns:
        np.ndarray: Padded matrix.
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
    Perform a reduction of two Gaussian distributions into a single mixture distribution.

    Args:
        mu1 (np.ndarray): Mean vector of the first Gaussian.
        var1 (np.ndarray): Covariance matrix of the first Gaussian.
        coef1 (float): Mixture weight for the first Gaussian.
        mu2 (np.ndarray): Mean vector of the second Gaussian.
        var2 (np.ndarray): Covariance matrix of the second Gaussian.
        coef2 (float): Mixture weight for the second Gaussian.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mixture mean vector and covariance matrix.
    """
    if mu1.ndim == 1:
        mu1 = np.atleast_2d(mu1).T
    if mu2.ndim == 1:
        mu2 = np.atleast_2d(mu2).T
    mu_mixture = mu1 * coef1 + mu2 * coef2
    m1 = mu1 - mu_mixture
    m2 = mu2 - mu_mixture
    var_mixture = coef1 * (var1 + m1 @ m1.T) + coef2 * (var2 + m2 @ m2.T)
    return np.float32(mu_mixture), np.float32(var_mixture)


def unstandardize_states(
    mu_norm: dict, std_norm: dict, norm_const_mean: float, norm_const_std: float
) -> Tuple[dict, dict]:
    """
    Unstandardize standardized mean and standard deviation of states.

    Args:
        mu_norm (dict): Standardized means by key.
        std_norm (dict): Standardized stds by key.
        norm_const_mean (float): Mean standardization constant.
        norm_const_std (float): Standard deviation standardization constant.

    Returns:
        Tuple[dict, dict]: Unstandardized means and stds.
    """
    mu_unnorm = {}
    std_unnorm = {}
    for key in mu_norm:
        _mu_norm = mu_norm[key]
        _std_norm = std_norm[key]
        _norm_const_mean = norm_const_mean if key == "local level" else 0
        mu_unnorm[key] = Normalizer.unstandardize(
            _mu_norm, _norm_const_mean, norm_const_std
        )
        std_unnorm[key] = Normalizer.unstandardize_std(_std_norm, norm_const_std)
    return mu_unnorm, std_unnorm


class GMA(object):
    """
    Gaussian Multiplicative Approximation (GMA).

    Approximate the product of two Gaussian variables by a Gausian distribution
    with exact moments calculation. The class allow augmenting the size of the
     state vector in order to include the product term.

    Attributes:
        mu (np.ndarray): Mean vector.
        var (np.ndarray): Covariance matrix.
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

    def multiply_and_augment(self, index1: int, index2: int):
        """
        Augment the state vector with mean and covariance from the product of
        the two variables referred by the two input indices.

        Args:
            index1 (int): Index of the first variable.
            index2 (int): Index of the second variable.
        """
        GMA_mu = np.vstack((self.mu, 0))
        GMA_var = np.append(self.var, np.zeros((1, self.var.shape[1])), axis=0)
        GMA_var = np.append(GMA_var, np.zeros((GMA_var.shape[0], 1)), axis=1)
        GMA_mu[-1] = self.mu[index1] * self.mu[index2] + self.var[index1][index2]
        GMA_var[-1, -1] = (
            self.var[index1][index1] * self.var[index2][index2]
            + self.var[index1][index2] ** 2
            + 2 * self.mu[index1] * self.mu[index2] * self.var[index1][index2]
            + self.var[index1][index1] * self.mu[index2] ** 2
            + self.var[index2][index2] * self.mu[index1] ** 2
        )
        for i in range(len(self.mu)):
            cov_i = (
                self.var[i][index1] * self.mu[index2]
                + self.var[i][index2] * self.mu[index1]
            )
            GMA_var[i][-1] = cov_i
            GMA_var[-1][i] = cov_i
        self.mu = GMA_mu
        self.var = GMA_var

    def swap(self, index1: int, index2: int):
        """
        Swap two variables in the mean and covariance structures.

        Args:
            index1 (int): Index of the first variable.
            index2 (int): Index of the second variable.
        """
        self.mu[[index1, index2]] = self.mu[[index2, index1]]
        self.var[[index1, index2]] = self.var[[index2, index1]]
        self.var[:, [index1, index2]] = self.var[:, [index2, index1]]

    def delete(self, index: int):
        """
        Remove a variable from the mean and covariance structures.

        Args:
            index (int): Index of the variable to delete.
        """
        self.mu = np.delete(self.mu, index, axis=0)
        self.var = np.delete(self.var, index, axis=0)
        self.var = np.delete(self.var, index, axis=1)

    def get_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current mean and covariance.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean vector and covariance matrix.
        """
        return self.mu, self.var
