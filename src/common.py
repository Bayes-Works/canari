from typing import Tuple, Optional
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
        mu_states_prior[lstm_indice] = mu_lstm_pred
        var_states_prior[lstm_indice, lstm_indice] = var_lstm_pred

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
) -> Tuple[np.ndarray, np.ndarray]:

    jcb = cross_cov_states @ np.linalg.pinv(var_states_prior, rcond=1e-3)
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
    var_lstm_input = np.concatenate(
        (
            lstm_output_history.var,
            np.zeros(len(input_covariates), dtype=np.float32),
        )
    )
    return mu_lstm_input, var_lstm_input
