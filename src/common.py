import numpy as np


def block_diag(*arrays: np.ndarray) -> np.ndarray:
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


def forward(
    mu_states_posterior: np.ndarray,
    var_states_posterior: np.ndarray,
    transition_matrix: np.ndarray,
    process_noise_matrix: np.ndarray,
):
    mu_states_prior = transition_matrix @ mu_states_posterior
    var_states_prior = (
        transition_matrix @ np.diagflat(var_states_posterior) @ transition_matrix.T
        + process_noise_matrix
    )
    return mu_states_prior, var_states_prior


def cal_obsevation(
    mu_states: np.ndarray,
    var_states: np.ndarray,
    observation_matrix: np.ndarray,
    observation_noise_matrix: np.ndarray,
):
    mu_obs_predicted = observation_matrix @ mu_states
    var_obs_predicted = (
        observation_matrix @ var_states @ observation_matrix.T
        + observation_noise_matrix
    )
    return mu_obs_predicted, var_obs_predicted


def backward(
    mu_states_prior: np.ndarray,
    var_states_prior: np.ndarray,
    observation_matrix: np.ndarray,
    observation_noise_matrix: np.ndarray,
    obs: float,
):
    mu_obs_predicted, var_obs_predicted = cal_obsevation(
        mu_states_prior, var_states_prior, observation_matrix, observation_noise_matrix
    )
    cov_states_time = observation_matrix @ var_states_prior
    delta_mu_states = cov_states_time.T / var_obs_predicted @ (obs - mu_obs_predicted)
    delta_var_states = -cov_states_time.T / var_obs_predicted @ cov_states_time
    return delta_mu_states, delta_var_states


def rts_smoother(
    mu_states_smooth: np.ndarray,
    var_states_smooth: np.ndarray,
    mu_states_prior: np.ndarray,
    var_states_prior: np.ndarray,
    cros_cov: np.ndarray,
):
    jcb = cros_cov @ np.linalg.pinv(var_states_prior)
    delta_mu_states = jcb @ (mu_states_smooth - mu_states_prior)
    delta_var_states = jcb @ (var_states_smooth - var_states_prior) @ jcb.T
    return delta_mu_states, delta_var_states
