from typing import Tuple
import unittest
import numpy as np
import numpy.testing as npt
from src import (
    BaseComponent,
    LocalAcceleration,
    LocalLevel,
    LocalTrend,
    Periodic,
    Autoregression,
    LstmNetwork,
    WhiteNoise,
    Model,
    forward,
    backward,
)


def predict(
    transition_matrix: np.ndarray,
    process_noise_matrix: np.ndarray,
    observation_matrix: np.ndarray,
    mu_states: np.ndarray,
    var_states: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu_states_true = transition_matrix @ mu_states
    var_states_true = (
        transition_matrix @ np.diagflat(var_states) @ transition_matrix.T
        + process_noise_matrix
    )
    mu_obs_true = observation_matrix @ mu_states_true
    var_obs_true = observation_matrix @ var_states_true @ observation_matrix.T

    obs = 0.5
    cov_obs_states = observation_matrix @ var_states_true
    delta_mu_states_true = cov_obs_states.T / var_obs_true @ (obs - mu_obs_true)
    delta_var_states_true = -cov_obs_states.T / var_obs_true @ cov_obs_states
    return mu_obs_true, var_obs_true, delta_mu_states_true, delta_var_states_true


def model_prediction(
    *components: BaseComponent,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Model]:
    """Function to be tested: model prediction"""

    model = Model(*components)
    mu_obs_pred, var_obs_pred, _, var_states_prior = forward(
        model.mu_states,
        model.var_states,
        model.transition_matrix,
        model.process_noise_matrix,
        model.observation_matrix,
    )
    obs = 0.5
    delta_mu_states_pred, delta_var_states_pred = backward(
        obs,
        mu_obs_pred,
        var_obs_pred,
        var_states_prior,
        model.observation_matrix,
    )
    return mu_obs_pred, var_obs_pred, delta_mu_states_pred, delta_var_states_pred, model


class TestModelPrediction(unittest.TestCase):
    """Test model prediction"""

    def test_local_level_periodic_autoregression(self):
        """
        Test model with local_acceleration, periodic, and autoregression components
        """

        period = 20
        w = 2 * np.pi / period

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, np.cos(w), np.sin(w), 0, 0],
                [0, -np.sin(w), np.cos(w), 0, 0],
                [0, 0, 0, 0.9, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        std_observation_noise = 0.1
        process_noise_matrix_true = np.zeros([5, 5])
        process_noise_matrix_true[4, 4] = std_observation_noise**2
        observation_matrix_true = np.array([[1, 1, 0, 1, 1]])
        mu_states_true = np.array([[0.15, 0.1, 0.2, 0.5, 0]]).T
        var_states_true = np.array([[0.25, 0.1, 0.2, 0.5, 0]]).T

        mu_obs_true, var_obs_true, delta_mu_states_true, delta_var_states_true = (
            predict(
                transition_matrix_true,
                process_noise_matrix_true,
                observation_matrix_true,
                mu_states_true,
                var_states_true,
            )
        )

        # Model's prediction
        (
            mu_obs_pred,
            var_obs_pred,
            delta_mu_states_pred,
            delta_var_states_pred,
            model,
        ) = model_prediction(
            LocalLevel(mu_states=[0.15], var_states=[0.25]),
            Periodic(period=period, mu_states=[0.1, 0.2], var_states=[0.1, 0.2]),
            Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5]),
            WhiteNoise(std_error=0.1),
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)
        npt.assert_allclose(model.transition_matrix, transition_matrix_true)
        npt.assert_allclose(model.process_noise_matrix, process_noise_matrix_true)
        npt.assert_allclose(model.observation_matrix, observation_matrix_true)
        npt.assert_allclose(model.mu_states, mu_states_true)
        npt.assert_allclose(model.var_states, var_states_true)
        npt.assert_allclose(delta_mu_states_true, delta_mu_states_pred)
        npt.assert_allclose(delta_var_states_true, delta_var_states_pred)

    def test_local_trend_periodic_autoregression(self):
        """
        Test model with local_acceleration, periodic, and autoregression components
        """

        period = 20
        w = 2 * np.pi / period

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, np.cos(w), np.sin(w), 0, 0],
                [0, 0, -np.sin(w), np.cos(w), 0, 0],
                [0, 0, 0, 0, 0.9, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        std_observation_noise = 0.1
        process_noise_matrix_true = np.zeros([6, 6])
        process_noise_matrix_true[5, 5] = std_observation_noise**2
        observation_matrix_true = np.array([[1, 0, 1, 0, 1, 1]])
        mu_states_true = np.array([[0.15, 0.5, 0.1, 0.2, 0.5, 0]]).T
        var_states_true = np.array([[0.3, 0.25, 0.1, 0.2, 0.5, 0]]).T

        mu_obs_true, var_obs_true, delta_mu_states_true, delta_var_states_true = (
            predict(
                transition_matrix_true,
                process_noise_matrix_true,
                observation_matrix_true,
                mu_states_true,
                var_states_true,
            )
        )

        # Model's prediction
        (
            mu_obs_pred,
            var_obs_pred,
            delta_mu_states_pred,
            delta_var_states_pred,
            model,
        ) = model_prediction(
            LocalTrend(mu_states=[0.15, 0.5], var_states=[0.3, 0.25]),
            Periodic(period=20, mu_states=[0.1, 0.2], var_states=[0.1, 0.2]),
            Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5]),
            WhiteNoise(std_error=0.1),
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)
        npt.assert_allclose(model.transition_matrix, transition_matrix_true)
        npt.assert_allclose(model.process_noise_matrix, process_noise_matrix_true)
        npt.assert_allclose(model.observation_matrix, observation_matrix_true)
        npt.assert_allclose(model.mu_states, mu_states_true)
        npt.assert_allclose(model.var_states, var_states_true)
        npt.assert_allclose(delta_mu_states_true, delta_mu_states_pred)
        npt.assert_allclose(delta_var_states_true, delta_var_states_pred)

    def test_local_acceleration_periodic_autoregression(self):
        """
        Test model with local_acceleration, periodic, and autoregression components
        """

        period = 20
        w = 2 * np.pi / period

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 1, 0.5, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, np.cos(w), np.sin(w), 0, 0],
                [0, 0, 0, -np.sin(w), np.cos(w), 0, 0],
                [0, 0, 0, 0, 0, 0.9, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        std_observation_noise = 0.1
        process_noise_matrix_true = np.zeros([7, 7])
        process_noise_matrix_true[6, 6] = std_observation_noise**2
        observation_matrix_true = np.array([[1, 0, 0, 1, 0, 1, 1]])
        mu_states_true = np.array([[0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0]]).T
        var_states_true = np.array([[0.1, 0.2, 0.3, 0.1, 0.2, 0.5, 0]]).T

        mu_obs_true, var_obs_true, delta_mu_states_true, delta_var_states_true = (
            predict(
                transition_matrix_true,
                process_noise_matrix_true,
                observation_matrix_true,
                mu_states_true,
                var_states_true,
            )
        )

        # Model's prediction
        (
            mu_obs_pred,
            var_obs_pred,
            delta_mu_states_pred,
            delta_var_states_pred,
            model,
        ) = model_prediction(
            LocalAcceleration(mu_states=[0.1, 0.1, 0.1], var_states=[0.1, 0.2, 0.3]),
            Periodic(period=20, mu_states=[0.1, 0.2], var_states=[0.1, 0.2]),
            Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5]),
            WhiteNoise(std_error=0.1),
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)
        npt.assert_allclose(model.transition_matrix, transition_matrix_true)
        npt.assert_allclose(model.process_noise_matrix, process_noise_matrix_true)
        npt.assert_allclose(model.observation_matrix, observation_matrix_true)
        npt.assert_allclose(model.mu_states, mu_states_true)
        npt.assert_allclose(model.var_states, var_states_true)
        npt.assert_allclose(delta_mu_states_true, delta_mu_states_pred)
        npt.assert_allclose(delta_var_states_true, delta_var_states_pred)

    def test_local_level_lstm_autoregression(self):
        """
        Test model with local_acceleration, lstm, and autoregression components
        """

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0.9, 0],
                [0, 0, 0, 0],
            ]
        )
        std_observation_noise = 0.1
        process_noise_matrix_true = np.zeros([4, 4])
        process_noise_matrix_true[3, 3] = std_observation_noise**2
        observation_matrix_true = np.array([[1, 1, 1, 1]])
        mu_states_true = np.array([[0.6, 0.6, 0.5, 0]]).T
        var_states_true = np.array([[0.7, 0.6, 0.5, 0]]).T

        mu_obs_true, var_obs_true, delta_mu_states_true, delta_var_states_true = (
            predict(
                transition_matrix_true,
                process_noise_matrix_true,
                observation_matrix_true,
                mu_states_true,
                var_states_true,
            )
        )

        # Model's prediction
        (
            mu_obs_pred,
            var_obs_pred,
            delta_mu_states_pred,
            delta_var_states_pred,
            model,
        ) = model_prediction(
            LocalLevel(mu_states=[0.6], var_states=[0.7]),
            LstmNetwork(
                look_back_len=52,
                num_features=1,
                num_layer=2,
                num_hidden_unit=50,
                mu_states=[0.6],
                var_states=[0.6],
            ),
            Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5]),
            WhiteNoise(std_error=0.1),
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)
        npt.assert_allclose(model.transition_matrix, transition_matrix_true)
        npt.assert_allclose(model.process_noise_matrix, process_noise_matrix_true)
        npt.assert_allclose(model.observation_matrix, observation_matrix_true)
        npt.assert_allclose(model.mu_states, mu_states_true)
        npt.assert_allclose(model.var_states, var_states_true)
        npt.assert_allclose(delta_mu_states_true, delta_mu_states_pred)
        npt.assert_allclose(delta_var_states_true, delta_var_states_pred)

    def test_local_trend_lstm_autoregression(self):
        """
        Test model with local_acceleration, lstm, and autoregression components
        """

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0.9, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        std_observation_noise = 0.1
        process_noise_matrix_true = np.zeros([5, 5])
        process_noise_matrix_true[4, 4] = std_observation_noise**2
        observation_matrix_true = np.array([[1, 0, 1, 1, 1]])
        mu_states_true = np.array([[0.6, 0.2, 0.6, 0.5, 0]]).T
        var_states_true = np.array([[0.7, 0.2, 0.6, 0.5, 0]]).T

        mu_obs_true, var_obs_true, delta_mu_states_true, delta_var_states_true = (
            predict(
                transition_matrix_true,
                process_noise_matrix_true,
                observation_matrix_true,
                mu_states_true,
                var_states_true,
            )
        )

        # Model's prediction
        (
            mu_obs_pred,
            var_obs_pred,
            delta_mu_states_pred,
            delta_var_states_pred,
            model,
        ) = model_prediction(
            LocalTrend(mu_states=[0.6, 0.2], var_states=[0.7, 0.2]),
            LstmNetwork(
                look_back_len=52,
                num_features=1,
                num_layer=2,
                num_hidden_unit=50,
                mu_states=[0.6],
                var_states=[0.6],
            ),
            Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5]),
            WhiteNoise(std_error=0.1),
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)
        npt.assert_allclose(model.transition_matrix, transition_matrix_true)
        npt.assert_allclose(model.process_noise_matrix, process_noise_matrix_true)
        npt.assert_allclose(model.observation_matrix, observation_matrix_true)
        npt.assert_allclose(model.mu_states, mu_states_true)
        npt.assert_allclose(model.var_states, var_states_true)
        npt.assert_allclose(delta_mu_states_true, delta_mu_states_pred)
        npt.assert_allclose(delta_var_states_true, delta_var_states_pred)

    def test_local_acceleration_lstm_autoregression(self):
        """
        Test model with local_acceleration, lstm, and autoregression components
        """

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 1, 0.5, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.9, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        std_observation_noise = 0.1
        process_noise_matrix_true = np.zeros([6, 6])
        process_noise_matrix_true[5, 5] = std_observation_noise**2
        observation_matrix_true = np.array([[1, 0, 0, 1, 1, 1]])
        mu_states_true = np.array([[0.1, 0.1, 0.1, 0.6, 0.5, 0]]).T
        var_states_true = np.array([[0.1, 0.2, 0.3, 0.6, 0.5, 0]]).T

        mu_obs_true, var_obs_true, delta_mu_states_true, delta_var_states_true = (
            predict(
                transition_matrix_true,
                process_noise_matrix_true,
                observation_matrix_true,
                mu_states_true,
                var_states_true,
            )
        )

        # Model's prediction
        (
            mu_obs_pred,
            var_obs_pred,
            delta_mu_states_pred,
            delta_var_states_pred,
            model,
        ) = model_prediction(
            LocalAcceleration(mu_states=[0.1, 0.1, 0.1], var_states=[0.1, 0.2, 0.3]),
            LstmNetwork(
                look_back_len=52,
                num_features=1,
                num_layer=2,
                num_hidden_unit=50,
                mu_states=[0.6],
                var_states=[0.6],
            ),
            Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5]),
            WhiteNoise(std_error=0.1),
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)
        npt.assert_allclose(model.transition_matrix, transition_matrix_true)
        npt.assert_allclose(model.process_noise_matrix, process_noise_matrix_true)
        npt.assert_allclose(model.observation_matrix, observation_matrix_true)
        npt.assert_allclose(model.mu_states, mu_states_true)
        npt.assert_allclose(model.var_states, var_states_true)
        npt.assert_allclose(delta_mu_states_true, delta_mu_states_pred)
        npt.assert_allclose(delta_var_states_true, delta_var_states_pred)


if __name__ == "__main__":
    unittest.main()
