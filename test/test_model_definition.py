from typing import List, Tuple
import unittest
import numpy as np
from base_component import BaseComponent
from baseline_component import LocalLevel, LocalTrend, LocalAcceleration
from common import forward, calc_observation
from periodic_component import Periodic
from autoregression_component import Autoregression
from lstm_component import Lstm
from model import Model


def model_definition(component: List) -> Tuple[np.ndarray, np.ndarray]:
    """Function to be tested: define model"""

    model = Model(components=component, std_observation_error=0.0)
    mu_obs_pred, var_obs_pred, _, _ = forward(
        model._mu_states,
        model._var_states,
        model._transition_matrix,
        model._process_noise_matrix,
        model._observation_matrix,
        model._observation_noise_matrix,
    )
    return mu_obs_pred, var_obs_pred


class TestModelDefinition(unittest.TestCase):
    """Test model definition"""

    def test_local_level_periodic_autoregression(self):
        """
        Test model with local_acceleration, periodic, and autoregression components
        """

        # Components
        local_level = LocalLevel(mu_states=[0.15], var_states=[0.25])
        periodic = Periodic(period=20, mu_states=[0.1, 0.2], var_states=[0.1, 0.2])
        autoregression = Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5])
        w = 2 * np.pi / periodic.period

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(w), np.sin(w), 0],
                [0, -np.sin(w), np.cos(w), 0],
                [0, 0, 0, 0.9],
            ]
        )
        process_noise_matrix_true = np.zeros([4, 4])
        observation_matrix_true = np.array([[1, 1, 0, 1]])
        mu_x = np.array([[0.15, 0.1, 0.2, 0.5]]).T
        var_x = np.array([[0.25, 0.1, 0.2, 0.5]]).T
        mu_states_true = transition_matrix_true @ mu_x
        var_states_true = (
            transition_matrix_true @ np.diagflat(var_x) @ transition_matrix_true.T
            + process_noise_matrix_true
        )
        mu_obs_true = observation_matrix_true @ mu_states_true
        var_obs_true = (
            observation_matrix_true @ var_states_true @ observation_matrix_true.T
        )

        # Model's prediction
        mu_obs_pred, var_obs_pred = model_definition(
            [local_level, periodic, autoregression]
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)

    def test_local_trend_periodic_autoregression(self):
        """
        Test model with local_acceleration, periodic, and autoregression components
        """

        # Components
        local_trend = LocalTrend(mu_states=[0.15, 0.5], var_states=[0.3, 0.25])
        periodic = Periodic(period=20, mu_states=[0.1, 0.2], var_states=[0.1, 0.2])
        autoregression = Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5])
        w = 2 * np.pi / periodic.period

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, np.cos(w), np.sin(w), 0],
                [0, 0, -np.sin(w), np.cos(w), 0],
                [0, 0, 0, 0, 0.9],
            ]
        )
        process_noise_matrix_true = np.zeros([5, 5])
        observation_matrix_true = np.array([[1, 0, 1, 0, 1]])
        mu_x = np.array([[0.15, 0.5, 0.1, 0.2, 0.5]]).T
        var_x = np.array([[0.3, 0.25, 0.1, 0.2, 0.5]]).T
        mu_states_true = transition_matrix_true @ mu_x
        var_states_true = (
            transition_matrix_true @ np.diagflat(var_x) @ transition_matrix_true.T
            + process_noise_matrix_true
        )
        mu_obs_true = observation_matrix_true @ mu_states_true
        var_obs_true = (
            observation_matrix_true @ var_states_true @ observation_matrix_true.T
        )

        # Model's prediction
        mu_obs_pred, var_obs_pred = model_definition(
            [local_trend, periodic, autoregression]
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)

    def test_local_acceleration_periodic_autoregression(self):
        """
        Test model with local_acceleration, periodic, and autoregression components
        """

        # Components
        local_acceleration = LocalAcceleration(
            mu_states=[0.1, 0.1, 0.1], var_states=[0.1, 0.2, 0.3]
        )
        periodic = Periodic(period=20, mu_states=[0.1, 0.2], var_states=[0.1, 0.2])
        autoregression = Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5])
        w = 2 * np.pi / periodic.period

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 1, 0.5, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, np.cos(w), np.sin(w), 0],
                [0, 0, 0, -np.sin(w), np.cos(w), 0],
                [0, 0, 0, 0, 0, 0.9],
            ]
        )
        process_noise_matrix_true = np.zeros([6, 6])
        observation_matrix_true = np.array([[1, 0, 0, 1, 0, 1]])
        mu_x = np.array([[0.1, 0.1, 0.1, 0.1, 0.2, 0.5]]).T
        var_x = np.array([[0.1, 0.2, 0.3, 0.1, 0.2, 0.5]]).T
        mu_states_true = transition_matrix_true @ mu_x
        var_states_true = (
            transition_matrix_true @ np.diagflat(var_x) @ transition_matrix_true.T
            + process_noise_matrix_true
        )
        mu_obs_true = observation_matrix_true @ mu_states_true
        var_obs_true = (
            observation_matrix_true @ var_states_true @ observation_matrix_true.T
        )

        # Model's prediction
        mu_obs_pred, var_obs_pred = model_definition(
            [local_acceleration, periodic, autoregression]
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)

    def test_local_level_lstm_autoregression(self):
        """
        Test model with local_acceleration, lstm, and autoregression components
        """

        # Component
        local_level = LocalLevel(mu_states=[0.6], var_states=[0.7])
        lstm = Lstm(
            look_back_len=52,
            num_features=1,
            num_layer=2,
            num_hidden_unit=50,
            mu_states=[0.6],
            var_states=[0.6],
        )
        autoregression = Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5])

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0.9],
            ]
        )
        process_noise_matrix_true = np.zeros([3, 3])
        observation_matrix_true = np.array([[1, 1, 1]])
        mu_x = np.array([[0.6, 0.6, 0.5]]).T
        var_x = np.array([[0.7, 0.6, 0.5]]).T
        mu_states_true = transition_matrix_true @ mu_x
        var_states_true = (
            transition_matrix_true @ np.diagflat(var_x) @ transition_matrix_true.T
            + process_noise_matrix_true
        )
        mu_obs_true = observation_matrix_true @ mu_states_true
        var_obs_true = (
            observation_matrix_true @ var_states_true @ observation_matrix_true.T
        )

        # Model's prediction
        mu_obs_pred, var_obs_pred = model_definition(
            [local_level, lstm, autoregression]
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)

    def test_local_trend_lstm_autoregression(self):
        """
        Test model with local_acceleration, lstm, and autoregression components
        """

        # Component
        local_trend = LocalTrend(mu_states=[0.6, 0.2], var_states=[0.7, 0.2])
        lstm = Lstm(
            look_back_len=52,
            num_features=1,
            num_layer=2,
            num_hidden_unit=50,
            mu_states=[0.6],
            var_states=[0.6],
        )
        autoregression = Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5])

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0.9],
            ]
        )
        process_noise_matrix_true = np.zeros([4, 4])
        observation_matrix_true = np.array([[1, 0, 1, 1]])
        mu_x = np.array([[0.6, 0.2, 0.6, 0.5]]).T
        var_x = np.array([[0.7, 0.2, 0.6, 0.5]]).T
        mu_states_true = transition_matrix_true @ mu_x
        var_states_true = (
            transition_matrix_true @ np.diagflat(var_x) @ transition_matrix_true.T
            + process_noise_matrix_true
        )
        mu_obs_true = observation_matrix_true @ mu_states_true
        var_obs_true = (
            observation_matrix_true @ var_states_true @ observation_matrix_true.T
        )

        # Model's prediction
        mu_obs_pred, var_obs_pred = model_definition(
            [local_trend, lstm, autoregression]
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)

    def test_local_acceleration_lstm_autoregression(self):
        """
        Test model with local_acceleration, lstm, and autoregression components
        """

        # Component
        local_acceleration = LocalAcceleration(
            mu_states=[0.1, 0.1, 0.1], var_states=[0.1, 0.2, 0.3]
        )
        lstm = Lstm(
            look_back_len=52,
            num_features=1,
            num_layer=2,
            num_hidden_unit=50,
            mu_states=[0.6],
            var_states=[0.6],
        )
        autoregression = Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5])

        # Expected results: ground true
        transition_matrix_true = np.array(
            [
                [1, 1, 0.5, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.9],
            ]
        )
        process_noise_matrix_true = np.zeros([5, 5])
        observation_matrix_true = np.array([[1, 0, 0, 1, 1]])
        mu_x = np.array([[0.1, 0.1, 0.1, 0.6, 0.5]]).T
        var_x = np.array([[0.1, 0.2, 0.3, 0.6, 0.5]]).T
        mu_states_true = transition_matrix_true @ mu_x
        var_states_true = (
            transition_matrix_true @ np.diagflat(var_x) @ transition_matrix_true.T
            + process_noise_matrix_true
        )
        mu_obs_true = observation_matrix_true @ mu_states_true
        var_obs_true = (
            observation_matrix_true @ var_states_true @ observation_matrix_true.T
        )

        # Model's prediction
        mu_obs_pred, var_obs_pred = model_definition(
            [local_acceleration, lstm, autoregression]
        )

        # Check if model's predictions match the ground true
        self.assertEqual(mu_obs_pred, mu_obs_true)
        self.assertEqual(var_obs_pred, var_obs_true)


if __name__ == "__main__":
    unittest.main()
