import unittest
import numpy as np
from base_component import BaseComponent
from baseline_component import LocalLevel, LocalTrend, LocalAcceleration
from periodic_component import Periodic
from residual_component import Autoregression
from lstm_component import Lstm
from lstm_ssm_model import LstmSsm
import common


def lstm_ssm_model_construction(*component: BaseComponent):
    model = LstmSsm(*component)
    mu_states, var_states = common.forward(
        model._mu_states,
        model._var_states,
        model._transition_matrix,
        model._process_noise_matrix,
    )
    mu_obs, var_obs = common.cal_obsevation(
        mu_states,
        var_states,
        model._observation_matrix,
        observation_noise_matrix=np.array([[0.0]]),
    )
    return mu_obs, var_obs


class TestLstmSSmModelConstruction(unittest.TestCase):
    """Test LSTM/SSSM model construction"""

    def test_baseline_periodic_autoregression(self):
        local_acceleration = LocalAcceleration(
            mu_states=[0.1, 0.1, 0.1], var_states=[0.1, 0.2, 0.3]
        )
        periodic = Periodic(period=20, mu_states=[0.1, 0.2], var_states=[0.1, 0.2])
        autoregression = Autoregression(phi=0.9, mu_states=[0.5], var_states=[0.5])
        w = 2 * np.pi / periodic.period
        mu_obs, var_obs = lstm_ssm_model_construction(
            local_acceleration, periodic, autoregression
        )

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
        process_nois_matrix_true = np.zeros([6, 6])
        observation_matrix_true = np.array([[1, 0, 0, 1, 0, 1]])
        mu_x = np.array([[0.1, 0.1, 0.1, 0.1, 0.2, 0.5]]).T
        var_x = np.array([[0.1, 0.2, 0.3, 0.1, 0.2, 0.5]]).T
        mu_states_true = transition_matrix_true @ mu_x
        var_states_true = (
            transition_matrix_true @ np.diagflat(var_x) @ transition_matrix_true.T
            + process_nois_matrix_true
        )
        mu_obs_true = observation_matrix_true @ mu_states_true
        var_obs_true = (
            observation_matrix_true @ var_states_true @ observation_matrix_true.T
        )

        self.assertEqual(mu_obs, mu_obs_true)
        self.assertEqual(var_obs, var_obs_true)

    def test_baseline_lstm_autoregression(self):
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
        mu_obs, var_obs = lstm_ssm_model_construction(
            local_acceleration, lstm, autoregression
        )

        transition_matrix_true = np.array(
            [
                [1, 1, 0.5, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.9],
            ]
        )
        process_nois_matrix_true = np.zeros([5, 5])
        observation_matrix_true = np.array([[1, 0, 0, 1, 1]])
        mu_x = np.array([[0.1, 0.1, 0.1, 0.6, 0.5]]).T
        var_x = np.array([[0.1, 0.2, 0.3, 0.6, 0.5]]).T
        mu_states_true = transition_matrix_true @ mu_x
        var_states_true = (
            transition_matrix_true @ np.diagflat(var_x) @ transition_matrix_true.T
            + process_nois_matrix_true
        )
        mu_obs_true = observation_matrix_true @ mu_states_true
        var_obs_true = (
            observation_matrix_true @ var_states_true @ observation_matrix_true.T
        )

        self.assertEqual(mu_obs, mu_obs_true)
        self.assertEqual(var_obs, var_obs_true)


if __name__ == "__main__":
    unittest.main()
