import numpy as np
import copy
from typing import Tuple, List, Dict, Optional
from src.model import Model
import src.common as common
from src.data_struct import SmootherStates


class SKF:
    """ "
    Switching Kalman Filter
    """

    def __init__(
        self,
        normal_model: Model,
        abnormal_model: Model,
        std_transition_error: Optional[float] = 0.0,
    ):
        self.norm_model = normal_model
        self.abnorm_model = abnormal_model
        self.std_transition_error = std_transition_error

    @staticmethod
    def pad_matrix(
        x: np.ndarray,
        idx,
        pad_row: Optional[np.ndarray] = None,
        pad_col: Optional[np.ndarray] = None,
    ):
        """
        Add padding for states
        """

        if pad_row is not None:
            x = np.insert(x, idx, pad_row, axis=0)
        if pad_col is not None:
            x = np.insert(x, idx, pad_col, axis=1)
        return x

    @staticmethod
    def create_compatible_model(source: Model, target: Model) -> Model:
        """ "
        Create compatiable model by padding states
        """
        # TODO: should not be like this, this class
        source.components = copy.copy(target.components)
        pad_row = np.zeros((source.num_states)).flatten()
        pad_col = np.zeros((target.num_states)).flatten()
        for i, state in enumerate(target.states_name):
            if state not in source.states_name:
                source.mu_states = SKF.pad_matrix(
                    source.mu_states, i, pad_row=np.zeros(1)
                )
                source.var_states = SKF.pad_matrix(
                    source.var_states, i, pad_row, pad_col
                )
                source.transition_matrix = SKF.pad_matrix(
                    source.transition_matrix, i, pad_row, pad_col
                )
                source.process_noise_matrix = SKF.pad_matrix(
                    source.process_noise_matrix, i, pad_row, pad_col
                )
                source.observation_matrix = SKF.pad_matrix(
                    source.observation_matrix, i, pad_col=np.zeros(1)
                )
                source.num_states = copy.copy(target.num_states)
                source.states_name = copy.copy(target.states_name)
                source.lstm_states_index = source.states_name.index("lstm")
                source.index_pad_states = i
        return source

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, SmootherStates]:
        """
        Train the LstmNetwork of the normal model
        """
        return self.norm_model.lstm_train(train_data, validation_data)

    def filter(
        self,
        data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filtering
        """

        # Initialize sub-models for SKF
        self.norm_model = SKF.create_compatible_model(
            source=self.norm_model, target=self.abnorm_model
        )
        self.abnorm_model.set_states(
            self.norm_model.mu_states, self.norm_model.var_states
        )

        self.norm_to_abnorm_model = self.abnorm_model.duplicate_model()
        self.abnorm_to_norm_model = self.norm_model.duplicate_model()

        index_pad_states = self.norm_model.index_pad_states
        self.norm_to_abnorm_model.process_noise_matrix[
            index_pad_states, index_pad_states
        ] = (self.std_transition_error**2)

        #
        mu_obs_preds = []
        var_obs_preds = []
        mu_lstm_pred = None
        var_lstm_pred = None

        # for time_step, (x, y) in enumerate(zip(data["x"], data["y"])):
        #     if self.norm_model.lstm_net:
        #         mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
        #             self.norm_model.lstm_output_history, x
        #         )
        #         mu_lstm_pred, var_lstm_pred = self.norm_model.lstm_net.forward(
        #             mu_x=mu_lstm_input, var_x=var_lstm_input
        #         )

        #         mu_obs_pred_1, var_obs_pred_1 = self.norm_model.forward(
        #             mu_lstm_pred, var_lstm_pred
        #         )
        #         mu_obs_pred_12, var_obs_pred_12 = self.norm_to_abnorm_model.forward(
        #             mu_lstm_pred, var_lstm_pred
        #         )
        #         mu_obs_pred_2, var_obs_pred_2 = self.abnorm_model.forward(
        #             mu_lstm_pred, var_lstm_pred
        #         )

        #         mu_obs_pred_21, var_obs_pred_21 = self.abnorm_to_norm_model.forward(
        #             mu_lstm_pred, var_lstm_pred
        #         )

        #     check = 1
