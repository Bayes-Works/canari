import numpy as np
from typing import Tuple, List, Dict
from src.model import Model
import src.common
from src.data_struct import SmootherStates


class SKF:
    """ "
    Switching Kalman Filter
    """

    def __init__(
        self,
        normal_model: Model,
        abnormal_model: Model,
    ):
        self.normal_model = normal_model
        self.abnormal_model = abnormal_model

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, SmootherStates]:
        """
        Train the LstmNetwork of the normal model
        """
        return self.normal_model.lstm_train(train_data, validation_data)
