from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Base model
    """

    def __init__(self):
        self._device = "cpu"
        self._num_threads = 1
        self._seeds = 1

    def __call__(self, time_series_data: np.ndarray):
        return self.detect_anomaly(time_series_data)

    @property
    def device(self) -> str:
        """Get the device"""
        return self._device

    @device.setter
    def device(self, value: str):
        """Set the sevice"""
        self._device = value

    @property
    def num_threads(self) -> int:
        """Get number of threads used"""
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value: int):
        """Set the number of threads"""
        self._num_threads = value

    def set_seeds(self, seeds: int):
        """Set the seeds"""
        self._seeds = seeds

    @abstractmethod
    def detect_anomaly(self, time_series_data: np.ndarray):
        """Detect anomaly"""
        pass

    @abstractmethod
    def save(self, filename: str):
        """Save the model to a file."""
        pass

    @abstractmethod
    def load(self, filename: str):
        """Load the model from a file."""
        pass
