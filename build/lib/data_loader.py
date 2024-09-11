from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd

from pytagi import Normalizer, Utils


class TimeSeriesDataloader:
    """Data loader for time series"""

    def __init__(
        self,
        value: np.ndarray,
        time: np.ndarray,
        output_col: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
        num_features: int,
        stride: int,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        ts_idx: Optional[int] = None,
        time_covariates: Optional[str] = None,
        keep_last_time_cov: Optional[bool] = False,
    ) -> None:
        self.value = value
        self.time = time
        self.output_col = output_col
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.num_features = num_features
        self.stride = stride
        self.x_mean = x_mean
        self.x_std = x_std
        self.ts_idx = ts_idx  # add time series index when data having multiple ts
        self.time_covariates = time_covariates  # for adding time covariates
        self.keep_last_time_cov = keep_last_time_cov
        self.dataset = self.process_data()

    @staticmethod
    def batch_generator(
        input_data: np.ndarray,
        output_data: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
        """
        Generator function to yield batches of data.
        """
        num_data = input_data.shape[0]
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, num_data, batch_size):
            # if start_idx + batch_size > num_data:
            #     continue
            end_idx = min(start_idx + batch_size, num_data)
            idx = indices[start_idx:end_idx]
            yield input_data[idx].flatten(), output_data[idx].flatten()

    def process_data(self) -> dict:
        """Process time series"""
        # Initialization
        utils = Utils()

        x = self.value
        date_time = self.time
        
        # Add time covariates
        if self.time_covariates is not None:
            date_time = np.array(date_time, dtype="datetime64")
            for time_cov in self.time_covariates:
                if time_cov == "hour_of_day":
                    hour_of_day = date_time.astype("datetime64[h]").astype(int) % 24
                    x = np.concatenate((x, hour_of_day), axis=1)
                elif time_cov == "day_of_week":
                    day_of_week = date_time.astype("datetime64[D]").astype(int) % 7
                    x = np.concatenate((x, day_of_week), axis=1)
                elif time_cov == "week_of_year":
                    week_of_year = (
                        date_time.astype("datetime64[W]").astype(int) % 52 + 1
                    )
                    x = np.concatenate((x, week_of_year), axis=1)
                elif time_cov == "month_of_year":
                    month_of_year = (
                        date_time.astype("datetime64[M]").astype(int) % 12 + 1
                    )
                    x = np.concatenate((x, month_of_year), axis=1)
                elif time_cov == "quarter_of_year":
                    month_of_year = (
                        date_time.astype("datetime64[M]").astype(int) % 12 + 1
                    )
                    quarter_of_year = (month_of_year - 1) // 3 + 1
                    x = np.concatenate((x, quarter_of_year), axis=1)

        # Normalizer
        if self.x_mean is None and self.x_std is None:
            self.x_mean, self.x_std = Normalizer.compute_mean_std(x)
        x = Normalizer.standardize(data=x, mu=self.x_mean, std=self.x_std)

        # Create rolling windows
        x_rolled, y_rolled = utils.create_rolling_window(
            data=x,
            output_col=self.output_col,
            input_seq_len=self.input_seq_len,
            output_seq_len=self.output_seq_len,
            num_features=self.num_features,
            stride=self.stride,
        )

        if self.keep_last_time_cov:
            x_rolled = self.remove_time_cov(x_rolled)

        # Dataloader
        dataset = {}
        dataset["value"] = (x_rolled, y_rolled)

        # NOTE: Datetime is saved for the visualization purpose
        dataset["date_time"] = [np.datetime64(date) for date in np.squeeze(date_time)]

        return dataset

    def remove_time_cov(self, x):
        x_new = np.zeros((len(x), self.input_seq_len + self.num_features - 1), dtype=np.float32)
        for i in range(0,len(x)):
            x_ = x[i]
            keep_idx = np.arange(0, len(x_), self.num_features)
            x_new[i] = np.concatenate((x_[keep_idx],x_[-self.num_features+1:]))
        return x_new


    def create_data_loader(self, batch_size: int, shuffle: bool = True):
        return self.batch_generator(*self.dataset["value"], batch_size, shuffle)
