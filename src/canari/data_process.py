"""
Data processing for time series tasks.

This module provides the `DataProcess` class to facilitate:
- Splitting into training, validation, and test sets
- Adding time covariates (hour, day, month, etc.)
- Normalizing the dataset based on training statistics
- Generating lagged versions of features
- Injecting synthetic anomalies for testing
- Decomposing signals into trend, seasonality, and residual components
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from pytagi import Normalizer


class DataProcess:
    """
    A class for preprocessing time series data.

    Args:
        data (pd.DataFrame): Input time series data with a datetime index.
        train_start (Optional[str]): Start datetime for training data.
        train_end (Optional[str]): End datetime for training data.
        validation_start (Optional[str]): Start datetime for validation data.
        validation_end (Optional[str]): End datetime for validation data.
        test_start (Optional[str]): Start datetime for test data.
        test_end (Optional[str]): End datetime for test data.
        train_split (Optional[float]): Proportion of data to allocate to training.
        validation_split (Optional[float]): Proportion for validation data.
        test_split (Optional[float]): Proportion for test data.
        time_covariates (Optional[List[str]]): Time covariates added to dataset
        output_col (list[int]): Indices of output columns in the data.
        normalization (Optional[bool]): Whether to apply normalization.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        validation_start: Optional[str] = None,
        validation_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        train_split: Optional[float] = None,
        validation_split: Optional[float] = 0.0,
        test_split: Optional[float] = 0.0,
        time_covariates: Optional[List[str]] = None,
        output_col: list[int] = [0],
        normalization: Optional[bool] = True,
    ) -> None:
        self.train_start = train_start
        self.train_end = train_end
        self.validation_start = validation_start
        self.validation_end = validation_end
        self.test_start = test_start
        self.test_end = test_end
        self.normalization = normalization
        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split
        self.time_covariates = time_covariates
        self.output_col = output_col

        data = data.astype("float32")
        self.data = data.copy()
        self.norm_const_mean, self.norm_const_std = None, None

        self.add_time_covariates()
        self.get_split_start_end_indices()
        self.compute_normalization_constants()

        # Covariates columns
        self.covariates_col = np.ones(self.data.shape[1], dtype=bool)
        self.covariates_col[self.output_col] = False

    def add_time_covariates(self):
        """
        Add time covariates to the dataset.
        Supported covariates include:
        - hour_of_day
        - day_of_week
        - day_of_year
        - week_of_year
        - month_of_year
        - quarter_of_year
        """
        if self.time_covariates is not None:
            for time_cov in self.time_covariates:
                if time_cov == "hour_of_day":
                    self.data["hour_of_day"] = np.float32(self.data.index.hour)
                elif time_cov == "day_of_week":
                    self.data["day_of_week"] = np.float32(self.data.index.dayofweek)
                elif time_cov == "day_of_year":
                    self.data["day_of_year"] = np.float32(self.data.index.dayofyear)
                elif time_cov == "week_of_year":
                    self.data["week_of_year"] = np.array(
                        self.data.index.isocalendar().week, dtype=np.float32
                    )
                elif time_cov == "month_of_year":
                    self.data["month"] = np.float32(self.data.index.month)
                elif time_cov == "quarter_of_year":
                    self.data["quarter"] = np.float32(self.data.index.quarter)

    def get_split_start_end_indices(self):
        """
        Determine start and end indices for training, validation, and test splits.
        """
        num_data = len(self.data)
        if self.train_split is not None:
            self.test_split = 1 - self.train_split - self.validation_split
            self.train_start = 0
            self.validation_start = int(np.floor(self.train_split * num_data))
            self.test_start = self.validation_start + int(
                np.ceil(self.validation_split * num_data)
            )
            self.train_end, self.validation_end, self.test_end = (
                self.validation_start,
                self.test_start,
                num_data,
            )
        else:
            self.train_start = (
                self.data.index.get_loc(self.train_start) if self.train_start else 0
            )
            self.validation_start = self.data.index.get_loc(self.validation_start)
            self.test_start = self.data.index.get_loc(self.test_start)
            if self.train_end is None:
                self.train_end = self.validation_start
            else:
                self.train_end = self.data.index.get_loc(self.train_end)
            if self.validation_end is None:
                self.validation_end = self.test_start
            else:
                self.validation_end = self.data.index.get_loc(self.validation_end)
            if self.test_end is None:
                self.test_end = num_data
            else:
                self.test_end = self.data.index.get_loc(self.test_end)

    def compute_normalization_constants(self):
        """
        Compute normalization statistics (mean, std) based on training data.
        """
        if self.normalization:
            self.norm_const_mean, self.norm_const_std = Normalizer.compute_mean_std(
                self.data.iloc[self.train_start : self.train_end].values
            )
        else:
            self.norm_const_mean = np.zeros(self.data.shape[1])
            self.norm_const_std = np.ones(self.data.shape[1])

    def normalize_data(self) -> np.ndarray:
        """
        Normalize the data using training statistics.

        Returns:
            np.ndarray: Normalized dataset.
        """
        return (
            Normalizer.standardize(
                data=self.data.values,
                mu=self.norm_const_mean,
                std=self.norm_const_std,
            )
            if self.normalization
            else self.data.values
        )

    def get_split_indices(self) -> Tuple[np.array, np.array, np.array]:
        """
        Get the index ranges for the three data splits.

        Returns:
            Tuple[np.array, np.array, np.array]: Train, validation, and test indices.
        """
        train_index = np.arange(self.train_start, self.train_end)
        validation_index = np.arange(self.validation_start, self.validation_end)
        test_index = np.arange(self.test_start, self.test_end)
        return train_index, validation_index, test_index

    def get_splits(
        self,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Retrieve the input, output pairs (x, y) for train, validation, and test sets.

        Returns:
            Tuple of dictionaries, each with keys "x" and "y".
        """
        data = self.normalize_data()
        return (
            # Train split
            {
                "x": data[self.train_start : self.train_end, self.covariates_col],
                "y": data[self.train_start : self.train_end, self.output_col],
            },
            # Validation split
            {
                "x": data[
                    self.validation_start : self.validation_end, self.covariates_col
                ],
                "y": data[self.validation_start : self.validation_end, self.output_col],
            },
            # Test split
            {
                "x": data[self.test_start : self.test_end, self.covariates_col],
                "y": data[self.test_start : self.test_end, self.output_col],
            },
            # All data
            {
                "x": data[: self.test_end, self.covariates_col],
                "y": data[: self.test_end, self.output_col],
            },
        )

    def get_data(
        self,
        split: str,
        normalization: Optional[bool] = False,
        column: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return a specific column's values for a given data split.

        Args:
            split (str): One of ['train', 'validation', 'test', 'all'].
            normalization (bool): Whether to normalize the output.
            column (Optional[int]): Specific column index to fetch.

        Returns:
            np.ndarray: The extracted values.
        """
        if normalization:
            data = self.normalize_data()
        else:
            data = self.data.values

        if column:
            data_column = column
        else:
            data_column = self.output_col

        train_index, val_index, test_index = self.get_split_indices()
        if split == "train":
            return data[train_index, data_column]
        elif split == "validation":
            return data[val_index, data_column]
        elif split == "test":
            return data[test_index, data_column]
        elif split == "all":
            return data[:, data_column]
        else:
            raise ValueError(
                "Invalid split type. Choose from 'train', 'validation', 'test', or 'all'."
            )

    def get_time(self, split: str) -> np.ndarray:
        """
        Get datetime indices corresponding to a given split.

        Args:
            split (str): One of ['train', 'validation', 'test', 'all'].

        Returns:
            np.ndarray: Array of timestamps.
        """
        train_index, val_index, test_index = self.get_split_indices()
        if split == "train":
            return self.data.index[train_index].to_numpy()
        elif split == "validation":
            return self.data.index[val_index].to_numpy()
        elif split == "test":
            return self.data.index[test_index].to_numpy()
        elif split == "all":
            return self.data.index.to_numpy()
        else:
            raise ValueError(
                "Invalid split type. Choose from 'train', 'validation', 'test', or 'all'."
            )

    @staticmethod
    def add_lagged_columns(df, lags_per_column):
        """
        Add lagged versions of each column in the dataset, then add to the dataset as
        new columns.

        Args:
            df (pd.DataFrame): Input DataFrame with datetime index.
            lags_per_column (list[int]): Number of lags per column.

        Returns:
            pd.DataFrame: New DataFrame with lagged columns.
        """
        df_new = pd.DataFrame(index=df.index)
        for col_idx, num_lags in enumerate(lags_per_column):
            col = df.iloc[:, col_idx]
            df_new[col.name] = col
            for lag in range(1, num_lags + 1):
                df_new[f"{col.name}_lag{lag}"] = col.shift(lag).fillna(0)

        if len(lags_per_column) < df.shape[1]:
            for col_idx in range(len(lags_per_column), df.shape[1]):
                col = df.iloc[:, col_idx]
                df_new[col.name] = col

        return df_new

    @staticmethod
    def add_synthetic_anomaly(
        data: Dict[str, np.ndarray],
        num_samples: int,
        slope: List[float],
        anomaly_start: Optional[float] = 0.33,
        anomaly_end: Optional[float] = 0.66,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Add synthetic anomalies to a time series dataset.

        Args:
            data (dict): Original data dict with "x" and "y".
            num_samples (int): Number of anomaly to generate.
            slope (list[float]): Slope for anomaly.
            anomaly_start (float): Start of anomaly window (0–1).
            anomaly_end (float): End of anomaly window (0–1).

        Returns:
            list: Data dicts with anomalies injected.
        """
        _data_with_anomaly = []
        len_data = len(data["y"])
        window_anomaly_start = int(np.ceil(len_data * anomaly_start))
        window_anomaly_end = int(np.ceil(len_data * anomaly_end))
        anomaly_start_history = np.random.randint(
            window_anomaly_start, window_anomaly_end, size=num_samples * len(slope)
        )

        for j, _slope in enumerate(slope):
            for i in range(num_samples):
                trend = np.zeros(len_data)
                change_point = anomaly_start_history[i + j * num_samples]
                trend_end_value = _slope * (len_data - change_point - 1)
                trend[change_point:] = np.linspace(
                    0, trend_end_value, len_data - change_point
                )
                _data_with_anomaly.append(data["y"].flatten() + trend)

        return [
            {
                "x": data["x"],
                "y": ts.reshape(-1, 1),
                "anomaly_timestep": timestep,
            }
            for ts, timestep in zip(_data_with_anomaly, anomaly_start_history)
        ]

    @staticmethod
    def decompose_data(data):
        """
        Decompose a signal into trend, seasonality, and residual using Fourier transform.

        Args:
            data (array-like): 1D numeric time series.

        Returns:
            tuple: (trend, slope, seasonality, residual)
        """

        def handle_missing_data(data):
            if np.isnan(data).any():
                data = (
                    pd.Series(data)
                    .interpolate(method="linear", limit_direction="both")
                    .to_numpy()
                )
            return data

        def estimate_window(data):
            fft_spectrum = np.fft.fft(data - np.mean(data))
            frequencies = np.fft.fftfreq(len(data))
            power = np.abs(fft_spectrum)
            peak_frequency = frequencies[np.argmax(power[1:]) + 1]
            period = int(1 / peak_frequency) if peak_frequency > 0 else len(data) // 10
            return max(3, period)

        def estimate_seasonality(data, window):
            period = window
            seasonality = np.zeros_like(data)
            for i in range(period):
                seasonality[i::period] = np.mean(data[i::period])
            return seasonality - np.mean(seasonality)

        def estimate_trend(deseasonalized_data):
            x = np.arange(len(deseasonalized_data))
            slope, intercept = np.polyfit(x, deseasonalized_data, 1)
            return slope * x + intercept, slope

        data = handle_missing_data(np.array(data))
        window = estimate_window(data)
        seasonality = estimate_seasonality(data, window)
        deseasonalized_data = data - seasonality
        trend, slope = estimate_trend(deseasonalized_data)
        residual = data - trend - seasonality
        return trend, slope, seasonality, residual
