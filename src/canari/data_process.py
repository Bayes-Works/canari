"""
Data processing tools for time series.
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from pytagi import Normalizer


class DataProcess:
    """
    This module provides the `DataProcess` class to facilitate:

    - Standardization of datasets based on training statistics
    - Splitting data into training, validation, and test sets
    - Adding time covariates (hour, day, month, etc.) to data
    - Generating lagged versions of features
    - Adding synthetic anomalies to data

    Args:
        data (pd.DataFrame): Input DataFrame with a datetime or numeric index.
        train_start (Optional[str]): Start index for training set.
        train_end (Optional[str]): End index for training set.
        validation_start (Optional[str]): Start index for validation set.
        validation_end (Optional[str]): End index for validation set.
        test_start (Optional[str]): Start index for test set.
        test_end (Optional[str]): End index for test set.
        train_split (Optional[float]): Fraction of data for training set.
        validation_split (Optional[float]): Fraction for validation set.
        test_split (Optional[float]): Fraction for test set.
        time_covariates (Optional[List[str]]): Time covariates added to dataset
        output_col (list[int]): Column's indice for target variable.
        standardization (Optional[bool]): Whether to apply data standardization
                                        (zero mean, unit standard deviation).

    Attributes:
        data (pd.DataFrame): The full dataset, including any added time covariates.
        train_start (Optional[str]): Start index for training set.
        train_end (Optional[str]): End index for training set.
        validation_start (Optional[str]): Start index for validation set.
        validation_end (Optional[str]): End index for validation set.
        test_start (Optional[str]): Start index for test set.
        test_end (Optional[str]): End index for test set.
        standardization (Optional[bool]): Whether standardization is applied. Defaults to True.
        train_split (Optional[float]): Fraction of data used for training.
        validation_split (Optional[float]): Fraction of data used for validation.
        test_split (Optional[float]): Fraction of data used for testing.
        time_covariates (Optional[List[str]]): Time covariates to add (e.g., "hour_of_day").
        output_col (list[int]): Indices of columns used as output/target variables.
        scale_const_mean (Optional[list]): Mean values used for standardization.
        scale_const_std (Optional[list]): Std dev values used for standardization.

    Examples:
        >>> import pandas as pd
        >>> from canari import DataProcess
        >>> dt_index = pd.date_range(start="2025-01-01", periods=11, freq="H")
        >>> data = pd.DataFrame({'value': np.linspace(0.1, 1.0, 11)},
                                index=dt_index)
        >>> dp = DataProcess(data,
                            train_split=0.7,
                            validation_split=0.2,
                            test_split=0.1,
                            time_covariates = ["hour_of_day"],
                            standardization=True,)
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
        standardization: Optional[bool] = True,
    ) -> None:
        self.train_start = train_start
        self.train_end = train_end
        self.validation_start = validation_start
        self.validation_end = validation_end
        self.test_start = test_start
        self.test_end = test_end
        self.standardization = standardization
        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split
        self.time_covariates = time_covariates
        self.output_col = output_col

        self.data = data.copy()
        self.scale_const_mean, self.scale_const_std = None, None

        self._add_time_covariates()
        self._get_split_start_end_indices()
        self._compute_standardization_constants()

        # Covariates columns
        self.covariates_col = np.ones(self.data.shape[1], dtype=bool)
        self.covariates_col[self.output_col] = False

    def _add_time_covariates(self):
        """Add time covariates to the data"""

        if self.time_covariates is None:
            return

        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("add_time_covariates requires a pd.DatetimeIndex")

        allowed = {
            "hour_of_day": lambda idx: idx.hour,
            "day_of_week": lambda idx: idx.dayofweek,
            "day_of_year": lambda idx: idx.dayofyear,
            "week_of_year": lambda idx: idx.isocalendar().week.astype(float),
            "month_of_year": lambda idx: idx.month,
            "quarter_of_year": lambda idx: idx.quarter,
        }
        extras = set(self.time_covariates) - set(allowed)
        if extras:
            raise ValueError(f"Unknown time covariates: {extras}")

        for cov in dict.fromkeys(self.time_covariates):
            vals = allowed[cov](self.data.index)
            self.data[cov] = np.array(vals)

    def _get_split_start_end_indices(self):
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

    def _compute_standardization_constants(self):
        """
        Compute standardization statistics (mean, std) based on training data.
        """
        if self.standardization:
            self.scale_const_mean, self.scale_const_std = Normalizer.compute_mean_std(
                self.data.iloc[self.train_start : self.train_end].values
            )
        else:
            self.scale_const_mean = np.zeros(self.data.shape[1])
            self.scale_const_std = np.ones(self.data.shape[1])

    def standardize_data(self) -> np.ndarray:
        """
        Standardize the data using training statistics.

        Returns:
            np.ndarray: Standardized dataset.
        """
        return (
            Normalizer.standardize(
                data=self.data.values,
                mu=self.scale_const_mean,
                std=self.scale_const_std,
            )
            if self.standardization
            else self.data.values
        )

    def get_split_indices(self) -> Tuple[np.array, np.array, np.array]:
        """
        Get the index ranges for the train, validation, and test splits.

        Returns:
            Tuple[np.array, np.array, np.array]: Train, validation, and test indices.

        Examples:
            >>> train_index, val_index, test_index = dp.get_split_indices()
        """
        train_index = np.arange(self.train_start, self.train_end)
        validation_index = np.arange(self.validation_start, self.validation_end)
        test_index = np.arange(self.test_start, self.test_end)
        return train_index, validation_index, test_index

    def get_splits(
        self,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
    ]:
        """Return training, validation, test splits, and all data

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
                A tuple containing a dictionary for each split.
        """

        data = self.standardize_data()
        if isinstance(self.data.index, pd.DatetimeIndex):
            freq = pd.infer_freq(self.data.index)
        else:
            freq = self.data.index[1] - self.data.index[0]

        return (
            # Train split
            {
                "x": data[self.train_start : self.train_end, self.covariates_col],
                "y": data[self.train_start : self.train_end, self.output_col],
                "covariates_col": self.covariates_col,
                "data_col_names": self.data.columns.tolist(),
                "start_date": self.data.index[self.train_start],
                "cov_names": self.data.columns[self.covariates_col].tolist(),
                "time_covariates": self.time_covariates,
                "scale_const_mean": self.scale_const_mean,
                "scale_const_std": self.scale_const_std,
                "freq": freq,
            },
            # Validation split
            {
                "x": data[
                    self.validation_start : self.validation_end, self.covariates_col
                ],
                "y": data[self.validation_start : self.validation_end, self.output_col],
                "covariates_col": self.covariates_col,
                "data_col_names": self.data.columns.tolist(),
                "freq": freq,
            },
            # Test split
            {
                "x": data[self.test_start : self.test_end, self.covariates_col],
                "y": data[self.test_start : self.test_end, self.output_col],
                "covariates_col": self.covariates_col,
                "data_col_names": self.data.columns.tolist(),
                "freq": freq,
            },
            # All data
            {
                "x": data[: self.test_end, self.covariates_col],
                "y": data[: self.test_end, self.output_col],
                "start_date": self.data.index[self.train_start],
                "covariates_col": self.covariates_col,
                "data_col_names": self.data.columns.tolist(),
                "freq": freq,
            },
        )

    def get_data(
        self,
        split: str,
        standardization: Optional[bool] = False,
        column: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return a specific column's values for a given data split.

        Args:
            split (str): One of ['train', 'validation', 'test', 'all'].
            Standardization (bool): Whether to standardize the output.
            column (Optional[int]):  Column index.

        Returns:
            np.ndarray: The extracted values.

        Examples:
            >>> values = dp.get_data(split="train", standardization=True, column=[0])
        """
        if standardization:
            data = self.standardize_data()
        else:
            data = self.data.values

        if column is not None:
            data_column = column
        else:
            data_column = self.output_col

        train_index, val_index, test_index = self.get_split_indices()
        if split == "train":
            return data[train_index][:, data_column]
        elif split == "validation":
            return data[val_index][:, data_column]
        elif split == "test":
            return data[test_index][:, data_column]
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

        Examples:
            >>> time = dp.get_time(split="train")
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
    def add_lagged_columns(
        data: pd.DataFrame, lags_per_column: list[int]
    ) -> pd.DataFrame:
        """
        Add lagged versions of each column in the dataset, then add to the dataset as
        new columns.

        Args:
            data (pd.DataFrame): Input DataFrame with datetime index.
            lags_per_column (list[int]): Number of lags per column.

        Returns:
            pd.DataFrame: New DataFrame with lagged columns.

        Examples:
            >>> data_lag = DataProcess.add_lagged_columns(data, [2])
        """
        df_new = pd.DataFrame(index=data.index)
        for col_idx, num_lags in enumerate(lags_per_column):
            col = data.iloc[:, col_idx]
            df_new[col.name] = col
            for lag in range(1, num_lags + 1):
                df_new[f"{col.name}_lag{lag}"] = col.shift(lag).fillna(0)

        if len(lags_per_column) < data.shape[1]:
            for col_idx in range(len(lags_per_column), data.shape[1]):
                col = data.iloc[:, col_idx]
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
        Add randomly generated synthetic anomalies to the original data.
        From the orginal data, choose a window between `anomaly_start` and `anomaly_end` (ratio: 0-1).
        Following a uniform distribution, it randomly chooses within this window where the anomaly starts.
        After the anomaly start, the data is linearly shifted with a rate of change define by
        `slope`.

        Args:
            data (dict): Data dict with "x" and "y".
            num_samples (int): Number of anomalies to generate.
            slope (list[float]): Slope for an anomaly.
            anomaly_start (float, optional): Start of the anomaly window (0-1). Defaults to 0.33.
            anomaly_end (float, optional): End of the anomaly window (0-1). Defaults to 0.66.

        Returns:
            list: Data dicts with anomalies injected.

        Examples:
            >>> train_set, val_set, test_set, all_data = dp.get_splits()
            >>> train_set_with_anomaly = DataProcess.add_synthetic_anomaly(
                                            train_set,
                                            num_samples=2,
                                            slope=[0.01, 0.1],
                                        )
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
    def decompose_data(data) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Decompose a time series into a linear trend, seasonality, and residual following:

         - Use a Fourier transform to estimate seasonality.
         - `Deseasonalized_data = data - seasonality`
         - Estimate a linear trend by fitting `Deseasonalized_data` with a first order polynomial
         - Estimate residual = data - trend - seasonality

        Args:
            data (np.ndarray): 1D array.

        Returns:
            tuple: (trend, slope_of_trend, seasonality, residual)


        Examples:
            >>> train_set, val_set, test_set, all_data = dp.get_splits()
            >>> trend, slope_of_trend, seasonality, residual = DataProcess.decompose_data(train_set["y"].flatten())
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
