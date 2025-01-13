from typing import Optional, Tuple, List, Dict
import copy
import numpy as np
import pandas as pd
from pytagi import Normalizer


class DataProcess:
    """
    Data process class
    """

    def __init__(
        self,
        data: pd.DataFrame,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        validation_start: Optional[str] = None,
        validation_end: Optional[str] = None,
        test_start: Optional[str] = None,
        train_split: Optional[float] = None,
        validation_split: Optional[float] = 0.0,
        test_split: Optional[float] = 0.0,
        time_covariates: Optional[List[str]] = None,
        output_col: list[int] = [0],
        normalization: Optional[bool] = True,
    ) -> None:
        self._train_start = train_start
        self._train_end = train_end
        self._validation_start = validation_start
        self._validation_end = validation_end
        self._test_start = test_start
        self._normalization = normalization
        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split
        self.time_covariates = time_covariates
        self.output_col = output_col

        self.data = data
        self.data_norm = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.train_data_norm = None
        self.validation_data_norm = None
        self.test_data_norm = None
        self.norm_const_mean = None
        self.norm_const_std = None

        # Add time covariates if needed
        self.add_time_covariates()

        # Covariates columns
        self.covariates_col = np.ones(self.data.shape[1], dtype=bool)
        self.covariates_col[self.output_col] = False

        # Split data
        self.split_data()

        #  Normalize data if needed
        self.normalization()

    def add_time_covariates(self):
        """
        Add time covariates to the data
        """

        if self.time_covariates is not None:
            for time_cov in self.time_covariates:
                if time_cov == "hour_of_day":
                    self.data["hour_of_day"] = self.data.index.hour
                elif time_cov == "day_of_week":
                    self.data["day_of_week"] = self.data.index.dayofweek
                elif time_cov == "day_of_year":
                    self.data["day_of_year"] = self.data.index.dayofyear
                elif time_cov == "week_of_year":
                    self.data["week_of_year"] = self.data.index.isocalendar().week
                elif time_cov == "month_of_year":
                    self.data["month"] = self.data.index.month
                elif time_cov == "quarter_of_year":
                    self.data["quarter"] = self.data.index.quarter

    def split_data(self):
        """
        Split data into train, validation, and test sets.
        """
        num_data = len(self.data)

        # Case 1: Splits are defined using ratios
        if self.train_split:
            self._train_end = int(np.floor(self.train_split * num_data))

            # # Determine validation and test split indices
            # if self.test_split == 0:
            #     self._validation_end = num_data
            # else:
            self._validation_end = self._train_end + int(
                np.ceil(self.validation_split * num_data)
            )

            # Extract train, validation, and test data and corresponding times
            self.train_data = self.data.iloc[: self._train_end].values.astype(
                np.float32
            )
            self.train_time = self.data.iloc[: self._train_end].index

            self.validation_data = self.data.iloc[
                self._train_end : self._validation_end
            ].values.astype(np.float32)
            self.validation_time = self.data.iloc[
                self._train_end : self._validation_end
            ].index

            self.test_data = self.data.iloc[self._validation_end :].values.astype(
                np.float32
            )
            self.test_time = self.data.iloc[self._validation_end :].index

        # Case 2: Splits are explicitly defined using indices
        else:
            if self._train_start is not None and self._train_end is not None:
                self.train_data = self.data.loc[
                    self._train_start : self._train_end
                ].values.astype(np.float32)
                self.train_time = self.data.loc[
                    self._train_start : self._train_end
                ].index

            if self._validation_start is not None and self._validation_end is not None:
                self.validation_data = self.data.loc[
                    self._validation_start : self._validation_end
                ].values.astype(np.float32)
                self.validation_time = self.data.loc[
                    self._validation_start : self._validation_end
                ].index

            if self._test_start is not None:
                self.test_data = self.data.loc[self._test_start :].values.astype(
                    np.float32
                )
                self.test_time = self.data.loc[self._test_start :].index

        # Store the overall time index
        self.time = self.data.index

    def normalization(self):
        """
        Nomalize data
        """

        if self._normalization:
            self.norm_const_mean, self.norm_const_std = Normalizer.compute_mean_std(
                self.train_data
            )
            self.data_norm = Normalizer.standardize(
                data=self.data.values.astype(np.float32),
                mu=self.norm_const_mean,
                std=self.norm_const_std,
            )
            self.train_data_norm = Normalizer.standardize(
                data=self.train_data, mu=self.norm_const_mean, std=self.norm_const_std
            )

            if self.validation_data is not None:
                self.validation_data_norm = Normalizer.standardize(
                    data=self.validation_data,
                    mu=self.norm_const_mean,
                    std=self.norm_const_std,
                )

            if self.test_data is not None:
                self.test_data_norm = Normalizer.standardize(
                    data=self.test_data,
                    mu=self.norm_const_mean,
                    std=self.norm_const_std,
                )

        else:
            self.train_data_norm = self.train_data
            self.validation_data_norm = self.validation_data
            self.test_data_norm = self.test_data
            self.data_norm = self.data.values

    def get_splits(
        self,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get the train, valiation, test splits
        """

        return (
            {
                "x": self.train_data_norm[:, self.covariates_col],
                "y": self.train_data_norm[:, self.output_col],
            },
            {
                "x": self.validation_data_norm[:, self.covariates_col],
                "y": self.validation_data_norm[:, self.output_col],
            },
            {
                "x": self.test_data_norm[:, self.covariates_col],
                "y": self.test_data_norm[:, self.output_col],
            },
            {
                "x": self.data_norm[:, self.covariates_col],
                "y": self.data_norm[:, self.output_col],
            },
        )

    @staticmethod
    def add_lagged_columns(df, lags_per_column):
        """
        Add lagged columns immediately after each original column.

        Parameters:
        df (pd.DataFrame): The original DataFrame.
        lags_per_column (list of int): Number of lags for each column, in order.

        Returns:
        pd.DataFrame: DataFrame with lagged columns added inline.
        """
        df_new = pd.DataFrame()
        current_position = 0  # To keep track of the insertion point

        for col_idx, num_lags in enumerate(lags_per_column):
            col = df.iloc[:, col_idx]
            df_new = pd.concat([df_new, col], axis=1)
            current_position += 1

            # Generate lagged columns
            for lag in range(1, num_lags + 1):
                lagged_col = col.shift(lag).fillna(0)
                df_new = pd.concat([df_new, lagged_col], axis=1)
                current_position += 1

        # If there are additional columns without specified lags, add them as is
        if len(lags_per_column) < df.shape[1]:
            for col_idx in range(len(lags_per_column), df.shape[1]):
                col = df.iloc[:, col_idx]
                df_new = pd.concat([df_new, col], axis=1)

        # Assign appropriate column names
        new_columns = []
        for col_idx, num_lags in enumerate(lags_per_column):
            new_columns.append(f"{df.columns[col_idx]}")  # Original column
            for lag in range(1, num_lags + 1):
                new_columns.append(f"{df.columns[col_idx]}_lag{lag}")  # Lagged columns
        # Add remaining columns without lags
        for col_idx in range(len(lags_per_column), df.shape[1]):
            new_columns.append(f"{df.columns[col_idx]}")

        df_new.columns = new_columns

        return df_new
