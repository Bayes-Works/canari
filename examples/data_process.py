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
        train_start: str,
        train_end: str,
        validation_start: Optional[str] = None,
        validation_end: Optional[str] = None,
        test_start: Optional[str] = None,
        time_covariates: Optional[List[str]] = None,
        output_col: list[int] = [0],
    ) -> None:
        self._train_start = train_start
        self._train_end = train_end
        self._validation_start = validation_start
        self._validation_end = validation_end
        self._test_start = test_start

        self.data = data
        self.time_covariates = time_covariates
        self.output_col = output_col
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.train_x = None
        self.train_y = None
        self.validation_x = None
        self.validation_y = None
        self.test_x = None
        self.test_y = None

        # # Add time covariates if needed
        if self.time_covariates is not None:
            self.add_time_covariates()

        # Split data
        self.split_data()

        #  Normalize data
        self.normalize_data()

    def add_time_covariates(self):
        """
        Add time covariates to the data
        """

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
        """ "
        Split data into train, validation and test sets
        """

        self.train_data = np.float32(
            self.data.loc[self._train_start : self._train_end].values
        )
        self.train_time = self.data.loc[self._train_start : self._train_end].index

        if self._validation_start is not None:
            self.validation_data = np.float32(
                self.data.loc[self._validation_start : self._validation_end].values
            )
            self.validation_time = self.data.loc[
                self._validation_start : self._validation_end
            ].index
        if self._test_start is not None:
            self.test_data = np.float32(self.data.loc[self._test_start :].values)
            self.test_time = self.data.loc[self._test_start :].index
        self.time = self.data.loc[self._train_start :].index

    def normalize_data(self):
        """
        Nomalize data
        """
        self.covariates_col = np.ones(self.train_data.shape[1], dtype=bool)
        self.covariates_col[self.output_col] = False

        self.data_mean, self.data_std = Normalizer.compute_mean_std(self.train_data)

        train_data_norm = Normalizer.standardize(
            data=self.train_data, mu=self.data_mean, std=self.data_std
        )
        self.train_y = train_data_norm[:, self.output_col]
        self.train_x = train_data_norm[:, self.covariates_col]
        self.x_normalized = self.train_x.copy()
        self.y_normalized = self.train_y.copy()

        if self.validation_data is not None:
            validation_data_norm = Normalizer.standardize(
                data=self.validation_data, mu=self.data_mean, std=self.data_std
            )
            self.validation_x = validation_data_norm[:, self.covariates_col]
            self.validation_y = validation_data_norm[:, self.output_col]
            self.x_normalized = np.concatenate(
                [self.x_normalized, self.validation_x], axis=0
            )
            self.y_normalized = np.concatenate(
                [self.y_normalized, self.validation_y], axis=0
            )

        if self.test_data is not None:
            test_data_norm = Normalizer.standardize(
                data=self.test_data, mu=self.data_mean, std=self.data_std
            )
            self.test_x = test_data_norm[:, self.covariates_col]
            self.test_y = test_data_norm[:, self.output_col]
            self.x_normalized = np.concatenate([self.x_normalized, self.test_x], axis=0)
            self.y_normalized = np.concatenate([self.y_normalized, self.test_y], axis=0)

    def get_splits(
        self,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get the train, valiation, test splits
        """

        return (
            {"x": self.train_x, "y": self.train_y},
            {"x": self.validation_x, "y": self.validation_y},
            {"x": self.test_x, "y": self.test_y},
            {"x": self.x_normalized, "y": self.y_normalized},
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
