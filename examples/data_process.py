from typing import Optional, Tuple, List
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
    ) -> None:
        self._train_start = train_start
        self._train_end = train_end
        self._validation_start = validation_start
        self._validation_end = validation_end
        self._test_start = test_start

        self.data = data
        self.time_covariates = time_covariates
        self.train_data = None
        self.validation_data = None
        self.test_data = None

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

        self.train_data = self.data.loc[self._train_start : self._train_end].values
        if self._validation_start is not None:
            self.validation_data = self.data.loc[
                self._validation_start : self._validation_end
            ].values
        if self._test_start is not None:
            self.test_data = self.data.loc[self._test_start :].values

    def normalize_data(self):
        """
        Nomalize data
        """

        self.data_mean, self.data_std = Normalizer.compute_mean_std(self.train_data)

        self.train_data = Normalizer.standardize(
            data=self.train_data, mu=self.data_mean, std=self.data_std
        )
        self.train_data = np.float32(self.train_data)

        if self.validation_data is not None:
            self.validation_data = Normalizer.standardize(
                data=self.validation_data, mu=self.data_mean, std=self.data_std
            )
            self.validation_data = np.float32(self.validation_data)

        if self.test_data is not None:
            self.test_data = Normalizer.standardize(
                data=self.test_data, mu=self.data_mean, std=self.data_std
            )
            self.test_data = np.float32(self.test_data)

    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the train, valiation, test splits
        """

        return self.train_data, self.validation_data, self.test_data
