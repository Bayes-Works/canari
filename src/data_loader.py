from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd

from pytagi import Normalizer, Utils
from sklearn.preprocessing import StandardScaler

class DataPreProcess:
    def __init__(
        self, 
        data: pd.DataFrame, 
        train_time_start: str, 
        val_time_start: str, 
        test_time_start: str, 
        test_time_end: str,
        time_covariates: Optional[str] = None,
        ) -> None:
        self.data = data.values
        self.time = data.index
        self.train_time_start   = pd.to_datetime(train_time_start)
        self.val_time_start     = pd.to_datetime(val_time_start)
        self.test_time_start    = pd.to_datetime(test_time_start)
        self.test_time_end      = pd.to_datetime(test_time_end)
        self.time_covariates    = time_covariates

        # Add time covariates when needed
        if self.time_covariates is not None:
            self.add_time_covariates() 
         
        # Perform preprocessing
        self.preprocess_data()
        
    def preprocess_data(self):
        # Filter data based on time ranges
        train_mask      = (self.time >= self.train_time_start) & (self.time < self.val_time_start)
        trainVal_mask   = (self.time >= self.train_time_start) & (self.time < self.test_time_start)
        val_mask        = (self.time >= self.val_time_start) & (self.time < self.test_time_start)
        test_mask       = (self.time >= self.test_time_start) & (self.time <= self.test_time_end)

        # split data
        data_train      = self.data[train_mask,:]
        data_val        = self.data[val_mask,:]
        data_test       = self.data[test_mask,:]
        self.time       = np.array(self.time, dtype="datetime64")
        self.time_train = self.time[train_mask]
        self.time_val   = self.time[val_mask]
        self.time_test  = self.time[test_mask]

        # Calulate the mean and std from the training and validation sets
        self.x_mean, self.x_std = Normalizer.compute_mean_std(self.data[trainVal_mask,:])
        self.data_train = Normalizer.standardize(data=data_train, mu=self.x_mean, std=self.x_std)
        self.data_val = Normalizer.standardize(data=data_val, mu=self.x_mean, std=self.x_std)
        self.data_test = Normalizer.standardize(data=data_test, mu=self.x_mean, std=self.x_std)

        # return data_train, data_val, data_test, x_mean, x_std


    def add_time_covariates(self):
        # Add time covariates
        time = self.time.values.reshape(-1, 1)
        time = np.array(time, dtype="datetime64")
        for time_cov in self.time_covariates:
            if time_cov == "hour_of_day":
                hour_of_day = time.astype("datetime64[h]").astype(int) % 24
                self.data = np.concatenate((self.data, hour_of_day), axis=1)
            elif time_cov == "day_of_week":
                day_of_week = time.astype("datetime64[D]").astype(int) % 7
                self.data = np.concatenate((self.data, day_of_week), axis=1)
            elif time_cov == "week_of_year":
                week_of_year = (
                    time.astype("datetime64[W]").astype(int) % 52 + 1
                )
                self.data = np.concatenate((self.data, week_of_year), axis=1)
            elif time_cov == "month_of_year":
                month_of_year = (
                    time.astype("datetime64[M]").astype(int) % 12 + 1
                )
                self.data = np.concatenate((self.data, month_of_year), axis=1)
            elif time_cov == "quarter_of_year":
                month_of_year = (
                    time.astype("datetime64[M]").astype(int) % 12 + 1
                )
                quarter_of_year = (month_of_year - 1) // 3 + 1
                self.data = np.concatenate((self.data, quarter_of_year), axis=1)

    @staticmethod
    def resample_data(data, time: str, freq: str, agg_func='mean') -> pd.DataFrame:
        """
        Resample the data based on the provided frequency and take the average for aggregation.
        
        Parameters:
        freq (str): Frequency for resampling, e.g., 'H' for hourly, 'W' for weekly, 'M' for monthly.
        
        Returns:
        pd.DataFrame: Resampled data with a uniform time step.
        """
        # Ensure that time is part of the DataFrame for resampling
        data.columns = [f'col_{i+1}' for i in range(data.shape[1])]
        data['time']  = pd.to_datetime(time[0], format='%Y-%m-%d %H:%M:%S')

        # Set 'time' as the index
        data.set_index('time', inplace=True)

        if agg_func == 'mean':
            resampled = data.resample(freq).mean()
        elif agg_func == 'sum':
            resampled = data.resample(freq).sum()
        elif agg_func == 'max':
            resampled = data.resample(freq).max()
        elif agg_func == 'min':
            resampled = data.resample(freq).min()
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")
        
        # resampled = resampled.reset_index()
        
        return resampled
    


        
