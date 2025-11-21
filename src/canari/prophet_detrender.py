import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Prophet import with fallback
try:
    from prophet import Prophet
except ImportError:  # older package name
    from fbprophet import Prophet


def fmt_freq(freq: str) -> str:
    """
    Normalize shorthand frequency codes to pandas-compatible ones if needed.
    This is mainly here in case you want to extend handling later.
    """
    return freq


class ProphetDetrender:
    """
    Decompose a time series into trend, seasonal component, and residuals
    using Prophet, while preserving missing values from the original series.
    """

    def __init__(
        self,
        freq: str | None = "D",
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        **prophet_kwargs,
    ) -> None:
        self.freq = freq
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.prophet_kwargs = prophet_kwargs

        self.model_ = None
        self.result_ = None
        self._nan_mask = None
        self._index = None

    @staticmethod
    def _prepare_prophet_df(series: pd.Series) -> pd.DataFrame:
        """
        Convert a pandas Series with DatetimeIndex to Prophet's expected format.
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series index must be a pandas.DatetimeIndex.")

        df = series.to_frame(name="y").reset_index()
        df.rename(columns={"index": "ds"}, inplace=True)
        return df

    def _get_freq(self, series: pd.Series) -> str:
        """Return frequency to use for reindexing."""
        if self.freq is not None:
            return self.freq

        inferred = pd.infer_freq(series.index)
        if inferred is None:
            # Fallback to daily if cannot infer
            return "D"
        return inferred

    def fit(self, series: pd.Series):
        """
        Fit the Prophet model and compute decomposition components.
        """
        # Ensure time-ordered index
        series = series.sort_index()
        freq = self._get_freq(series)

        # Regularize index to chosen frequency
        full_idx = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq=fmt_freq(freq),
        )
        series = series.reindex(full_idx)

        # Store internal state
        self._index = full_idx
        self._nan_mask = series.isna()

        # Fill NaNs for fitting (Prophet cannot handle NaNs)
        y_filled = series.copy()
        y_filled = y_filled.ffill().bfill()

        df_prophet = self._prepare_prophet_df(y_filled)

        # Fit Prophet model
        self.model_ = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            **self.prophet_kwargs,
        )
        self.model_.fit(df_prophet)

        # Predict on same time grid
        forecast = self.model_.predict(df_prophet[["ds"]])

        # ---- robust seasonal calculation ----
        if "seasonal" in forecast.columns:
            seasonal = forecast["seasonal"].values
        else:
            # Sum all known seasonal components
            seasonal_components = []

            # components from model_.seasonalities (e.g. 'yearly', 'weekly', 'daily', custom)
            for name in getattr(self.model_, "seasonalities", {}).keys():
                if name in forecast.columns:
                    seasonal_components.append(name)

            # holidays if present
            if "holidays" in forecast.columns:
                seasonal_components.append("holidays")

            if seasonal_components:
                seasonal = forecast[seasonal_components].sum(axis=1).values
            else:
                # no seasonal component at all
                seasonal = np.zeros(len(forecast))

        # Build result DataFrame
        result = pd.DataFrame(index=full_idx)
        result["y"] = series.values  # original with NaNs
        result["trend"] = forecast["trend"].values
        result["seasonal"] = seasonal
        result["yhat"] = forecast["yhat"].values

        # Residuals: y - yhat, but keep NaN where original y was NaN
        result["residual"] = result["y"] - result["yhat"]
        result.loc[self._nan_mask, "residual"] = np.nan

        self.result_ = result
        return self

    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """Fit the model and return the decomposition."""
        self.fit(series)
        return self.result_.copy()

    def plot(self):
        """Quick plot of original, trend, seasonal, and residuals."""
        if self.result_ is None:
            raise RuntimeError("Call fit() or fit_transform() before plot().")

        result = self.result_

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(result.index, result["y"])
        axes[0].set_title("Original series (with missing values)")

        axes[1].plot(result.index, result["trend"])
        axes[1].set_title("Trend")

        axes[2].plot(result.index, result["seasonal"])
        axes[2].set_title("Seasonal component (total)")

        axes[3].plot(result.index, result["residual"])
        axes[3].set_title("Residuals (NaN where original data was missing)")

        plt.tight_layout()
        plt.show()
