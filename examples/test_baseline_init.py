import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TimeSeriesDecomposition:
    def __init__(self, data):
        self.data = np.array(data)
        self.window = self.estimate_window()

    def moving_average(self, data, window):
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def estimate_window(self):
        # Using Fourier Transform to determine dominant frequency
        fft_spectrum = np.fft.fft(self.data - np.mean(self.data))
        frequencies = np.fft.fftfreq(len(self.data))
        power = np.abs(fft_spectrum)
        peak_frequency = frequencies[np.argmax(power[1:]) + 1]  # Ignore DC component
        period = int(1 / peak_frequency) if peak_frequency > 0 else len(self.data) // 10
        return max(3, period)  # Ensure a minimum window size

    def estimate_seasonality(self, data):
        period = self.window  # Use detected period from Fourier Transform
        seasonality = np.zeros_like(data)
        for i in range(period):
            seasonality[i::period] = np.mean(data[i::period])
        return seasonality - np.mean(seasonality)  # Zero mean seasonality

    def estimate_trend(self, deseasonalized_data):
        x = np.arange(len(deseasonalized_data))
        slope, intercept = np.polyfit(x, deseasonalized_data, 1)  # Linear trend
        trend = slope * x + intercept
        return trend, slope

    def decompose(self):
        seasonality = self.estimate_seasonality(self.data)
        deseasonalized_data = self.data - seasonality
        trend, slope = self.estimate_trend(deseasonalized_data)
        residual = self.data - trend - seasonality
        return trend, seasonality, residual, slope


# Example Usage
data = (
    np.sin(np.linspace(0, 4 * np.pi, 100))
    + np.linspace(0, 1, 100)
    + np.random.normal(0, 0.1, 100)
)
decomposer = TimeSeriesDecomposition(data)
trend, seasonality, residual, slope = decomposer.decompose()

print(f"Estimated Slope of Linear Trend: {slope}")

plt.figure(figsize=(10, 5))
plt.plot(data, label="Original Data")
plt.plot(trend, label="Trend")
plt.plot(seasonality, label="Seasonality")
plt.plot(residual, label="Residual")
plt.legend()
plt.show()
