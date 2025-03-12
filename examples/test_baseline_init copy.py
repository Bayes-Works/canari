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

    def estimate_trend(self):
        trend = self.moving_average(self.data, self.window)
        trend_padded = np.pad(trend, (self.window // 2, self.window // 2), mode="edge")
        return trend_padded[: len(self.data)]

    def estimate_seasonality(self, detrended):
        period = self.window  # Use detected period from Fourier Transform
        seasonality = np.zeros_like(detrended)
        for i in range(period):
            seasonality[i::period] = np.mean(detrended[i::period])
        return seasonality

    def decompose(self):
        trend = self.estimate_trend()
        detrended = self.data - trend
        seasonality = self.estimate_seasonality(detrended)
        residual = self.data - trend - seasonality
        return trend, seasonality, residual


# Example Usage
data = (
    np.sin(np.linspace(0, 4 * np.pi, 100))
    + np.linspace(0, 1, 100)
    + np.random.normal(0, 0.1, 100)
)
decomposer = TimeSeriesDecomposition(data)
trend, seasonality, residual = decomposer.decompose()

plt.figure(figsize=(10, 5))
plt.plot(data, label="Original Data")
plt.plot(trend, label="Trend")
plt.plot(seasonality, label="Seasonality")
plt.plot(residual, label="Residual")
plt.legend()
plt.show()
