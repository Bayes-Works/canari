import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df.head()

# Keep 20% of the data
df = df.iloc[:int(len(df) * 1)]

m = Prophet(changepoint_range=1, n_changepoints =int(len(df) * 0.05), changepoint_prior_scale=0.001)
m.fit(df)

future = m.make_future_dataframe(periods=365)
print(future.tail())

# forecast = m.predict(future)
forecast = m.predict(df)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Multiply forecast['yhat_lower'] and forecast['yhat_upper'] by a scale
forecast['yhat_lower'] = forecast['yhat'] - (forecast['yhat'] - forecast['yhat_lower']) * 1
forecast['yhat_upper'] = forecast['yhat'] + (forecast['yhat_upper'] - forecast['yhat']) * 1

forecast['anomaly'] = ((df['y'] < forecast['yhat_lower']) | (df['y'] > forecast['yhat_upper'])).astype(int)

print(forecast)
# Plot the anomalies in figure 1
fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast)
# for i in range(len(forecast)):
#     if forecast['anomaly'][i] == 1:
#         plt.plot(forecast['ds'][i], df['y'][i], 'r.')

fig2 = m.plot_components(forecast)
plt.show()