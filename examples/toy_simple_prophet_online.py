import pandas as pd
from prophet import Prophet
import matplotlib
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot


df_raw = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df_raw.head()

# Genetrate percentages_check from 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, ... , 1
percentages_check = [i / 100 for i in range(10, 101, 5)]

# Keep 20% of the data
for i, percentage in enumerate(percentages_check):
    df = df_raw.iloc[:int(len(df_raw) * percentage)]

    m = Prophet(changepoint_range=1, n_changepoints =int(len(df) * 0.1), changepoint_prior_scale=0.005)
    m.fit(df)

    # future = m.make_future_dataframe(periods=365)
    # print(future.tail())

    # forecast = m.predict(future)
    forecast = m.predict(df)
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Multiply forecast['yhat_lower'] and forecast['yhat_upper'] by a scale
    forecast['yhat_lower'] = forecast['yhat'] - (forecast['yhat'] - forecast['yhat_lower']) * 1
    forecast['yhat_upper'] = forecast['yhat'] + (forecast['yhat_upper'] - forecast['yhat']) * 1

    forecast['anomaly'] = ((df['y'] < forecast['yhat_lower']) | (df['y'] > forecast['yhat_upper'])).astype(int)

    # print(forecast)
    fig1 = m.plot(forecast)
    a = add_changepoints_to_plot(fig1.gca(), m, forecast)
    # Set ylimit to 0, 100
    fig1.gca().set_ylim(4.883487353908684, 13.225949723825204)
    # Set xlimit to 0, 100
    fig1.gca().set_xlim(13708.85, 16968.15)

    # Plot the anomalies in figure 1
    # for i in range(len(forecast)):
    #     if forecast['anomaly'][i] == 1:
    #         plt.plot(forecast['ds'][i], df['y'][i], 'r.')

    # fig2 = m.plot_components(forecast)
    # Show the figure automatically, and close for the next loop
    # plt.show()

    # # Plot the trend rate of change
    # # Trend change point threshold: m.params['delta'][0] > 0.01
    # plt.figure()
    # plt.plot(m.params['delta'][0])
    # plt.axhline(0, color='black', linestyle='--')
    # plt.title('Trend Rate of Change')
    # plt.xlabel('Date')
    # plt.ylabel('Rate of Trend Change')
    
    if i != len(percentages_check) - 1:
        plt.pause(0.5)
        plt.close(fig1)
    elif i == len(percentages_check) - 1:
        fig2 = m.plot_components(forecast)
        print("--------finished--------")
        plt.show()
        # # Get the ylimit of the first figure
        # print(fig1.gca().get_ylim())
        # print(fig1.gca().get_xlim())