# arima_model.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def train_model(df):
    series = df['value'].asfreq('s')
    return ARIMA(series, order=(1, 1, 1)).fit()


def forecast(df, model, horizon):
    fc_values = model.forecast(steps=horizon)
    last_time = df.index.max()
    times = [last_time + pd.Timedelta(seconds=i) for i in range(1, horizon+1)]
    return pd.Series(fc_values.values, index=times)
