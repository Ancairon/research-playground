# hw_model.py
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def train_model(df):
    series = df['value'].asfreq('s')
    return ExponentialSmoothing(series, trend='add', seasonal=None).fit(optimized=True, use_boxcox=False,
                                                                        remove_bias=False,
                                                                        maxiter=1000,
                                                                        method='nm')


def forecast(df, model, horizon):
    fc_values = model.forecast(steps=horizon)
    last_time = df.index.max()
    times = [last_time + pd.Timedelta(seconds=i) for i in range(1, horizon+1)]
    return pd.Series(fc_values.values, index=times)
