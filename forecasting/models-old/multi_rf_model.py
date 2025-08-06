# multi_rf_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def train_model(df, lag_window, horizon):
    data = pd.DataFrame(index=df.index)
    for lag in range(1, lag_window+1):
        data[f'lag_{lag}'] = df['value'].shift(lag)
    for h in range(1, horizon+1):
        data[f't+{h}'] = df['value'].shift(-h)
    data.dropna(inplace=True)

    X = data[[f'lag_{i}' for i in range(1, lag_window+1)]].values
    y = data[[f't+{h}' for h in range(1, horizon+1)]].values

    base = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X, y)
    return model


def forecast(df, model, lag_window, horizon):
    recent = df['value'].iloc[-lag_window:].values
    X_next = recent[::-1].reshape(1, -1)
    preds = model.predict(X_next)[0]
    last_time = df.index[-1]
    times = [last_time + pd.Timedelta(seconds=i) for i in range(1, horizon+1)]
    return pd.Series(preds, index=times)
