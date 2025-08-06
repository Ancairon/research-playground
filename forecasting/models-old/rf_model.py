

# rf_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def create_lag_features(df, lag_window):
    df_feat = pd.DataFrame(index=df.index)
    for lag in range(1, lag_window+1):
        df_feat[f'lag_{lag}'] = df['value'].shift(lag)
    df_feat['target'] = df['value']
    return df_feat.dropna()


def train_model(df, lag_window):
    feats = create_lag_features(df, lag_window)
    X = feats[[f'lag_{i}' for i in range(1, lag_window+1)]].values
    y = feats['target'].values
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X, y)
    return model


def forecast_next(df, model, lag_window):
    recent = df['value'].iloc[-lag_window:].values
    Xn = recent[::-1].reshape(1, -1)
    return model.predict(Xn)[0]

