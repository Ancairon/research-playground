import pandas as pd
from pycaret.time_series import *

# Load the dataset

data = pd.read_csv("_system.cpu_user.csv", header=0,
                   index_col=0, parse_dates=True)

# Display the first few rows
print(data.head())

# Initialize the PyCaret environment for time series forecasting
ts_setup = setup(
    data=data,
    target='1',
    session_id=123,
    fold=3,
    fh=10,
    use_gpu = True
)

ts_setup.models()

# Compare all models
best_model = compare_models(sort='MASE')  # Sort by MASE (Mean Absolute Scaled Error)

# Tune the best model
tuned_model = tune_model(best_model)


# Make predictions for the next secs
future_forecast = predict_model(tuned_model)
print(future_forecast)

# Plot model diagnostics
plot_model(tuned_model, plot='diagnostics')


# Plot forecast
plot_model(tuned_model, plot='forecast')


# Save the trained model
save_model(tuned_model, 'multi_step_forecast_model')