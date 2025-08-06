import argparse
import pandas as pd
from pycaret.time_series import TSForecastingExperiment

def main():
    parser = argparse.ArgumentParser(
        description="Train, compare (including naive) and tune models on a CSV"
    )
    parser.add_argument('--csv',        required=True,
                        help='Path to input CSV file')
    parser.add_argument('--date_col',   default='timestamp',
                        help='Name or zero-based index (as string) of the datetime column')
    parser.add_argument('--target_col', default='value',
                        help='Name or zero-based index (as string) of the target column')
    parser.add_argument('--freq',       default='S',
                        help='Frequency for PeriodIndex (e.g., S for seconds)')
    parser.add_argument('--horizon',    type=int, default=10,
                        help='Forecast horizon')
    parser.add_argument('--seasonal_period', type=int, default=60,
                        help='Seasonal period for PyCaret setup')
    parser.add_argument('--session_id', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # 1) Read CSV
    df = pd.read_csv(args.csv)

    # 2) Select date and target columns
    if args.date_col not in df.columns or args.target_col not in df.columns:
        raise ValueError(f"Columns '{args.date_col}' and/or '{args.target_col}' not found in CSV.")
    df = df[[args.date_col, args.target_col]].copy()
    df.columns = ['date', 'target']

    # 3) Parse dates
    if pd.api.types.is_integer_dtype(df['date']) or pd.api.types.is_float_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], unit='s')
    else:
        df['date'] = pd.to_datetime(df['date'])

    # 4) Set datetime index & extract series
    df.set_index('date', inplace=True)
    y = df['target']

    # 5) Convert to PeriodIndex
    y.index = pd.PeriodIndex(y.index, freq=args.freq)

    # 6) Initialize experiment
    exp = TSForecastingExperiment()
    exp.setup(
        data=y,
        fh=args.horizon,
        seasonal_period=args.seasonal_period,
        session_id=args.session_id
    )

    # 7) Determine which baseline models are available
    available = exp.models().index.tolist()
    desired = ['naive', 'snaive', 'et_cds_dt', 'xgboost_cds_dt']
    include = [m for m in desired if m in available]
    print(f"\nAvailable models for comparison: {include}")

    # 8) Compare selected models
    print("\n=== Initial Comparison ===")
    exp.compare_models(include=include, sort='MAE')
    print(exp.pull().to_string(index=False))

    tuned_models = {}
    # 9) Tune ExtraTrees and XGBoost if present
    if 'et_cds_dt' in include:
        print("=== Tuning ExtraTrees (et_cds_dt) ===")
        et_model = exp.create_model('et_cds_dt')
        tuned_et = exp.tune_model(et_model, optimize='MAE')
        print(exp.pull().to_string(index=False))
        # record tuned model
        tuned_models['ExtraTrees'] = tuned_et
        # print tuned parameters
        try:
            et_params = tuned_et.steps[-1][1].get_params()
        except Exception:
            et_params = tuned_et.get_params()
        print("Tuned ExtraTrees parameters:", et_params)

    if 'xgboost_cds_dt' in include:
        print("=== Tuning XGBoost (xgboost_cds_dt) ===")
        xgb_model = exp.create_model('xgboost_cds_dt')
        tuned_xgb = exp.tune_model(xgb_model, optimize='MAE')
        print(exp.pull().to_string(index=False))
        # record tuned model
        tuned_models['XGBoost'] = tuned_xgb
        # print tuned parameters
        try:
            xgb_params = tuned_xgb.steps[-1][1].get_params()
        except Exception:
            xgb_params = tuned_xgb.get_params()
        print("Tuned XGBoost parameters:", xgb_params)

    # 10) Final performance including naive baselines and tuned models['XGBoost'] = tuned_xgb

        # 10) Final performance including naive baselines and tuned models
    final_include = []
    # always include naive baselines if available
    if 'naive' in include:
        final_include.append('naive')
    if 'snaive' in include:
        final_include.append('snaive')
    # include tuned model objects
    for model in tuned_models.values():
        final_include.append(model)

    if final_include:
        print("=== Final Performance Including Baselines and Tuned Models ===")
        # compare_models accepts both IDs and model objects
        exp.compare_models(include=final_include, sort='MAE')
        print(exp.pull().to_string(index=False))

if __name__ == '__main__':
    main()
