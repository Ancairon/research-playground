# xgb_unified_forecaster.py
import argparse
import time
from collections import deque

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pycaret.time_series import TSForecastingExperiment

import xgboost
# patch __version__ if missing
if not hasattr(xgboost, "__version__"):
    try:
        import importlib.metadata as importlib_metadata
        xgboost.__version__ = importlib_metadata.version("xgboost")
    except Exception:
        xgboost.__version__ = "1.6.0"


def getDataFromAPI(ip, context, dimension, seconds_back):
    url = (
        f"http://{ip}:19999/api/v3/data?"
        f"contexts={context}&dimensions={dimension}&"
        f"after=-{seconds_back}&before=0&points={seconds_back}&"
        f"group=average&format=json&options=seconds,jsonwrap"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    rows = resp.json()['result']['data']
    df = pd.DataFrame(rows, columns=['timestamp', 'value'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.asfreq('s').ffill()
    return df


def _safe_mean(x): return float(np.mean(x)) if len(x) else float('nan')
def _safe_max(x): return float(np.max(x)) if len(x) else float('nan')
def _safe_min(x): return float(np.min(x)) if len(x) else float('nan')


def _safe_pct(x, q): return float(
    np.percentile(x, q)) if len(x) else float('nan')


def _build_xgb_pipeline(series: pd.Series, horizon: int, session_id: int):
    y = series.copy()
    y.index = pd.PeriodIndex(y.index, freq='S')
    exp = TSForecastingExperiment()
    exp.setup(
        data=y,
        fh=horizon,
        seasonal_period=60,
        session_id=session_id,
        numeric_imputation_target="ffill"
    )
    xgb_pipe = exp.create_model('xgboost_cds_dt')
    final_xgb = exp.finalize_model(xgb_pipe)
    return exp, final_xgb


def main(
        csv=None,
        ip=None,
        context=None,
        dimension=None,
        horizon=10,
        random_state=42,
        window=3600,
        target_col="value",
        date_col="timestamp",
        retrain_threshold=5,
        max_errors=1
):

    if csv:
        df_csv = pd.read_csv(csv)

        if np.issubdtype(df_csv[date_col].dtype, np.number):
            df_csv[date_col] = pd.to_datetime(df_csv[date_col], unit='s')
        else:
            df_csv[date_col] = pd.to_datetime(df_csv[date_col])
        df_csv.set_index(date_col, inplace=True)
        feed_df = df_csv[[target_col]].rename(columns={target_col: 'value'})
        window = min(window, len(feed_df))
        init_df = feed_df.iloc[:window].copy()
        feed_ptr = window  # simulation pointer
    else:

        init_df = getDataFromAPI(ip, context, dimension, window)

    exp, pipeline = _build_xgb_pipeline(
        init_df['value'], horizon, random_state)
    print("[XGB] Initial pipeline trained.")

    plt.ion()
    fig, ax = plt.subplots()

    errors = deque(maxlen=max_errors)   # rolling abs errors for window-average
    result_avg_err = []                 # list of window-averaged abs errors
    signed_errors = []                  # pointwise signed errors y_true - y_pred
    y_true_list = []
    y_pred_list = []
    prev_pred = None
    holdback = 0

    try:
        while True:
            if csv:
                if feed_ptr > len(feed_df):
                    print("End of simulation feed.")
                    break
                live_df = feed_df.iloc[:feed_ptr].copy()
                feed_ptr += 1
            else:
                live_df = getDataFromAPI(ip, context, dimension, window)

            ts = live_df['value'].copy()
            ts.index = pd.PeriodIndex(ts.index, freq='S')

            # update history
            pipeline.update(ts)

            # forecast
            fh = list(range(1, horizon + 1))
            fc = pipeline.predict(fh=fh)
            fc.index = fc.index.to_timestamp(freq='S')
            preds = fc.values.tolist()

            if prev_pred is not None:
                y_true_now = live_df['value'].iloc[-1]
                err_signed = y_true_now - prev_pred
                signed_errors.append(err_signed)
                y_true_list.append(y_true_now)
                y_pred_list.append(prev_pred)

                err_abs = abs(err_signed)
                errors.append(err_abs)

                if retrain_threshold is not None and len(errors) == errors.maxlen:
                    avg_err = float(np.mean(errors))
                    print("AVERAGE_ERROR", avg_err, holdback)
                    result_avg_err.append(avg_err)

                    if avg_err > retrain_threshold:
                        holdback += 1
                        if holdback == 5:
                            print(
                                "Major anomaly Holdback for 5 seconds before retrain")
                            time.sleep(5)
                            holdback = 0

                        try:
                            start_t = time.time()
                            exp, pipeline = _build_xgb_pipeline(
                                live_df['value'], horizon, random_state
                            )
                            print(
                                f"Retraining time: {time.time() - start_t:.3f}s\n")
                        except Exception as e:
                            print(
                                f"[XGB] Retrain failed (continuing with existing pipeline): {e}")
                        finally:
                            errors.clear()
                    else:
                        print("zeroing the holdback")
                        holdback = 0
                        errors.clear()

            prev_pred = preds[0] if preds else None

            recent = ts.to_timestamp(freq='S').iloc[-10:]
            ax.clear()
            ax.plot(recent.index, recent.values, 'b-', label='Actual')
            ax.plot(fc.index,      fc.values,    'r--o',
                    label=f'Forecast ({horizon}s)')

            all_times = recent.index.union(fc.index)
            ax.set_xlim(all_times.min(), all_times.max())
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()

            ax.set_ylim(0, 100)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nLive forecasting stopped by user.")

    avg_mean = _safe_mean(result_avg_err)
    avg_max = _safe_max(result_avg_err)
    avg_min = _safe_min(result_avg_err)
    p80 = _safe_pct(result_avg_err, 80)
    p95 = _safe_pct(result_avg_err, 95)
    p99 = _safe_pct(result_avg_err, 99)

    y_true_arr = np.array(y_true_list, dtype=float)
    y_pred_arr = np.array(y_pred_list, dtype=float)
    mbe = float(np.mean(y_true_arr - y_pred_arr)
                ) if len(y_true_arr) else float('nan')
    pbias = (
        float(100.0 * np.sum(y_true_arr - y_pred_arr) / np.sum(y_true_arr))
        if len(y_true_arr) and np.sum(y_true_arr) != 0 else float('nan')
    )

    print("AVERAGE", avg_mean, "MAX", avg_max, "MIN", avg_min)
    print(f"P80 {p80}  P95 {p95}  P99 {p99}")
    print(f"MBE {mbe}  PBIAS% {pbias}")

    return {
        "mean_avg_err": avg_mean,
        "max_avg_err": avg_max,
        "min_avg_err": avg_min,
        "p80_avg_err": p80,
        "p95_avg_err": p95,
        "p99_avg_err": p99,
        "mbe": mbe,
        "pbias_pct": pbias
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Unified XGB forecaster (ET-style interface)")
    parser.add_argument('--csv',            default=None, help='CSV file path')
    parser.add_argument(
        '--ip',             default='localhost', help='Netdata IP')
    parser.add_argument(
        '--context',        default='system.cpu', help='Netdata chart context')
    parser.add_argument('--dimension',      default='user',
                        help='Netdata dimension key')
    parser.add_argument('--horizon',        type=int,
                        default=10, help='Forecast horizon (steps)')
    parser.add_argument('--random_state',   type=int,
                        default=42, help='Random seed / session_id')
    parser.add_argument('--window',         type=int,
                        default=3600, help='Seconds of history to fetch')
    parser.add_argument('--target_col',     default='value',
                        help='CSV target column if using CSV')
    parser.add_argument('--date_col',       default='timestamp',
                        help='CSV datetime column if using CSV')
    parser.add_argument('--retrain_threshold', type=float, default=5,
                        help='Avg abs error threshold to trigger retrain')
    parser.add_argument('--max_errors',     type=int, default=1,
                        help='Errors to average before threshold check')
    args = parser.parse_args()

    main(
        csv=args.csv,
        ip=args.ip,
        context=args.context,
        dimension=args.dimension,
        horizon=args.horizon,
        random_state=args.random_state,
        window=args.window,
        target_col=args.target_col,
        date_col=args.date_col,
        retrain_threshold=args.retrain_threshold,
        max_errors=args.max_errors
    )
