from extra_trees_minimal import main as etmain

score = etmain(csv='system.cpu_user.csv',
       ip=None,
       context=None,
       dimension=None,
       date_col='timestamp',
       target_col="value",
       max_lag=10,
       test_size=.3,
       n_estimators=222,
       random_state=42,
       horizon=10,
       window=3600,
       retrain_threshold=5
       )