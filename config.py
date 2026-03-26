# config.py

DATA_PATH = r"D:\project1\pythonProject\QMT\单股票每日态势识别\论文\dataset_model_baseline_longer_trend.parquet"

TARGET_THRESHOLD = 0.01
TRAIN_END_DATE = "2023-01-01"

CLIP_LOW_Q = 0.01
CLIP_HIGH_Q = 0.99

TOP_K_LIST = [0.01, 0.03, 0.05]

BASELINE_PLUS_LONGER_TREND_RAW_FEATURES = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "momentum_change",
    "ret_1d_minus_5d",
    "ret_3d_minus_10d",
    "ret_1d_minus_3d",
    "roc_20",
    "ema30_slope_vr",
    "ema30_slope",
    "ma30_slope",
    "bias_60_vr",
    "bias_60",
    "board_rank_20d_pct",
    "board_rs_20d",
    "ema60_slope",
    "ema90_slope",
    "ema180_slope",
    "ma60_slope",
    "ma180_slope",
]

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}
