from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from config import (
    TARGET_THRESHOLD,
    TRAIN_END_DATE,
    CLIP_LOW_Q,
    CLIP_HIGH_Q,
    BASELINE_PLUS_LONGER_TREND_RAW_FEATURES,
)


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: List[str]
    target_col: str
    id_cols: List[str]


class DataPipeline:
    """
    统一数据处理入口：
    1. 读取原始数据
    2. 构建二分类标签
    3. 构造变化量特征
    4. 选择主因子组
    5. 时间切分
    6. 按训练集分位数 clip
    """

    def __init__(
        self,
        data_path: str,
        target_threshold: float = TARGET_THRESHOLD,
        train_end_date: str = TRAIN_END_DATE,
        clip_low_q: float = CLIP_LOW_Q,
        clip_high_q: float = CLIP_HIGH_Q,
    ) -> None:
        self.data_path = data_path
        self.target_threshold = target_threshold
        self.train_end_date = pd.Timestamp(train_end_date)
        self.clip_low_q = clip_low_q
        self.clip_high_q = clip_high_q

        self.target_col = "regime_binary"
        self.id_cols = ["date", "ticker", "r_future_5"]

    def load_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path).copy()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.target_col] = (df["r_future_5"] > self.target_threshold).astype(int)
        return df

    def build_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "ret_1d_minus_5d" not in df.columns:
            df["ret_1d_minus_5d"] = df["ret_1d"] - df["ret_5d"]

        if "ret_3d_minus_10d" not in df.columns:
            df["ret_3d_minus_10d"] = df["ret_3d"] - df["ret_10d"]

        if "ret_1d_minus_3d" not in df.columns:
            df["ret_1d_minus_3d"] = df["ret_1d"] - df["ret_3d"]

        return df

    def get_feature_cols(self) -> List[str]:
        return BASELINE_PLUS_LONGER_TREND_RAW_FEATURES

    def select_columns(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        required_cols = self.id_cols + [self.target_col] + feature_cols
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        out = df[required_cols].copy()
        out = out.dropna(subset=feature_cols + [self.target_col]).reset_index(drop=True)
        return out

    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = df[df["date"] < self.train_end_date].copy()
        test_df = df[df["date"] >= self.train_end_date].copy()

        train_df = train_df.sort_values(["date", "ticker"]).reset_index(drop=True)
        test_df = test_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        return train_df, test_df

    def clip_by_train_quantile(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = train_df.copy()
        test_df = test_df.copy()

        for col in feature_cols:
            lower = train_df[col].quantile(self.clip_low_q)
            upper = train_df[col].quantile(self.clip_high_q)

            train_df[col] = train_df[col].clip(lower, upper)
            test_df[col] = test_df[col].clip(lower, upper)

        return train_df, test_df

    def build_bundle(self) -> DatasetBundle:
        print("[1/6] Loading data...")
        df = self.load_data()
        print("loaded shape:", df.shape)

        print("[2/6] Building target...")
        df = self.build_target(df)

        print("[3/6] Building delta features...")
        df = self.build_delta_features(df)

        print("[4/6] Selecting feature columns...")
        feature_cols = self.get_feature_cols()
        df = self.select_columns(df, feature_cols)
        print("selected shape:", df.shape)

        print("[5/6] Splitting train/test...")
        train_df, test_df = self.split_train_test(df)
        print("train shape:", train_df.shape, "test shape:", test_df.shape)

        print("[6/6] Clipping by train quantiles...")
        train_df, test_df = self.clip_by_train_quantile(train_df, test_df, feature_cols)
        print("clip done.")

        return DatasetBundle(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=self.target_col,
            id_cols=self.id_cols,
        )


if __name__ == "__main__":
    from config import DATA_PATH

    pipeline = DataPipeline(data_path=DATA_PATH)
    bundle = pipeline.build_bundle()

    print("Train shape:", bundle.train_df.shape)
    print("Test shape :", bundle.test_df.shape)
    print("Target col :", bundle.target_col)
    print("Num features:", len(bundle.feature_cols))
    print("Features:")
    print(bundle.feature_cols)

    print("\nTrain label distribution:")
    print(bundle.train_df[bundle.target_col].value_counts(normalize=True))

    print("\nTest label distribution:")
    print(bundle.test_df[bundle.target_col].value_counts(normalize=True))