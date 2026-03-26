from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    preprocess_mode: str


class DataPipelineStandard:
    """
    只服务于标准化相关实验的数据处理入口：
    preprocess_mode 支持:
    - zscore   : 每日横截面 z-score
    - standard : 按训练集整体均值方差做 StandardScaler
    """

    def __init__(
        self,
        data_path: str,
        preprocess_mode: str = "zscore",
        target_threshold: float = TARGET_THRESHOLD,
        train_end_date: str = TRAIN_END_DATE,
        clip_low_q: float = CLIP_LOW_Q,
        clip_high_q: float = CLIP_HIGH_Q,
    ) -> None:
        self.data_path = data_path
        self.preprocess_mode = preprocess_mode.lower()
        self.target_threshold = target_threshold
        self.train_end_date = pd.Timestamp(train_end_date)
        self.clip_low_q = clip_low_q
        self.clip_high_q = clip_high_q

        if self.preprocess_mode not in {"zscore", "standard"}:
            raise ValueError("preprocess_mode must be one of {'zscore', 'standard'}")

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

    def select_columns(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
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

    @staticmethod
    def cross_sectional_zscore(
        df_in: pd.DataFrame,
        feature_cols: List[str],
        date_col: str = "date",
    ) -> pd.DataFrame:
        out = df_in.copy()

        for col in feature_cols:
            grp = out.groupby(date_col)[col]
            mean_ = grp.transform("mean")
            std_ = grp.transform("std").replace(0, np.nan)
            out[col] = (out[col] - mean_) / std_

        out[feature_cols] = out[feature_cols].fillna(0.0)
        return out

    @staticmethod
    def standard_scale(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = train_df.copy()
        test_df = test_df.copy()

        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        return train_df, test_df

    def apply_preprocessing(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = self.clip_by_train_quantile(train_df, test_df, feature_cols)

        if self.preprocess_mode == "zscore":
            train_df = self.cross_sectional_zscore(train_df, feature_cols, date_col="date")
            test_df = self.cross_sectional_zscore(test_df, feature_cols, date_col="date")
        elif self.preprocess_mode == "standard":
            train_df, test_df = self.standard_scale(train_df, test_df, feature_cols)

        return train_df, test_df

    def build_bundle(self) -> DatasetBundle:
        print(f"[PipelineStandard] preprocess_mode = {self.preprocess_mode}")

        df = self.load_data()
        print("[1/5] Loaded:", df.shape)

        df = self.build_target(df)
        df = self.build_delta_features(df)

        feature_cols = self.get_feature_cols()
        df = self.select_columns(df, feature_cols)
        print("[2/5] Selected:", df.shape)

        train_df, test_df = self.split_train_test(df)
        print("[3/5] Train/Test:", train_df.shape, test_df.shape)

        train_df, test_df = self.apply_preprocessing(train_df, test_df, feature_cols)
        print("[4/5] Preprocessing done.")

        print("[5/5] Bundle ready.")

        return DatasetBundle(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=self.target_col,
            id_cols=self.id_cols,
            preprocess_mode=self.preprocess_mode,
        )