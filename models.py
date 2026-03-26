from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from lightgbm import LGBMClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
    CatBoostClassifier = None

from config import LIGHTGBM_PARAMS


# =========================
# 默认参数
# =========================
DEFAULT_LOGISTIC_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 5,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 1024,
    "learning_rate_init": 1e-3,
    "max_iter": 50,
    "random_state": 42,
    "early_stopping": True,
    "validation_fraction": 0.1,
}

DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

DEFAULT_CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": 42,
    "verbose": False,
}


# =========================
# 模型构建
# =========================
def build_model(model_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    根据模型名创建模型实例。

    支持:
    - logistic
    - random_forest
    - lightgbm
    - mlp
    - xgboost
    - catboost
    """
    params = params or {}
    model_name = model_name.lower()

    if model_name == "logistic":
        final_params = DEFAULT_LOGISTIC_PARAMS.copy()
        final_params.update(params)
        return LogisticRegression(**final_params)

    if model_name == "random_forest":
        final_params = DEFAULT_RF_PARAMS.copy()
        final_params.update(params)
        return RandomForestClassifier(**final_params)

    if model_name == "lightgbm":
        final_params = LIGHTGBM_PARAMS.copy()
        final_params.update(params)
        return LGBMClassifier(**final_params)

    if model_name == "mlp":
        final_params = DEFAULT_MLP_PARAMS.copy()
        final_params.update(params)
        return MLPClassifier(**final_params)

    if model_name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError(
                "xgboost is not installed. Please install it first, e.g. pip install xgboost"
            )
        final_params = DEFAULT_XGB_PARAMS.copy()
        final_params.update(params)
        return XGBClassifier(**final_params)

    if model_name == "catboost":
        if not HAS_CATBOOST:
            raise ImportError(
                "catboost is not installed. Please install it first, e.g. pip install catboost"
            )
        final_params = DEFAULT_CATBOOST_PARAMS.copy()
        final_params.update(params)
        return CatBoostClassifier(**final_params)

    raise ValueError(f"Unsupported model_name: {model_name}")


# =========================
# 统一训练器
# =========================
@dataclass
class ModelTrainer:
    model_name: str
    model_params: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.model_params = self.model_params or {}
        self.model = build_model(self.model_name, self.model_params)

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> None:
        X_train = train_df[feature_cols]
        y_train = train_df[target_col].astype(int)

        self.model.fit(X_train, y_train)

    def predict_proba(
        self,
        test_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        X_test = test_df[feature_cols]

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(f"Model {self.model_name} does not support predict_proba().")

        probs = self.model.predict_proba(X_test)

        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]

        raise ValueError(
            f"Unexpected predict_proba output shape for model {self.model_name}: {probs.shape}"
        )

    def get_feature_importance(self, feature_cols: list[str]) -> Optional[pd.DataFrame]:
        """
        返回特征重要性:
        - 树模型: feature_importances_
        - Logistic: |coef|
        - 其他不支持则返回 None
        """
        # 树模型
        if hasattr(self.model, "feature_importances_"):
            return pd.DataFrame({
                "feature": feature_cols,
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False)

        # 线性模型
        if hasattr(self.model, "coef_"):
            coef = np.ravel(self.model.coef_)
            return pd.DataFrame({
                "feature": feature_cols,
                "importance": np.abs(coef),
                "coef": coef,
            }).sort_values("importance", ascending=False)

        return None