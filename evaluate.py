from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from config import TOP_K_LIST


@dataclass
class EvaluationResult:
    summary: Dict[str, float]
    topk_details: Dict[str, pd.DataFrame]
    bucket_df: pd.DataFrame
    prediction_df: pd.DataFrame


class Evaluator:
    """
    二分类统一评估器：
    - Accuracy
    - AUC
    - Top K hit rate
    - Top K future return
    - Probability bucket
    """

    def __init__(self, top_k_list: Optional[List[float]] = None) -> None:
        self.top_k_list = top_k_list if top_k_list is not None else TOP_K_LIST

    @staticmethod
    def build_prediction_df(
        test_df: pd.DataFrame,
        pred_prob: np.ndarray,
        target_col: str,
        id_cols: List[str],
    ) -> pd.DataFrame:
        out = test_df[id_cols + [target_col]].copy()
        out["pred_up_prob"] = pred_prob
        out["pred_label"] = (out["pred_up_prob"] >= 0.5).astype(int)
        return out

    @staticmethod
    def calc_topk_metrics(
        df_in: pd.DataFrame,
        score_col: str,
        actual_col: str,
        top_frac: float,
    ) -> pd.DataFrame:
        rows = []

        for dt, g in df_in.groupby("date"):
            g = g.sort_values(score_col, ascending=False).copy()
            n = len(g)
            if n == 0:
                continue

            k = max(1, int(n * top_frac))
            top_g = g.head(k)

            rows.append({
                "date": dt,
                "n": n,
                "k": k,
                "hit_rate": top_g[actual_col].mean(),
                "future_return": top_g["r_future_5"].mean(),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def calc_probability_bucket(
        df_in: pd.DataFrame,
        score_col: str = "pred_up_prob",
        actual_col: str = "regime_binary",
        q: int = 10,
    ) -> pd.DataFrame:
        tmp = df_in.copy()
        tmp["prob_bin"] = pd.qcut(tmp[score_col], q=q, labels=False, duplicates="drop")

        bucket = tmp.groupby("prob_bin").agg(
            avg_up_prob=(score_col, "mean"),
            actual_up_rate=(actual_col, "mean"),
            avg_future_ret=("r_future_5", "mean"),
            count=("ticker", "count")
        ).reset_index()

        return bucket

    def evaluate(
        self,
        test_df: pd.DataFrame,
        pred_prob: np.ndarray,
        target_col: str,
        id_cols: List[str],
    ) -> EvaluationResult:
        prediction_df = self.build_prediction_df(
            test_df=test_df,
            pred_prob=pred_prob,
            target_col=target_col,
            id_cols=id_cols,
        )

        y_true = prediction_df[target_col].astype(int)
        y_pred = prediction_df["pred_label"].astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, prediction_df["pred_up_prob"])

        topk_details: Dict[str, pd.DataFrame] = {}
        summary: Dict[str, float] = {
            "accuracy": accuracy,
            "auc": auc,
            "market_mean_future_return": prediction_df["r_future_5"].mean(),
            "market_up_ratio": prediction_df[target_col].mean(),
        }

        for frac in self.top_k_list:
            k_name = f"top{int(frac * 100)}"

            topk_df = self.calc_topk_metrics(
                df_in=prediction_df,
                score_col="pred_up_prob",
                actual_col=target_col,
                top_frac=frac,
            )
            topk_details[k_name] = topk_df

            summary[f"{k_name}_hit_rate"] = topk_df["hit_rate"].mean()
            summary[f"{k_name}_future_return"] = topk_df["future_return"].mean()

        bucket_df = self.calc_probability_bucket(
            df_in=prediction_df,
            score_col="pred_up_prob",
            actual_col=target_col,
            q=10,
        )

        summary["top_bin_actual_up_rate"] = bucket_df.iloc[-1]["actual_up_rate"]
        summary["top_bin_avg_future_ret"] = bucket_df.iloc[-1]["avg_future_ret"]

        return EvaluationResult(
            summary=summary,
            topk_details=topk_details,
            bucket_df=bucket_df,
            prediction_df=prediction_df,
        )

    @staticmethod
    def print_summary(model_name: str, summary: Dict[str, float]) -> None:
        print("\n==============================")
        print(f"MODEL: {model_name}")
        print("==============================")
        for k, v in summary.items():
            if isinstance(v, (float, int, np.floating)):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")