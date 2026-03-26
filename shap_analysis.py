from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from config import DATA_PATH
from data_pipeline import DataPipeline
from models import ModelTrainer


# =========================
# 1. 配置
# =========================
MODEL_NAME = "xgboost"
MODEL_PARAMS = {}   # 如果后面你要改 xgboost 参数，可以在这里补

PREPROCESS_MODE = "raw"   # 主 SHAP 就固定用 raw
TOP_FRAC = 0.01           # 重点分析 Top 1%

# SHAP 抽样规模（避免太慢）
TEST_SAMPLE_N = 20000
YEAR_SAMPLE_N = 10000
TOP_SAMPLE_MAX_N = 5000

OUTPUT_DIR = Path("shap_outputs_xgboost_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 工具函数
# =========================
def sample_df(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state).copy()


def get_top_frac_df(df: pd.DataFrame, score_col: str, top_frac: float) -> pd.DataFrame:
    rows = []
    for dt, g in df.groupby("date"):
        g = g.sort_values(score_col, ascending=False)
        k = max(1, int(len(g) * top_frac))
        rows.append(g.head(k))
    return pd.concat(rows, axis=0, ignore_index=False)


def save_bar_plot(series: pd.Series, title: str, out_path: Path, top_n: int = 20):
    plot_data = series.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plot_data.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel("mean(|SHAP value|)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_shap_values_tree(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 兼容不同版本 shap / xgboost
    if isinstance(shap_values, list):
        shap_array = shap_values[1]
    else:
        shap_array = shap_values

    if hasattr(shap_array, "values"):
        shap_array = shap_array.values

    return explainer, shap_array


# =========================
# 3. 主流程
# =========================
def main():
    print("Loading data pipeline...")
    pipeline = DataPipeline(data_path=DATA_PATH)
    bundle = pipeline.build_bundle()

    print("Train shape:", bundle.train_df.shape)
    print("Test shape :", bundle.test_df.shape)
    print("Num features:", len(bundle.feature_cols))

    # =========================
    # 3.1 训练 XGBoost + raw
    # =========================
    print("\nTraining model...")
    trainer = ModelTrainer(model_name=MODEL_NAME, model_params=MODEL_PARAMS)
    trainer.fit(
        train_df=bundle.train_df,
        feature_cols=bundle.feature_cols,
        target_col=bundle.target_col,
    )

    pred_prob = trainer.predict_proba(
        test_df=bundle.test_df,
        feature_cols=bundle.feature_cols,
    )

    pred_df = bundle.test_df[bundle.id_cols + [bundle.target_col]].copy()
    pred_df["pred_up_prob"] = pred_prob
    pred_df["year"] = pd.to_datetime(pred_df["date"]).dt.year

    # =========================
    # 3.2 全局 SHAP（测试集抽样）
    # =========================
    print("\nComputing GLOBAL SHAP...")
    global_sample_df = sample_df(bundle.test_df, TEST_SAMPLE_N)
    X_global = global_sample_df[bundle.feature_cols]

    explainer, shap_array_global = compute_shap_values_tree(trainer.model, X_global)
    shap_df_global = pd.DataFrame(shap_array_global, columns=bundle.feature_cols, index=X_global.index)

    global_mean_abs = shap_df_global.abs().mean().sort_values(ascending=False)

    print("\n==============================")
    print("GLOBAL SHAP TOP 20")
    print("==============================")
    print(global_mean_abs.head(20))

    global_mean_abs.to_csv(
        OUTPUT_DIR / "global_shap_importance.csv",
        header=["mean_abs_shap"],
        encoding="utf-8-sig",
    )

    save_bar_plot(
        global_mean_abs,
        title="XGBoost Raw - Global SHAP Importance",
        out_path=OUTPUT_DIR / "global_shap_bar.png",
        top_n=20,
    )

    # summary plot
    try:
        shap.summary_plot(
            shap_array_global,
            X_global,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "global_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"global summary_plot failed: {e}")

    # =========================
    # 3.3 分年份 SHAP
    # =========================
    print("\nComputing YEARLY SHAP...")
    yearly_rows = []

    test_df_with_year = bundle.test_df.copy()
    test_df_with_year["year"] = pd.to_datetime(test_df_with_year["date"]).dt.year

    for yr in sorted(test_df_with_year["year"].dropna().unique()):
        year_df = test_df_with_year[test_df_with_year["year"] == yr].copy()
        year_df = sample_df(year_df, YEAR_SAMPLE_N)

        if len(year_df) == 0:
            continue

        X_year = year_df[bundle.feature_cols]
        _, shap_array_year = compute_shap_values_tree(trainer.model, X_year)
        shap_df_year = pd.DataFrame(shap_array_year, columns=bundle.feature_cols, index=X_year.index)

        year_mean_abs = shap_df_year.abs().mean().sort_values(ascending=False)

        print(f"\n==============================")
        print(f"YEAR {yr} SHAP TOP 10")
        print("==============================")
        print(year_mean_abs.head(10))

        tmp = pd.DataFrame({
            "year": yr,
            "feature": year_mean_abs.index,
            "mean_abs_shap": year_mean_abs.values,
        })
        yearly_rows.append(tmp)

        save_bar_plot(
            year_mean_abs,
            title=f"XGBoost Raw - SHAP Importance ({yr})",
            out_path=OUTPUT_DIR / f"year_{yr}_shap_bar.png",
            top_n=15,
        )

    yearly_shap_df = pd.concat(yearly_rows, ignore_index=True)
    yearly_shap_df.to_csv(
        OUTPUT_DIR / "yearly_shap_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =========================
    # 3.4 Top 1% 样本 SHAP
    # =========================
    print("\nComputing TOP 1% SHAP...")
    top_df = get_top_frac_df(pred_df, score_col="pred_up_prob", top_frac=TOP_FRAC)
    top_df = bundle.test_df.loc[top_df.index].copy()

    if len(top_df) > TOP_SAMPLE_MAX_N:
        top_df = sample_df(top_df, TOP_SAMPLE_MAX_N)

    X_top = top_df[bundle.feature_cols]
    _, shap_array_top = compute_shap_values_tree(trainer.model, X_top)
    shap_df_top = pd.DataFrame(shap_array_top, columns=bundle.feature_cols, index=X_top.index)

    top_mean_abs = shap_df_top.abs().mean().sort_values(ascending=False)

    print("\n==============================")
    print("TOP 1% SHAP TOP 20")
    print("==============================")
    print(top_mean_abs.head(20))

    top_mean_abs.to_csv(
        OUTPUT_DIR / "top1_shap_importance.csv",
        header=["mean_abs_shap"],
        encoding="utf-8-sig",
    )

    save_bar_plot(
        top_mean_abs,
        title="XGBoost Raw - Top 1% SHAP Importance",
        out_path=OUTPUT_DIR / "top1_shap_bar.png",
        top_n=20,
    )

    try:
        shap.summary_plot(
            shap_array_top,
            X_top,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top1_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"top1 summary_plot failed: {e}")

    print("\nAll SHAP outputs saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()