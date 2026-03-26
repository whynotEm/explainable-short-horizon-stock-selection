from __future__ import annotations

import pandas as pd

from config import DATA_PATH
from data_pipeline import DataPipeline
from models import ModelTrainer
from evaluate import Evaluator


def run_model_compare() -> None:
    # =========================
    # 1. 数据准备
    # =========================
    pipeline = DataPipeline(data_path=DATA_PATH)
    bundle = pipeline.build_bundle()

    print("Train shape:", bundle.train_df.shape)
    print("Test shape :", bundle.test_df.shape)
    print("Target col :", bundle.target_col)
    print("Num features:", len(bundle.feature_cols))

    print("\nTrain label distribution:")
    print(bundle.train_df[bundle.target_col].value_counts(normalize=True))

    print("\nTest label distribution:")
    print(bundle.test_df[bundle.target_col].value_counts(normalize=True))

    # =========================
    # 2. 评估器
    # =========================
    evaluator = Evaluator()

    # =========================
    # 3. 模型规格
    # =========================
    model_specs = [
        {
            "display_name": "logistic",
            "model_name": "logistic",
            "params": {},
        },
        {
            "display_name": "lightgbm",
            "model_name": "lightgbm",
            "params": {},
        },
        {
            "display_name": "lightgbm_shallow",
            "model_name": "lightgbm",
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 4,
                "num_leaves": 15,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
        {
            "display_name": "lightgbm_conservative",
            "model_name": "lightgbm",
            "params": {
                "n_estimators": 800,
                "learning_rate": 0.02,
                "max_depth": -1,
                "num_leaves": 31,
                "min_child_samples": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
        {
            "display_name": "xgboost",
            "model_name": "xgboost",
            "params": {},
        },
        {
            "display_name": "catboost",
            "model_name": "catboost",
            "params": {},
        },

        # =========================
        # 可选模型：默认先注释
        # =========================

        # {
        #     "display_name": "mlp",
        #     "model_name": "mlp",
        #     "params": {},
        # },

        # {
        #     "display_name": "random_forest",
        #     "model_name": "random_forest",
        #     "params": {
        #         "n_estimators": 50,
        #         "max_depth": 8,
        #         "min_samples_leaf": 50,
        #         "class_weight": None,
        #         "random_state": 42,
        #         "n_jobs": 2,
        #     },
        # },
    ]

    all_summaries = []

    # =========================
    # 4. 逐个训练 + 评估
    # =========================
    for spec in model_specs:
        display_name = spec["display_name"]
        model_name = spec["model_name"]
        params = spec["params"]

        print("\n========================================")
        print(f"Running model: {display_name}")
        print("========================================")

        try:
            trainer = ModelTrainer(model_name=model_name, model_params=params)
            trainer.fit(
                train_df=bundle.train_df,
                feature_cols=bundle.feature_cols,
                target_col=bundle.target_col,
            )

            pred_prob = trainer.predict_proba(
                test_df=bundle.test_df,
                feature_cols=bundle.feature_cols,
            )

            result = evaluator.evaluate(
                test_df=bundle.test_df,
                pred_prob=pred_prob,
                target_col=bundle.target_col,
                id_cols=bundle.id_cols,
            )

            evaluator.print_summary(display_name, result.summary)

            print("\nProbability bucket:")
            print(result.bucket_df)

            feat_imp = trainer.get_feature_importance(bundle.feature_cols)
            if feat_imp is not None:
                print("\nTop feature importance:")
                print(feat_imp.head(10))

            row = {"model_name": display_name}
            row.update(result.summary)
            all_summaries.append(row)

        except Exception as e:
            print(f"[ERROR] Model {display_name} failed: {e}")

    # =========================
    # 5. 汇总表
    # =========================
    summary_df = pd.DataFrame(all_summaries).sort_values(
        by=["top5_future_return", "auc"],
        ascending=False,
    )

    print("\n========================================")
    print("MODEL COMPARISON SUMMARY")
    print("========================================")
    print(summary_df)

    out_path = "model_compare_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    run_model_compare()