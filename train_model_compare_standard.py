from __future__ import annotations

import pandas as pd

from config import DATA_PATH
from data_pipeline_standard import DataPipelineStandard
from models import ModelTrainer
from evaluate import Evaluator


TREE_MODEL_SPECS = [
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
]

SCALE_SENSITIVE_MODEL_SPECS = [
    {
        "display_name": "logistic",
        "model_name": "logistic",
        "params": {},
    },
    {
        "display_name": "mlp",
        "model_name": "mlp",
        "params": {},
    },
]


def run_one_block(preprocess_mode: str, model_specs: list[dict], model_group: str, evaluator: Evaluator):
    print("\n==================================================")
    print(f"{model_group.upper()} MODELS | PREPROCESS = {preprocess_mode}")
    print("==================================================")

    pipeline = DataPipelineStandard(
        data_path=DATA_PATH,
        preprocess_mode=preprocess_mode,
    )
    bundle = pipeline.build_bundle()

    print("Train shape:", bundle.train_df.shape)
    print("Test shape :", bundle.test_df.shape)
    print("Target col :", bundle.target_col)
    print("Num features:", len(bundle.feature_cols))

    print("\nTrain label distribution:")
    print(bundle.train_df[bundle.target_col].value_counts(normalize=True))

    print("\nTest label distribution:")
    print(bundle.test_df[bundle.target_col].value_counts(normalize=True))

    rows = []

    for spec in model_specs:
        display_name = spec["display_name"]
        model_name = spec["model_name"]
        params = spec["params"]

        print("\n----------------------------------------")
        print(f"Running: {display_name} [{preprocess_mode}]")
        print("----------------------------------------")

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

            evaluator.print_summary(f"{display_name} [{preprocess_mode}]", result.summary)

            row = {
                "model_name": display_name,
                "model_group": model_group,
                "preprocess_mode": preprocess_mode,
            }
            row.update(result.summary)
            rows.append(row)

        except Exception as e:
            print(f"[ERROR] {display_name} [{preprocess_mode}] failed: {e}")

    return rows


def run_model_compare_standard() -> None:
    evaluator = Evaluator()
    all_rows = []

    # 1) 树模型：只做 zscore
    all_rows.extend(
        run_one_block(
            preprocess_mode="zscore",
            model_specs=TREE_MODEL_SPECS,
            model_group="tree",
            evaluator=evaluator,
        )
    )

    # 2) 尺度敏感模型：做 zscore
    all_rows.extend(
        run_one_block(
            preprocess_mode="zscore",
            model_specs=SCALE_SENSITIVE_MODEL_SPECS,
            model_group="scale_sensitive",
            evaluator=evaluator,
        )
    )

    # 3) 尺度敏感模型：做 standard
    all_rows.extend(
        run_one_block(
            preprocess_mode="standard",
            model_specs=SCALE_SENSITIVE_MODEL_SPECS,
            model_group="scale_sensitive",
            evaluator=evaluator,
        )
    )

    summary_df = pd.DataFrame(all_rows).sort_values(
        by=["preprocess_mode", "top5_future_return", "auc"],
        ascending=[True, False, False],
    )

    print("\n==================================================")
    print("STANDARD / ZSCORE EXPERIMENT SUMMARY")
    print("==================================================")
    print(summary_df)

    out_path = "model_compare_summary_standard_and_zscore.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    run_model_compare_standard()