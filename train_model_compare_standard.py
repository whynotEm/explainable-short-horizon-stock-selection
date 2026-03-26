from __future__ import annotations

import pandas as pd

from config import DATA_PATH
from data_pipeline_standard import DataPipelineStandard
from models import ModelTrainer
from evaluate import Evaluator


def run_model_compare_standard() -> None:
    pipeline = DataPipelineStandard(data_path=DATA_PATH)
    bundle = pipeline.build_bundle()

    print("Train shape:", bundle.train_df.shape)
    print("Test shape :", bundle.test_df.shape)
    print("Target col :", bundle.target_col)
    print("Num features:", len(bundle.feature_cols))

    print("\nTrain label distribution:")
    print(bundle.train_df[bundle.target_col].value_counts(normalize=True))

    print("\nTest label distribution:")
    print(bundle.test_df[bundle.target_col].value_counts(normalize=True))

    evaluator = Evaluator()
    all_summaries = []

    model_specs = [
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

    for spec in model_specs:
        display_name = spec["display_name"]
        model_name = spec["model_name"]
        params = spec["params"]

        print("\n========================================")
        print(f"Running model: {display_name} [standard]")
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

            evaluator.print_summary(f"{display_name} [standard]", result.summary)

            row = {
                "model_name": display_name,
                "preprocess_mode": "standard",
            }
            row.update(result.summary)
            all_summaries.append(row)

        except Exception as e:
            print(f"[ERROR] {display_name} [standard] failed: {e}")

    summary_df = pd.DataFrame(all_summaries).sort_values(
        by=["top5_future_return", "auc"],
        ascending=False,
    )

    print("\n========================================")
    print("STANDARD MODEL SUMMARY")
    print("========================================")
    print(summary_df)

    out_path = "model_compare_summary_standard.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    run_model_compare_standard()