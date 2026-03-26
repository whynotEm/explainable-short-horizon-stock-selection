import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. 路径
# =========================
csv_path = r"D:\project1\pythonProject\stock\model_compare_summary_all.csv"
out_dir = Path(r"D:\project1\pythonProject\stock\figures")
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(csv_path)
print("Columns:")
print(df.columns.tolist())
print("\nHead:")
print(df.head())

# =========================
# 3. 顺序设定
# =========================
preprocess_order = ["raw", "zscore", "standard"]
model_order = [
    "lightgbm",
    "lightgbm_shallow",
    "lightgbm_conservative",
    "xgboost",
    "catboost",
    "logistic",
    "mlp",
]

if "preprocess_mode" in df.columns:
    df["preprocess_mode"] = pd.Categorical(
        df["preprocess_mode"],
        categories=preprocess_order,
        ordered=True
    )

if "model_name" in df.columns:
    df["model_name"] = pd.Categorical(
        df["model_name"],
        categories=model_order,
        ordered=True
    )

df = df.sort_values(["model_name", "preprocess_mode"]).reset_index(drop=True)

# =========================
# 4. 自动识别 accuracy 列
# =========================
accuracy_col = None
for candidate in ["accuracy", "acc"]:
    if candidate in df.columns:
        accuracy_col = candidate
        break

print(f"\nDetected accuracy column: {accuracy_col}")

# =========================
# 5. 工具函数
# =========================
def has_required_cols(data, cols):
    missing = [c for c in cols if c not in data.columns]
    if missing:
        print(f"Skip because missing columns: {missing}")
        return False
    return True


def save_bar_by_metric(data, metric, title, filename, ylabel=None, ascending=False):
    """
    单一指标总体排序图：
    x = model + preprocess
    y = metric
    """
    if not has_required_cols(data, ["model_name", "preprocess_mode", metric]):
        return

    plot_df = data.copy().sort_values(metric, ascending=ascending)

    plt.figure(figsize=(14, 7))
    x_labels = [
        f"{m}\n({p})"
        for m, p in zip(
            plot_df["model_name"].astype(str),
            plot_df["preprocess_mode"].astype(str)
        )
    ]
    y = plot_df[metric]

    plt.bar(x_labels, y)
    plt.title(title)
    plt.ylabel(ylabel if ylabel else metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

    print(f"Saved: {filename}")


def save_grouped_bar(data, metric, title, filename, ylabel=None):
    """
    分组柱状图：
    index = model_name
    columns = preprocess_mode
    values = metric
    """
    if not has_required_cols(data, ["model_name", "preprocess_mode", metric]):
        return

    pivot_df = data.pivot(
        index="model_name",
        columns="preprocess_mode",
        values=metric
    )

    pivot_df = pivot_df.reindex(model_order)
    existing_preprocess_cols = [c for c in preprocess_order if c in pivot_df.columns]
    pivot_df = pivot_df[existing_preprocess_cols]

    ax = pivot_df.plot(kind="bar", figsize=(14, 7))
    ax.set_title(title)
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.set_xlabel("model_name")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

    print(f"Saved: {filename}")


def save_heatmap(data, metrics, title, filename, normalize_within_model=False):
    """
    热力图：
    行 = model_name + preprocess_mode
    列 = metrics

    normalize_within_model=False:
        直接展示原始值

    normalize_within_model=True:
        对每个 model_name 内部、每个 metric 做 min-max 标准化
        用于比较同一个模型下 raw / zscore / standard 谁更强
    """
    needed_cols = ["model_name", "preprocess_mode"] + metrics
    if not has_required_cols(data, needed_cols):
        return

    plot_df = data.copy()
    plot_df["row_label"] = (
        plot_df["model_name"].astype(str) + " (" + plot_df["preprocess_mode"].astype(str) + ")"
    )

    # 保证顺序
    label_order = []
    for m in model_order:
        for p in preprocess_order:
            label_order.append(f"{m} ({p})")

    plot_df = plot_df.set_index("row_label")
    plot_df = plot_df.reindex([lab for lab in label_order if lab in plot_df.index])

    heat_df = plot_df[metrics].astype(float).copy()

    if normalize_within_model:
        # 按模型内部比较 raw / zscore / standard
        tmp = plot_df.reset_index()
        norm_rows = []

        for model in model_order:
            sub = tmp[tmp["model_name"].astype(str) == model].copy()
            if sub.empty:
                continue

            sub = sub.set_index("row_label")
            sub_metrics = sub[metrics].astype(float).copy()

            for col in metrics:
                col_min = sub_metrics[col].min()
                col_max = sub_metrics[col].max()
                if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
                    sub_metrics[col] = 0.5
                else:
                    sub_metrics[col] = (sub_metrics[col] - col_min) / (col_max - col_min)

            norm_rows.append(sub_metrics)

        if norm_rows:
            heat_df = pd.concat(norm_rows, axis=0)
            heat_df = heat_df.reindex([lab for lab in label_order if lab in heat_df.index])

    # 动态调整画布大小
    fig_w = max(10, len(metrics) * 1.3)
    fig_h = max(8, len(heat_df) * 0.45)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(heat_df.values, aspect="auto")

    plt.title(title)
    plt.xticks(range(len(metrics)), metrics, rotation=30, ha="right")
    plt.yticks(range(len(heat_df.index)), heat_df.index)

    # 数值标注
    for i in range(heat_df.shape[0]):
        for j in range(heat_df.shape[1]):
            val = heat_df.iloc[i, j]
            if pd.notna(val):
                if normalize_within_model:
                    text_str = f"{val:.2f}"
                else:
                    text_str = f"{val:.3f}"
                plt.text(j, i, text_str, ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im)
    cbar.set_label("Normalized Score" if normalize_within_model else "Metric Value")

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


# =========================
# 6. 主图：以 Top1 为核心
# =========================
save_bar_by_metric(
    df,
    metric="top1_future_return",
    title="Top 1% Future Return by Model and Preprocessing",
    filename="bar_top1_future_return_all.png",
    ylabel="Top 1% Future Return",
)

save_bar_by_metric(
    df,
    metric="top1_hit_rate",
    title="Top 1% Hit Rate by Model and Preprocessing",
    filename="bar_top1_hit_rate_all.png",
    ylabel="Top 1% Hit Rate",
)

# =========================
# 7. 稳健性补充：Top3 / Top5 / AUC / Top Bin
# =========================
save_bar_by_metric(
    df,
    metric="top3_future_return",
    title="Top 3% Future Return by Model and Preprocessing",
    filename="bar_top3_future_return_all.png",
    ylabel="Top 3% Future Return",
)

save_bar_by_metric(
    df,
    metric="top5_future_return",
    title="Top 5% Future Return by Model and Preprocessing",
    filename="bar_top5_future_return_all.png",
    ylabel="Top 5% Future Return",
)

save_bar_by_metric(
    df,
    metric="top3_hit_rate",
    title="Top 3% Hit Rate by Model and Preprocessing",
    filename="bar_top3_hit_rate_all.png",
    ylabel="Top 3% Hit Rate",
)

save_bar_by_metric(
    df,
    metric="top5_hit_rate",
    title="Top 5% Hit Rate by Model and Preprocessing",
    filename="bar_top5_hit_rate_all.png",
    ylabel="Top 5% Hit Rate",
)

save_bar_by_metric(
    df,
    metric="auc",
    title="AUC by Model and Preprocessing",
    filename="bar_auc_all.png",
    ylabel="AUC",
)

save_bar_by_metric(
    df,
    metric="top_bin_avg_future_ret",
    title="Top Bin Average Future Return by Model and Preprocessing",
    filename="bar_top_bin_avg_future_ret_all.png",
    ylabel="Top Bin Avg Future Return",
)

save_bar_by_metric(
    df,
    metric="top_bin_actual_up_rate",
    title="Top Bin Actual Up Rate by Model and Preprocessing",
    filename="bar_top_bin_actual_up_rate_all.png",
    ylabel="Top Bin Actual Up Rate",
)

if accuracy_col is not None:
    save_bar_by_metric(
        df,
        metric=accuracy_col,
        title=f"{accuracy_col.upper()} by Model and Preprocessing",
        filename=f"bar_{accuracy_col}_all.png",
        ylabel=accuracy_col.upper(),
    )

# =========================
# 8. 分组图：看预处理方法影响
# =========================
save_grouped_bar(
    df,
    metric="top1_future_return",
    title="Top 1% Future Return: Raw vs Z-score vs Standard",
    filename="grouped_top1_future_return.png",
    ylabel="Top 1% Future Return",
)

save_grouped_bar(
    df,
    metric="top1_hit_rate",
    title="Top 1% Hit Rate: Raw vs Z-score vs Standard",
    filename="grouped_top1_hit_rate.png",
    ylabel="Top 1% Hit Rate",
)

save_grouped_bar(
    df,
    metric="top3_future_return",
    title="Top 3% Future Return: Raw vs Z-score vs Standard",
    filename="grouped_top3_future_return.png",
    ylabel="Top 3% Future Return",
)

save_grouped_bar(
    df,
    metric="top3_hit_rate",
    title="Top 3% Hit Rate: Raw vs Z-score vs Standard",
    filename="grouped_top3_hit_rate.png",
    ylabel="Top 3% Hit Rate",
)

save_grouped_bar(
    df,
    metric="top5_future_return",
    title="Top 5% Future Return: Raw vs Z-score vs Standard",
    filename="grouped_top5_future_return.png",
    ylabel="Top 5% Future Return",
)

save_grouped_bar(
    df,
    metric="top5_hit_rate",
    title="Top 5% Hit Rate: Raw vs Z-score vs Standard",
    filename="grouped_top5_hit_rate.png",
    ylabel="Top 5% Hit Rate",
)

save_grouped_bar(
    df,
    metric="auc",
    title="AUC: Raw vs Z-score vs Standard",
    filename="grouped_auc.png",
    ylabel="AUC",
)

save_grouped_bar(
    df,
    metric="top_bin_avg_future_ret",
    title="Top Bin Avg Future Return: Raw vs Z-score vs Standard",
    filename="grouped_top_bin_avg_future_ret.png",
    ylabel="Top Bin Avg Future Return",
)

if accuracy_col is not None:
    save_grouped_bar(
        df,
        metric=accuracy_col,
        title=f"{accuracy_col.upper()}: Raw vs Z-score vs Standard",
        filename=f"grouped_{accuracy_col}.png",
        ylabel=accuracy_col.upper(),
    )

# =========================
# 9. 单独给树模型和尺度敏感模型出图
# =========================
if "model_group" in df.columns:
    for group_name in df["model_group"].dropna().unique():
        sub = df[df["model_group"] == group_name].copy()

        save_grouped_bar(
            sub,
            metric="top1_future_return",
            title=f"{group_name.capitalize()} Models: Top 1% Future Return",
            filename=f"grouped_top1_future_return_{group_name}.png",
            ylabel="Top 1% Future Return",
        )

        save_grouped_bar(
            sub,
            metric="top1_hit_rate",
            title=f"{group_name.capitalize()} Models: Top 1% Hit Rate",
            filename=f"grouped_top1_hit_rate_{group_name}.png",
            ylabel="Top 1% Hit Rate",
        )

        save_grouped_bar(
            sub,
            metric="top5_future_return",
            title=f"{group_name.capitalize()} Models: Top 5% Future Return",
            filename=f"grouped_top5_future_return_{group_name}.png",
            ylabel="Top 5% Future Return",
        )

        save_grouped_bar(
            sub,
            metric="top5_hit_rate",
            title=f"{group_name.capitalize()} Models: Top 5% Hit Rate",
            filename=f"grouped_top5_hit_rate_{group_name}.png",
            ylabel="Top 5% Hit Rate",
        )

        save_grouped_bar(
            sub,
            metric="auc",
            title=f"{group_name.capitalize()} Models: AUC",
            filename=f"grouped_auc_{group_name}.png",
            ylabel="AUC",
        )

        if accuracy_col is not None:
            save_grouped_bar(
                sub,
                metric=accuracy_col,
                title=f"{group_name.capitalize()} Models: {accuracy_col.upper()}",
                filename=f"grouped_{accuracy_col}_{group_name}.png",
                ylabel=accuracy_col.upper(),
            )

# =========================
# 10. 热力图（重点新增）
# =========================
heatmap_metrics = []

if accuracy_col is not None:
    heatmap_metrics.append(accuracy_col)

heatmap_metrics += [
    "auc",
    "top1_hit_rate",
    "top3_hit_rate",
    "top5_hit_rate",
    "top1_future_return",
    "top3_future_return",
    "top5_future_return",
]

heatmap_metrics = [m for m in heatmap_metrics if m in df.columns]

# 原始值热力图：看绝对表现
save_heatmap(
    df,
    metrics=heatmap_metrics,
    title="Model × Preprocessing × Metrics (Raw Values)",
    filename="heatmap_model_preprocess_metrics_raw.png",
    normalize_within_model=False,
)

# 分模型标准化热力图：看同一模型内部 raw/zscore/standard 谁更强
save_heatmap(
    df,
    metrics=heatmap_metrics,
    title="Model × Preprocessing × Metrics (Within-Model Normalized)",
    filename="heatmap_model_preprocess_metrics_normalized.png",
    normalize_within_model=True,
)

# =========================
# 11. 输出最优结果表
# =========================
best_cols = [
    "model_name",
    "model_group",
    "preprocess_mode",
    "auc",
    "top1_hit_rate",
    "top3_hit_rate",
    "top5_hit_rate",
    "top1_future_return",
    "top3_future_return",
    "top5_future_return",
    "top_bin_actual_up_rate",
    "top_bin_avg_future_ret",
]

if accuracy_col is not None:
    best_cols.insert(3, accuracy_col)

best_cols = [c for c in best_cols if c in df.columns]

best_df = df[best_cols].sort_values(
    ["top1_future_return", "top3_future_return", "top5_future_return", "auc"],
    ascending=False
).reset_index(drop=True)

best_df.to_csv(
    out_dir / "best_model_ranking_top1_first.csv",
    index=False,
    encoding="utf-8-sig"
)

best_df_top5 = df[best_cols].sort_values(
    ["top5_future_return", "top3_future_return", "top1_future_return", "auc"],
    ascending=False
).reset_index(drop=True)

best_df_top5.to_csv(
    out_dir / "best_model_ranking_top5_first.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n图和排名表已保存到：", out_dir)
print("\nTop10 by Top1-first ranking:")
print(best_df.head(10))