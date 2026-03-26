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
print(df.columns.tolist())
print(df.head())

# 统一顺序
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
    df["preprocess_mode"] = pd.Categorical(df["preprocess_mode"], categories=preprocess_order, ordered=True)

if "model_name" in df.columns:
    df["model_name"] = pd.Categorical(df["model_name"], categories=model_order, ordered=True)

df = df.sort_values(["model_name", "preprocess_mode"]).reset_index(drop=True)

# =========================
# 3. 画图函数
# =========================
def save_bar_by_metric(data, metric, title, filename, ylabel=None):
    plot_df = data.copy().sort_values(metric, ascending=False)

    plt.figure(figsize=(12, 6))
    x = [f"{m}\n({p})" for m, p in zip(plot_df["model_name"].astype(str), plot_df["preprocess_mode"].astype(str))]
    y = plot_df[metric]

    plt.bar(x, y)
    plt.title(title)
    plt.ylabel(ylabel if ylabel else metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def save_grouped_bar(data, metric, title, filename, ylabel=None):
    pivot_df = data.pivot(index="model_name", columns="preprocess_mode", values=metric)

    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind="bar", figsize=(12, 6))
    plt.title(title)
    plt.ylabel(ylabel if ylabel else metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


# =========================
# 4. 最重要的总体图
# =========================
save_bar_by_metric(
    df,
    metric="top5_future_return",
    title="Top 5% Future Return by Model and Preprocessing",
    filename="bar_top5_future_return_all.png",
    ylabel="Top 5% Future Return",
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
    metric="top5_hit_rate",
    title="Top 5% Hit Rate by Model and Preprocessing",
    filename="bar_top5_hit_rate_all.png",
    ylabel="Top 5% Hit Rate",
)

# =========================
# 5. 按模型分组，看不同预处理影响
# =========================
save_grouped_bar(
    df,
    metric="top5_future_return",
    title="Top 5% Future Return: Raw vs Z-score vs Standard",
    filename="grouped_top5_future_return.png",
    ylabel="Top 5% Future Return",
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

# =========================
# 6. 单独给树模型和尺度敏感模型出图
# =========================
if "model_group" in df.columns:
    for group_name in df["model_group"].dropna().unique():
        sub = df[df["model_group"] == group_name].copy()

        save_grouped_bar(
            sub,
            metric="top5_future_return",
            title=f"{group_name.capitalize()} Models: Top 5% Future Return",
            filename=f"grouped_top5_future_return_{group_name}.png",
            ylabel="Top 5% Future Return",
        )

        save_grouped_bar(
            sub,
            metric="auc",
            title=f"{group_name.capitalize()} Models: AUC",
            filename=f"grouped_auc_{group_name}.png",
            ylabel="AUC",
        )

# =========================
# 7. 输出一份最优结果表
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

best_df = df[best_cols].sort_values(
    ["top5_future_return", "auc"],
    ascending=False
).reset_index(drop=True)

best_df.to_csv(out_dir / "best_model_ranking.csv", index=False, encoding="utf-8-sig")

print("图和排名表已保存到：", out_dir)
print(best_df.head(10))