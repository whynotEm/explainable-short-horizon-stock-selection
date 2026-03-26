# Explainable Short-Horizon Stock Selection

A machine learning framework for short-horizon stock selection, optimized for **Top-K return performance** and enhanced with **SHAP-based interpretability**.

---
## 📌 Project Overview

This project builds a machine learning pipeline to identify stocks with strong return potential over the next 5 trading days.

Instead of focusing on classification accuracy alone, this project emphasizes:

* Cross-sectional ranking
* Top-K return performance
* Model interpretability (SHAP)

---

## 🎯 Objective

Given a set of features for each stock on each trading day, predict whether the stock will achieve positive returns over the next 5 days.

**Target definition:**

```text
regime_binary = 1 if r_future_5 > 0.01 else 0
```

**Data split:**

* Train: before 2023
* Test: after 2023

---

## 🧠 Key Ideas

* Treat stock prediction as a **ranking problem**
* Focus on **Top-K return instead of accuracy**
* Compare multiple models and preprocessing methods
* Use SHAP to understand model behavior

---

## 📊 Dataset

The dataset used in this project contains **7,167,829 observations** across **3,876 Chinese A-share stocks**, covering the period from **2016-10-17 to 2026-01-19** (2,251 trading days).

Each row represents a **stock-day observation**, and the dataset is structured as a cross-sectional panel.

The dataset is not included in this repository.

Download from:

👉 [Releases → stock_shap dataset]

### Columns

Index([
'date', 'ticker', 'r_future_5', 'r_past_10',
'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d',
'momentum_change',
'ret_1d_minus_5d', 'ret_3d_minus_10d', 'ret_1d_minus_3d',
'roc_20',
'ema30_slope_vr', 'ema30_slope', 'ma30_slope',
'bias_60_vr', 'bias_60',
'board_rank_20d_pct', 'board_rs_20d',
'ema60_slope', 'ema90_slope', 'ema180_slope',
'ma60_slope', 'ma180_slope',
'macro_regime_3', 'micro_sentiment_ema5',
'trend60', 'breadth_mom',
'vol20', 'dispersion', 'high20_ratio',
'所属行业'
])

### Target

The prediction target is defined as:

```text
regime_binary = 1 if r_future_5 > 0.01 else 0
```
---

## 🧪 Feature Engineering

Feature construction follows a combination of **momentum, trend, and mean-reversion signals**.

### 1. Return-based Features

- `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`
- Capture short- and medium-term returns

---

### 2. Delta Features (automatically constructed)

The following features are generated in the data pipeline:

- `ret_1d_minus_5d = ret_1d - ret_5d`
- `ret_3d_minus_10d = ret_3d - ret_10d`
- `ret_1d_minus_3d = ret_1d - ret_3d`

These features capture **acceleration and trend change**.

---

### 3. Momentum Features

- `roc_20`
- `momentum_change`

Used to measure trend strength and continuation.

---

### 4. Trend Features

- EMA slopes:  
  `ema30_slope`, `ema60_slope`, `ema90_slope`, `ema180_slope`

- MA slopes:  
  `ma30_slope`, `ma60_slope`, `ma180_slope`

These features reflect **direction and strength of trends**.

---

### 5. Mean Reversion Features

- `bias_60`
- `bias_60_vr`

Capture price deviation from long-term averages.

---

### 6. Cross-sectional Features

- `board_rank_20d_pct`
- `board_rs_20d`

Measure **relative strength within the market**.

---

### 7. Market & Breadth Features

- `macro_regime_3`
- `trend60`
- `breadth_mom`
- `dispersion`
- `vol20`
- `high20_ratio`

Provide **market-wide context and regime information**.

---

### Final Feature Set

The final training feature set is selected from a predefined feature group in the pipeline:

- `BASELINE_PLUS_LONGER_TREND_RAW_FEATURES`

and processed through:

- Missing value filtering
- Time-based split
- Quantile clipping (to remove outliers)

## ⚙️ Pipeline

1. Data loading
2. Feature engineering
3. Label construction
4. Outlier clipping
5. Train-test split (time-based)
6. Preprocessing:

   * raw
   * z-score (cross-sectional)
   * standard scaling
7. Model training
8. Evaluation
9. SHAP analysis

---

## 🤖 Models

* Logistic Regression
* Random Forest
* LightGBM
* XGBoost
* CatBoost
* MLP

---

## 📈 Evaluation Metrics

* AUC
* Top 1% / 3% / 5% hit rate
* Top 1% / 3% / 5% future return ⭐
* Probability bucket performance

---

## 🏆 Key Results

* XGBoost achieves the best **Top 1% return performance**
* LightGBM shows stable performance across metrics
* MLP achieves highest AUC but weaker trading performance

> **Key insight:**
> Classification accuracy does not directly translate to investment performance.

---
## 📊 Model Performance

### Overall Comparison

![Top5 Return](figures/bar_top5_future_return_all.png)

![Top5 Hit Rate](figures/bar_top5_hit_rate_all.png)

---

### Grouped Comparison

![Grouped Return](figures/grouped_top5_future_return.png)

![Grouped AUC](figures/grouped_auc.png)
## 🔍 SHAP Analysis

We select **XGBoost (raw features)** as the final model based on Top 1% return.

### Global Feature Importance

![Global SHAP](shap_outputs_xgboost_raw/global_shap_bar.png)

---

### SHAP Summary Plot

![SHAP Summary](shap_outputs_xgboost_raw/global_shap_summary.png)

---
### Top 1% Feature Importance

![Top1 SHAP](shap_outputs_xgboost_raw/top1_shap_bar.png)

---

### Stability Across Years

![SHAP 2023](shap_outputs_xgboost_raw/year_2023_shap_bar.png)
![SHAP 2024](shap_outputs_xgboost_raw/year_2024_shap_bar.png)
![SHAP 2025](shap_outputs_xgboost_raw/year_2025_shap_bar.png)

---

### Key Insights

* **Momentum effect**

  * High `board_rs_20d`, `roc_20` → higher probability of future gains

* **Trend effect**

  * Positive slope features → bullish signal

* **Mean reversion**

  * High `bias_60` → negative contribution

> The model captures a combination of:
> **Momentum + Trend + Mean Reversion**

---

### Stability Across Years


![SHAP 2023](shap_outputs_xgboost_raw/year_2023_shap_bar.png)
![SHAP 2024](shap_outputs_xgboost_raw/year_2024_shap_bar.png)


Feature importance remains consistent across years, indicating stable patterns rather than overfitting.

---

## 📂 Project Structure

```text
.
├── figures/                         # Model performance plots
│   ├── bar_top5_future_return_all.png
│   ├── bar_top5_hit_rate_all.png
│   ├── grouped_auc.png
│   └── grouped_top5_future_return.png
│
├── shap_outputs_xgboost_raw/        # SHAP results (final XGBoost model)
│   ├── global_shap_bar.png
│   ├── global_shap_summary.png
│   ├── top1_shap_bar.png
│   ├── year_2023_shap_bar.png
│   ├── year_2024_shap_bar.png
│   └── yearly_shap_importance.csv
│
├── model_compare_summary*.csv       # Model comparison results
│
├── config.py                        # Config & paths
├── data_pipeline.py                 # Raw feature pipeline
├── data_pipeline_standard.py        # Standardized / z-score pipeline
├── evaluate.py                      # Evaluation metrics
├── models.py                        # Model definitions
├── plot_figure.py                   # Visualization scripts
├── shap_analysis.py                 # SHAP analysis
├── train_model_compare.py           # Model comparison (raw)
├── train_model_compare_standard.py  # Model comparison (preprocessed)
├── requirements.txt
└── README.md

---

## 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python train_model_compare.py
```
```bash
python train_model_compare_standard.py
```

Compare preprocessing:

```bash
python train_model_compare.py
```
```bash
python train_model_compare_standard.py
```
Run SHAP analysis:

```bash
python shap_analysis.py
```

---

## ⭐ Highlights

* Time-based split (no data leakage)
* Focus on **Top-K return (practical trading metric)**
* Multi-model comparison
* SHAP interpretability
* Stable feature importance across years.

---

## 📌 Future Work

* Rolling window backtesting
* Portfolio construction
* Transaction cost modeling
* Feature expansion (volume, fundamentals)
* Model ensemble
