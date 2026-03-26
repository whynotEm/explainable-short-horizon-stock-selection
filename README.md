# Explainable Short-Horizon Stock Selection

A machine learning framework for short-horizon stock selection, optimized for **Top-K return performance** and enhanced with **SHAP-based interpretability**.

---

## 📌 Project Overview

This project builds a machine learning pipeline to identify stocks with strong return potential over the next 5 trading days.

Instead of focusing on classification accuracy alone, this project emphasizes:

- Cross-sectional ranking
- Top-K return performance
- Model interpretability (SHAP)

---

## 🎯 Objective

Given a set of features for each stock on each trading day, predict whether the stock will achieve positive returns over the next 5 days.

**Target definition:**

```text
regime_binary = 1 if r_future_5 > 0.01 else 0
