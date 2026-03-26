# explainable-short-horizon-stock-selection

📌 Project Overview

Explainable Short-Horizon Stock Selection

This project builds a machine learning pipeline for short-horizon stock selection, aiming to identify stocks with strong return potential over the next 5 trading days.

Unlike traditional classification tasks, this project focuses on ranking performance (Top-K returns) rather than pure accuracy, and integrates SHAP-based interpretability to understand model decisions.

🎯 Objective

Given a set of technical features for each stock on each trading day, the goal is to predict whether the stock will achieve a positive return over the next 5 days:

Target:

regime_binary = 1 if r_future_5 > 0.01 else 0
Train/Test split:
Train: before 2023
Test: after 2023
🧠 Key Ideas
Treat stock selection as a cross-sectional ranking problem
Focus on Top-K return performance
Compare multiple models under different preprocessing methods
Use SHAP to interpret model behavior and validate financial intuition
