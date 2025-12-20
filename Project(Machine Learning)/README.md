# Demand Forecasting Module

This module (`ml_model.py`) contains the `DemandForecaster` class, which is responsible for training a Machine Learning model to forecast sales demand based on historical data. It handles the entire pipeline from data loading to prediction.

## Overview

The system uses a **HistGradientBoostingRegressor** from Scikit-Learn, which is efficient for large datasets and handles tabular data well.

**Key Features:**
- **Automated Data Loading:** Fetches sales records directly from the SQL database.
- **Feature Engineering:** Automatically extracts temporal features (Year, Month, Day, Day of Week) from dates.
- **Preprocessing:** Handles categorical variables (`family`, `store_nbr`) using OneHotEncoding.
- **Model:** Uses a Histogram-based Gradient Boosting Regression tree.
- **Metrics:** Calculates MAE, RMSE, and R2 Score after training.

## Prerequisites

Make sure you have the following Python libraries installed. You can install them via pip:

```bash
pip install pandas numpy scikit-learn sqlalchemy python-dateutil
