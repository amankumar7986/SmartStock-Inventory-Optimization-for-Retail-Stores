# SmartStock Inventory Optimization – Machine Learning Pipeline

SmartStock is an end-to-end inventory optimization system that forecasts sales, analyzes stock levels, and produces automated reports for retail stores.  
This project uses a LightGBM-based time-series model along with custom feature engineering, stock-estimation logic, and visual analytics.

---

## 🚀 Features

### ✔ Complete ML Pipeline
- Loads dataset from **Hugging Face**: `t4tiana/store-sales-time-series-forecasting`
- Automatic:
  - Data preprocessing  
  - Feature engineering (lag, rolling, time features)  
  - Train / Validation / Test split  
  - Model training with LightGBM + early stopping  
  - Model saving (`.pkl`)  

### ✔ Inventory Forecasting & Health Classification
For every (store, product):
- Forecasts upcoming sales
- Computes 14-day average usage
- Estimates recommended stock level
- Labels stock health:
  - 🔴 UNDERSTOCK (Reorder Needed)
  - 🟢 OVERSTOCK (Reduce Inventory)
  - ⚪ BALANCED

### ✔ Automated Outputs (Generated)
- `models/lightgbm_store_sales.pkl`
- `reports/stock_report.csv`
- Visuals under `outputs/`:
  - `val_metric.png`
  - `test_pred_vs_true.png`
  - `category_stock_status.png`
  - `store_category_summary.png` (grouped bar chart)

### ✔ Optional Interactive Plots (Plotly)
- Interactive prediction charts  
- Interactive store-level grouped bar chart  

---


