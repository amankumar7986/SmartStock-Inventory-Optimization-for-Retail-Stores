import sys, subprocess, os, math
py = sys.executable

def pip_install(packages):
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade"] + packages)

try:
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
    import lightgbm as lgb
    from datasets import load_dataset
    from sklearn.metrics import mean_squared_error
    import joblib
except Exception:
    pip_install(["pandas>=1.5","numpy>=1.24","matplotlib>=3.6","seaborn>=0.12",
                 "lightgbm>=3.3","datasets>=2.14","scikit-learn>=1.2","joblib>=1.2","pyarrow>=11.0"])
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
    import lightgbm as lgb
    from datasets import load_dataset
    from sklearn.metrics import mean_squared_error
    import joblib

# Notebook inline plotting (works in Jupyter)
%matplotlib inline

# ---- CONFIG ----
HF_DATASET_ID = "t4tiana/store-sales-time-series-forecasting"
PROJECT_PDF_PATH = "/mnt/data/Final Project pdf.pdf"
PROJECT_PDF_URL = "file://" + PROJECT_PDF_PATH

TARGET_COL = "sales"
DATE_COL = "date"
STORE_COL = "store_nbr"
PRODUCT_COL = "family"

LAGS = [7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

TEST_DAYS = 30
VAL_DAYS = 30

BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "lightgbm_store_sales.pkl")
REPORT_CSV = os.path.join(REPORTS_DIR, "stock_report.csv")
VAL_PLOT = os.path.join(OUTPUTS_DIR, "val_metric.png")
PRED_PLOT = os.path.join(OUTPUTS_DIR, "test_pred_vs_true.png")
CATEGORY_BAR = os.path.join(OUTPUTS_DIR, "category_stock_status.png")
STORE_HEATMAP = os.path.join(OUTPUTS_DIR, "store_category_summary.png")  # reused path for bar chart

RANDOM_STATE = 42
STOCK_MULTIPLIER = 1.2

# ---- UTILITIES ----
def safe_save_model(payload, path=MODEL_PATH):
    joblib.dump(payload, path)
    print("Saved model to", path)

def _get_best_iteration_for_predict(model):
    bi = None
    if hasattr(model, "best_iteration"):
        bi = getattr(model, "best_iteration")
        if isinstance(bi, (int,)) and bi > 0:
            return bi
    return None

def plot_train_val_from_model(model, out=VAL_PLOT):
    try:
        evals = {}
        if hasattr(model, "evals_result"):
            try:
                evals = model.evals_result()
            except Exception:
                evals = {}
        train_vals = []
        valid_vals = []
        if evals:
            if "train" in evals and isinstance(evals["train"], dict):
                metric = next(iter(evals["train"].keys()))
                train_vals = evals["train"].get(metric, [])
            if "valid" in evals and isinstance(evals["valid"], dict):
                metric = next(iter(evals["valid"].keys()))
                valid_vals = evals["valid"].get(metric, [])
        if (len(train_vals) == 0 and len(valid_vals) == 0) and hasattr(model, "record_evals"):
            try:
                rec = getattr(model, "record_evals")
                if "train" in rec and isinstance(rec["train"], dict):
                    metric = next(iter(rec["train"].keys()))
                    train_vals = rec["train"][metric].get("eval", [])
                if "valid" in rec and isinstance(rec["valid"], dict):
                    metric = next(iter(rec["valid"].keys()))
                    valid_vals = rec["valid"][metric].get("eval", [])
            except Exception:
                pass

        if len(train_vals) == 0 and len(valid_vals) == 0:
            print("No training/validation history found in model to plot.")
            return

        plt.figure(figsize=(8,4))
        if len(train_vals) > 0:
            plt.plot(train_vals, label="train")
        if len(valid_vals) > 0:
            plt.plot(valid_vals, label="valid")
        plt.xlabel("iteration")
        plt.ylabel("metric")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out)
        plt.show()
        plt.close()
        print("Saved validation plot to", out)
    except Exception as e:
        print("Could not plot train/val:", e)

def plot_pred_vs_true(y_true, y_pred, out=PRED_PLOT, sample_n=500):
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = min(len(y_true), len(y_pred), sample_n)
        if n == 0:
            print("No values to plot for predictions.")
            return
        plt.figure(figsize=(10,4))
        plt.plot(y_true[:n], label="true")
        plt.plot(y_pred[:n], label="pred")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out)
        plt.show()
        plt.close()
        print("Saved pred vs true to", out)
    except Exception as e:
        print("Could not plot pred vs true:", e)

def plot_category_status(report_df, out=CATEGORY_BAR):
    try:
        plt.figure(figsize=(10,6))
        order = report_df["stock_status"].value_counts().index
        sns.countplot(data=report_df, x="stock_status", order=order)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(out)
        plt.show()
        plt.close()
        print("Saved category status chart to", out)
    except Exception as e:
        print("Could not plot category status:", e)

# ---- NEW: Simple grouped bar chart per store (replaces heatmap) ----
def plot_store_category_bar(report_df, out=STORE_HEATMAP):
    try:
        # Count per store per stock_status
        counts = (
            report_df.groupby([STORE_COL, "stock_status"])
            .size()
            .reset_index(name="count")
        )

        # Pivot for grouped bar chart
        pivot_df = counts.pivot(index=STORE_COL, columns="stock_status", values="count").fillna(0)

        # Plot grouped bar chart (matplotlib via DataFrame.plot)
        ax = pivot_df.plot(kind="bar", figsize=(14, 6))
        ax.set_xlabel("Store Number")
        ax.set_ylabel("Count of Categories")
        ax.set_title("Stock Status Summary Per Store")
        ax.set_xticklabels([str(x) for x in pivot_df.index], rotation=45)
        plt.tight_layout()
        plt.savefig(out)
        plt.show()
        plt.close()
        print("Saved store stock status bar chart to", out)
    except Exception as e:
        print("Could not plot store bar chart:", e)

# ---- DATA LOADING ----
print("Loading dataset from Hugging Face:", HF_DATASET_ID)
ds = load_dataset(HF_DATASET_ID, split="train")
df = pd.DataFrame(ds)
rename_map = {"store": STORE_COL, "family": PRODUCT_COL, "sales": TARGET_COL, "date": DATE_COL}
df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
print("Loaded, rows:", len(df), "columns:", df.shape[1])
display(df.head())

# ---- FEATURES ----
def prepare_basic_features(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([STORE_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)
    df["day"] = df[DATE_COL].dt.day
    df["month"] = df[DATE_COL].dt.month
    df["year"] = df[DATE_COL].dt.year
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["weekofyear"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df[DATE_COL].dt.is_month_start.astype(int)
    df["is_month_end"] = df[DATE_COL].dt.is_month_end.astype(int)
    df[STORE_COL] = df[STORE_COL].astype("category")
    df[PRODUCT_COL] = df[PRODUCT_COL].astype("category")
    return df

def add_lag_and_rolling(df):
    df = df.copy()
    df = df.sort_values([STORE_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)
    groups = [STORE_COL, PRODUCT_COL]
    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby(groups, observed=True)[TARGET_COL].shift(lag)
    for win in ROLLING_WINDOWS:
        df[f"rmean_{win}"] = df.groupby(groups, observed=True)[TARGET_COL].apply(
            lambda x: x.shift(1).rolling(window=win, min_periods=1).mean()
        ).values
        df[f"rstd_{win}"] = df.groupby(groups, observed=True)[TARGET_COL].apply(
            lambda x: x.shift(1).rolling(window=win, min_periods=1).std().fillna(0)
        ).values
    return df

df = prepare_basic_features(df)
df = add_lag_and_rolling(df)
print("After lagging/features, shape:", df.shape)

min_needed = max(max(LAGS), max(ROLLING_WINDOWS))
df = df[df[DATE_COL] >= (df[DATE_COL].min() + pd.Timedelta(days=min_needed))].reset_index(drop=True)
print("After removing insufficient-history rows:", df.shape)

# ---- SPLIT ----
max_date = df[DATE_COL].max()
test_start = max_date - pd.Timedelta(days=TEST_DAYS - 1)
val_start = test_start - pd.Timedelta(days=VAL_DAYS)
train_df = df[df[DATE_COL] < val_start].copy()
val_df = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] < test_start)].copy()
test_df = df[df[DATE_COL] >= test_start].copy()
print("train, val, test shapes:", train_df.shape, val_df.shape, test_df.shape)

# ---- PREPARE FEATURES & DTYPE HANDLING ----
def get_feature_columns(df):
    exclude = [DATE_COL, TARGET_COL]
    return [c for c in df.columns if c not in exclude and c != "index"]

feature_cols = get_feature_columns(df)
categorical_feats = [c for c in [STORE_COL, PRODUCT_COL] if c in feature_cols]
numeric_feats = [c for c in feature_cols if c not in categorical_feats]

for part in (train_df, val_df, test_df):
    for c in numeric_feats:
        if c in part.columns:
            part.loc[:, c] = part[c].fillna(-1)

for c in categorical_feats:
    if not pd.api.types.is_categorical_dtype(train_df[c]):
        train_df.loc[:, c] = train_df[c].astype("category")
    if "__missing__" not in list(train_df[c].cat.categories):
        train_df.loc[:, c] = train_df[c].cat.add_categories(["__missing__"])
    train_cats = list(train_df[c].cat.categories)
    for part in (val_df, test_df):
        if not pd.api.types.is_categorical_dtype(part[c]):
            part.loc[:, c] = part[c].astype("category")
        part.loc[:, c] = part[c].cat.set_categories(train_cats)
    for part in (train_df, val_df, test_df):
        part.loc[:, c] = part[c].fillna("__missing__").astype("category")

# ---- TRAIN ----
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
}

dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[TARGET_COL], categorical_feature=categorical_feats, free_raw_data=False)
dval = lgb.Dataset(val_df[feature_cols], label=val_df[TARGET_COL], reference=dtrain, categorical_feature=categorical_feats, free_raw_data=False)

print("Training LightGBM (logs every 200 iters)...")
model = lgb.train(
    params,
    dtrain,
    num_boost_round=4000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=200)],
)

best_iter = _get_best_iteration_for_predict(model)
val_pred = model.predict(val_df[feature_cols], num_iteration=best_iter)
test_pred = model.predict(test_df[feature_cols], num_iteration=best_iter)
print("Val RMSE:", math.sqrt(mean_squared_error(val_df[TARGET_COL], val_pred)))
print("Test RMSE:", math.sqrt(mean_squared_error(test_df[TARGET_COL], test_pred)))

safe_save_model({"model": model, "meta": {"feature_cols": feature_cols, "categorical_feats": categorical_feats}}, MODEL_PATH)

# ---- PLOTS & SAVE ----
plot_train_val_from_model(model, out=VAL_PLOT)
plot_pred_vs_true(test_df[TARGET_COL].values, test_pred, out=PRED_PLOT)

# ---- FORECAST LATEST & STOCK ESTIMATION ----
latest = df.sort_values(DATE_COL).groupby([STORE_COL, PRODUCT_COL]).tail(1).reset_index(drop=True)

for c in numeric_feats:
    if c in latest.columns:
        latest.loc[:, c] = latest[c].fillna(-1)
for c in categorical_feats:
    if c in latest.columns:
        latest.loc[:, c] = latest[c].astype("category")
        latest.loc[:, c] = latest[c].cat.set_categories(train_df[c].cat.categories).fillna("__missing__").astype("category")

latest["forecast"] = model.predict(latest[feature_cols], num_iteration=best_iter)

def compute_last14(df):
    recs = []
    grouped = df.sort_values(DATE_COL).groupby([STORE_COL, PRODUCT_COL])
    for (s, f), g in grouped:
        vals = g.tail(14)[TARGET_COL].values
        avg = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
        recs.append((s, f, avg))
    return pd.DataFrame(recs, columns=[STORE_COL, PRODUCT_COL, "last_14d_avg"])

last14 = compute_last14(df)
latest = latest.merge(last14, on=[STORE_COL, PRODUCT_COL], how="left")
cat_means = df.groupby(PRODUCT_COL)[TARGET_COL].mean().rename("category_mean").reset_index()
latest = latest.merge(cat_means, on=PRODUCT_COL, how="left")
latest["last_14d_avg"] = latest["last_14d_avg"].fillna(latest["category_mean"]).fillna(0.0)
latest["estimated_stock"] = latest["last_14d_avg"] * STOCK_MULTIPLIER

def classify_row(r):
    if r["forecast"] > r["estimated_stock"]:
        return "UNDERSTOCK 🔴 (Reorder Needed)"
    elif r["forecast"] < r["estimated_stock"] * 0.7:
        return "OVERSTOCK 🟢 (Reduce Inventory)"
    else:
        return "BALANCED ⚪"

latest["stock_status"] = latest.apply(classify_row, axis=1)
report_df = latest[[STORE_COL, PRODUCT_COL, "forecast", "last_14d_avg", "estimated_stock", "stock_status"]].copy()
report_df.to_csv(REPORT_CSV, index=False)
print("Saved stock report to", REPORT_CSV)

plot_category_status(report_df, out=CATEGORY_BAR)
# Use new grouped bar chart instead of heatmap
plot_store_category_bar(report_df, out=STORE_HEATMAP)

print("\nDone. Key files:")
print("Model ->", MODEL_PATH)
print("Report ->", REPORT_CSV)
print("Validation plot ->", VAL_PLOT)
print("Pred vs True ->", PRED_PLOT)
print("Category bar ->", CATEGORY_BAR)
print("Store grouped-bar ->", STORE_HEATMAP)
print("\nProject PDF path:", PROJECT_PDF_PATH)
print("Project PDF URL:", PROJECT_PDF_URL)

# show head of report
display(report_df.head(30))
