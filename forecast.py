# ===============================================
# Project: Demand Forecasting & Inventory Optimization
# Step 5: ML Forecasting using XGBoost
# Author: Prem
# ===============================================

# -------------------------
# 1. Import Libraries
# -------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 2. Load Feature-Engineered Dataset
# -------------------------
data = pd.read_csv("train_featured.csv", parse_dates=["Date"])
print(f"Dataset Loaded: {data.shape}")

# -------------------------
# 3. Feature Selection
# -------------------------
features = [
    "Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3",
    "MarkDown4","MarkDown5","CPI","Unemployment","Size",
    "IsHoliday","Lag_1","Lag_4","Rolling_Mean_4","Rolling_Mean_12",
    "Quarter","WeekOfYear"
]

target = "Weekly_Sales"

X = data[features]
y = data[target]

# -------------------------
# 4. Train-Test Split (by date)
# -------------------------
train_data = data[data['Date'] < '2012-01-01']
val_data = data[data['Date'] >= '2012-01-01']

X_train = train_data[features]
y_train = train_data[target]

X_val = val_data[features]
y_val = val_data[target]

print(f"Training Shape: {X_train.shape}, Validation Shape: {X_val.shape}")

# -------------------------
# 5. Train XGBoost Regressor (Fixed for newer versions)
# -------------------------
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("🔹 Training XGBoost model...")

# Newer XGBoost requires eval_metric with early_stopping_rounds
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",        # required for early stopping
    early_stopping_rounds=50,
    verbose=50
)

# -------------------------
# 6. Model Evaluation
# -------------------------
y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

print(f"\nModel Performance on Validation Set:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# -------------------------
# 7. Save Model
# -------------------------
joblib.dump(model, "xgb_sales_model.pkl")
print("✅ XGBoost model saved as 'xgb_sales_model.pkl'")

# -------------------------
# 8. Feature Importance Plot
# -------------------------
importance = model.feature_importances_
feat_importance = pd.Series(importance, index=features).sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=feat_importance.values, y=feat_importance.index)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
