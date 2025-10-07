# ===============================================
# Project: Demand Forecasting & Inventory Optimization
# Step 6: Inventory Optimization using Forecasted Sales
# Author: Prem
# ===============================================

import pandas as pd
import numpy as np
import joblib

# -------------------------
# 1. Load Feature-Engineered Dataset
# -------------------------
data = pd.read_csv("train_featured.csv", parse_dates=["Date"])
print(f"Dataset Loaded: {data.shape}")

# -------------------------
# 2. Load Trained XGBoost Model
# -------------------------
model = joblib.load("xgb_sales_model.pkl")
print("✅ XGBoost model loaded.")

# -------------------------
# 3. Predict Weekly Sales
# -------------------------
features = [
    "Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3",
    "MarkDown4","MarkDown5","CPI","Unemployment","Size",
    "IsHoliday","Lag_1","Lag_4","Rolling_Mean_4","Rolling_Mean_12",
    "Quarter","WeekOfYear"
]

data["Predicted_Sales"] = model.predict(data[features])
print("✅ Weekly sales forecast generated.")

# -------------------------
# 4. Inventory Calculations
# -------------------------
# Parameters
lead_time_weeks = 2           # e.g., 2 weeks lead time
service_level = 0.95
Z = 1.65                      # Z-factor for 95% service level

# Group by Store & Dept
inventory = data.groupby(["Store","Dept"]).agg(
    Avg_Weekly_Demand=("Predicted_Sales","mean"),
    Demand_Std=("Predicted_Sales","std")
).reset_index()

# Safety Stock
inventory["Safety_Stock"] = Z * inventory["Demand_Std"] * np.sqrt(lead_time_weeks)

# Reorder Point
inventory["ROP"] = (inventory["Avg_Weekly_Demand"] * lead_time_weeks) + inventory["Safety_Stock"]

# Optional: EOQ (example, requires cost info)
# inventory["EOQ"] = np.sqrt( (2 * inventory["Avg_Weekly_Demand"]*52 * 100) / 2 )
# 100 = holding cost per unit, 2 = order cost (example numbers)

# -------------------------
# 5. Save Inventory Recommendations
# -------------------------
inventory.to_csv("inventory_optimization.csv", index=False)
print("✅ Inventory optimization saved as 'inventory_optimization.csv'")

# -------------------------
# 6. Quick Preview
# -------------------------
print("\nSample Inventory Recommendations:")
print(inventory.head(10))
