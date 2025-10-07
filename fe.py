# feature_engineering.py
# Step 4: Feature Engineering for Demand Forecasting & Inventory Optimization
# Author: Prem

import pandas as pd
import numpy as np

# -------------------------------
# Step 1: Load the cleaned dataset
# -------------------------------
print("🔹 Loading cleaned dataset...")
data = pd.read_csv("train_full_cleaned.csv", parse_dates=["Date"])
print(f"Dataset Loaded: {data.shape}")

# -------------------------------
# Step 2: Feature Creation
# -------------------------------

# 1️⃣ Lag Features – capturing last week and last month sales
data = data.sort_values(by=["Store", "Dept", "Date"])
data["Lag_1"] = data.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
data["Lag_4"] = data.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)

# 2️⃣ Rolling Mean Features
data["Rolling_Mean_4"] = data.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(window=4).mean()
data["Rolling_Mean_12"] = data.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(window=12).mean()

# 3️⃣ Holiday Encoding – convert boolean to binary
data["IsHoliday"] = data["IsHoliday"].astype(int)

# 4️⃣ Seasonality Indicators
data["Quarter"] = data["Date"].dt.quarter
data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)

# -------------------------------
# Step 3: Handle Missing Values in new features
# -------------------------------
lag_cols = ["Lag_1", "Lag_4", "Rolling_Mean_4", "Rolling_Mean_12"]
for col in lag_cols:
    data[col] = data[col].fillna(data[col].mean())

# -------------------------------
# Step 4: Drop duplicates and verify
# -------------------------------
data = data.drop_duplicates()
print(f"After feature creation: {data.shape}")

# -------------------------------
# Step 5: Save final modeling dataset
# -------------------------------
data.to_csv("train_featured.csv", index=False)
print("✅ Feature-engineered dataset saved as 'train_featured.csv'")

# -------------------------------
# Step 6: Quick Feature Summary
# -------------------------------
print("\n🔍 Feature Summary:")
print(data[["Weekly_Sales", "Lag_1", "Lag_4", "Rolling_Mean_4", "Rolling_Mean_12", "IsHoliday", "Quarter"]].head(10))
