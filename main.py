# ===============================================
# Project: Demand Forecasting & Inventory Optimization
# Step 2: Data Exploration & Cleaning
# Author: Prem
# ===============================================

# -------------------------
# 1. Import Required Libraries
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 2. Load Datasets
# -------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
stores = pd.read_csv("stores.csv")
features = pd.read_csv("features.csv")

# Preview datasets
print("Train Dataset Head:\n", train.head())
print("Features Dataset Head:\n", features.head())
print("Stores Dataset Head:\n", stores.head())

# -------------------------
# 3. Merge Datasets
# -------------------------
# Merge train with features
train_full = pd.merge(train, features, on=['Store', 'Date'], how='left')

# Merge with stores data
train_full = pd.merge(train_full, stores, on='Store', how='left')

# Convert Date column to datetime
train_full['Date'] = pd.to_datetime(train_full['Date'])

# Preview merged dataset
print("Merged Dataset Info:\n", train_full.info())
print("Merged Dataset Head:\n", train_full.head())

# -------------------------
# 4. Handle Missing Values
# -------------------------
# Check missing values
print("Missing Values Before Handling:\n", train_full.isna().sum())

# Fill MarkDowns missing values with 0
markdown_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
train_full[markdown_cols] = train_full[markdown_cols].fillna(0)

# Forward fill CPI and Unemployment by store
train_full[['CPI','Unemployment']] = train_full.groupby('Store')[['CPI','Unemployment']].ffill()

# Verify missing values after cleaning
print("Missing Values After Handling:\n", train_full.isna().sum())

# -------------------------
# 5. Reconcile IsHoliday Column
# -------------------------
train_full['IsHoliday'] = train_full['IsHoliday_x']  # keep train's column
train_full.drop(['IsHoliday_x','IsHoliday_y'], axis=1, inplace=True)

# -------------------------
# 6. Create Time Features
# -------------------------
train_full['Year'] = train_full['Date'].dt.year
train_full['Month'] = train_full['Date'].dt.month
train_full['Week'] = train_full['Date'].dt.isocalendar().week
train_full['Day'] = train_full['Date'].dt.day

# -------------------------
# 7. Check for Outliers in Weekly_Sales
# -------------------------
plt.figure(figsize=(12,6))
sns.boxplot(train_full['Weekly_Sales'])
plt.title("Weekly Sales Distribution")
plt.show()

# Clip extreme values at 1st and 99th percentile
q1 = train_full['Weekly_Sales'].quantile(0.01)
q99 = train_full['Weekly_Sales'].quantile(0.99)
train_full['Weekly_Sales'] = train_full['Weekly_Sales'].clip(q1, q99)

# -------------------------
# 8. Save Cleaned Dataset
# -------------------------
train_full.to_csv("train_full_cleaned.csv", index=False)
print("Cleaned dataset saved as 'train_full_cleaned.csv'.")
