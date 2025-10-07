# ===============================================
# Project: Demand Forecasting & Inventory Optimization
# Step 3: Exploratory Data Analysis (EDA)
# Author: Prem
# ===============================================

# -------------------------
# 1. Import Libraries
# -------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 2. Load Cleaned Dataset
# -------------------------
train_full = pd.read_csv("train_full_cleaned.csv")
train_full['Date'] = pd.to_datetime(train_full['Date'])

# -------------------------
# 3. Overview Statistics
# -------------------------
print("Dataset Shape:", train_full.shape)
print("Columns:", train_full.columns)
print(train_full.describe())

# -------------------------
# 4. Weekly Sales Trend (Overall)
# -------------------------
plt.figure(figsize=(14,6))
weekly_sales = train_full.groupby('Date')['Weekly_Sales'].sum()
weekly_sales.plot()
plt.title("Overall Weekly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.show()

# -------------------------
# 5. Weekly Sales Trend by Store Type
# -------------------------
plt.figure(figsize=(14,6))
for store_type in train_full['Type'].unique():
    temp = train_full[train_full['Type']==store_type].groupby('Date')['Weekly_Sales'].sum()
    plt.plot(temp.index, temp.values, label=f'Store Type {store_type}')
plt.title("Weekly Sales by Store Type")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.legend()
plt.show()

# -------------------------
# 6. Impact of Holidays on Sales
# -------------------------
plt.figure(figsize=(10,5))
holiday_sales = train_full.groupby('IsHoliday')['Weekly_Sales'].sum()
sns.barplot(x=holiday_sales.index, y=holiday_sales.values)
plt.title("Total Sales: Holiday vs Non-Holiday")
plt.ylabel("Weekly Sales")
plt.show()

# -------------------------
# 7. Department-wise Sales Trend (Top 5 Departments)
# -------------------------
dept_sales = train_full.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False)
top5_depts = dept_sales.head(5).index.tolist()
plt.figure(figsize=(14,6))
for dept in top5_depts:
    temp = train_full[train_full['Dept']==dept].groupby('Date')['Weekly_Sales'].sum()
    plt.plot(temp.index, temp.values, label=f'Dept {dept}')
plt.title("Top 5 Departments: Weekly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.show()

# -------------------------
# 8. Correlation Heatmap (Numerical Features)
# -------------------------
numeric_cols = ['Weekly_Sales','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment','Size']
plt.figure(figsize=(12,8))
sns.heatmap(train_full[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# 9. Effect of Markdown on Sales (Sample)
# -------------------------
plt.figure(figsize=(12,6))
sns.scatterplot(x='MarkDown1', y='Weekly_Sales', data=train_full, alpha=0.3)
plt.title("Impact of MarkDown1 on Weekly Sales")
plt.xlabel("MarkDown1")
plt.ylabel("Weekly Sales")
plt.show()

# -------------------------
# 10. Store Size vs Total Sales
# -------------------------
store_sales = train_full.groupby('Store').agg({'Weekly_Sales':'sum','Size':'first'}).reset_index()
plt.figure(figsize=(10,6))
sns.scatterplot(x='Size', y='Weekly_Sales', data=store_sales)
plt.title("Store Size vs Total Sales")
plt.xlabel("Store Size (sq.ft)")
plt.ylabel("Total Sales")
plt.show()

# -------------------------
# 11. Save EDA Summary (Optional)
# -------------------------
eda_summary = train_full.describe()
eda_summary.to_csv("EDA_Summary.csv", index=True)
print("EDA Summary saved as 'EDA_Summary.csv'.")
