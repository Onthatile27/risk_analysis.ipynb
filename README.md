# risk_analysis.ipynb
# Data-Driven Risk Detection & Business Loss Prevention
# Author: Onthatile Nkunika

# -----------------------------
# STEP 1: Import Libraries
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# STEP 2: Load Dataset
# -----------------------------
# Place your CSV file in the 'data/' folder
data = pd.read_csv('../data/transactions.csv')

# Quick peek at the data
print(data.head())
print(data.info())

# -----------------------------
# STEP 3: Data Cleaning
# -----------------------------
# Check for missing values
print(data.isnull().sum())

# Fill missing Amount with 0
data['Amount'] = data['Amount'].fillna(0)

# Remove duplicate transactions
data = data.drop_duplicates()

# -----------------------------
# STEP 4: Basic Analysis
# -----------------------------
# Total transactions
total_transactions = len(data)

# Average transaction amount
avg_amount = data['Amount'].mean()

# Failed transactions rate
failed_rate = len(data[data['Status'] == 'Failed']) / total_transactions * 100

print(f"Total Transactions: {total_transactions}")
print(f"Average Amount: {avg_amount:.2f}")
print(f"Failed Transaction Rate: {failed_rate:.2f}%")

# -----------------------------
# STEP 5: Risk Detection
# -----------------------------
# High-value transactions = amount > mean + 2*std
threshold = data['Amount'].mean() + 2 * data['Amount'].std()
high_risk_transactions = data[data['Amount'] > threshold]

print(f"High-risk transactions (> mean + 2*std): {len(high_risk_transactions)}")

# Users with multiple failed attempts
failed_users = data[data['Status'] == 'Failed'].groupby('User_ID').size()
multiple_failed_users = failed_users[failed_users > 1]

print("Users with multiple failed attempts:")
print(multiple_failed_users)

# Unusual locations (example: locations outside top 3 most common)
top_locations = data['Location'].value_counts().nlargest(3).index
unusual_location_tx = data[~data['Location'].isin(top_locations)]
print(f"Transactions from unusual locations: {len(unusual_location_tx)}")

# -----------------------------
# STEP 6: Visualization
# -----------------------------
sns.set_style('whitegrid')

# Histogram of transaction amounts
plt.figure(figsize=(8,5))
sns.histplot(data['Amount'], bins=30, kde=True, color='skyblue')
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.savefig('../visuals/amount_histogram.png')
plt.show()

# Bar chart of failed transactions per user
failed_per_user = data[data['Status'] == 'Failed'].groupby('User_ID').size()
plt.figure(figsize=(8,5))
failed_per_user.plot(kind='bar', color='salmon')
plt.title('Failed Transactions per User')
plt.xlabel('User ID')
plt.ylabel('Failed Transactions')
plt.savefig('../visuals/failed_per_user.png')
plt.show()

# Line chart: failed transactions over time
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
failed_over_time = data[data['Status'] == 'Failed'].groupby(data['Timestamp'].dt.date).size()
plt.figure(figsize=(8,5))
failed_over_time.plot(marker='o', linestyle='-')
plt.title('Failed Transactions Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Failed Transactions')
plt.xticks(rotation=45)
plt.savefig('../visuals/failed_over_time.png')
plt.show()

# -----------------------------
# STEP 7: Save Risk Report
# -----------------------------
# Combine high-risk, multiple-failed, and unusual location
risk_report = pd.concat([
    high_risk_transactions,
    data[data['User_ID'].isin(multiple_failed_users.index)],
    unusual_location_tx
]).drop_duplicates()

risk_report.to_csv('../data/risk_report.csv', index=False)
print("Risk report saved as 'risk_report.csv'")

