import pandas as pd

# Load the sampled CSV file
data = pd.read_csv("us_accidents_sample.csv")

# Number of rows and columns
num_rows, num_columns = data.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# Percentage of missing values per column
missing_values = data.isnull().mean() * 100
print("\nPercentage of missing values per column:")
print(missing_values[missing_values > 0].sort_values(ascending=False))

# Number of unique values per column
unique_counts = data.nunique()
print("\nNumber of unique values per column:")
print(unique_counts.sort_values(ascending=False))

# Data types of each column
print("\nData types of each column:")
print(data.dtypes)

# Basic statistics for numerical columns
print("\nBasic statistics for numerical columns:")
print(data.describe())

# Basic statistics for categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
print("\nBasic statistics for categorical columns:")
print(data[categorical_columns].describe())

# Preview first few rows
print("\nPreview of the first few rows:")
print(data.head())