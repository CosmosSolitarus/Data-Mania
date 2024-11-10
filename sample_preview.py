import pandas as pd
import time

start_time = time.time()

# Load the sampled CSV file
data = pd.read_csv("us_accidents_cleaned.csv")

# Columns
print(list(data.columns))

# Number of rows and columns
num_rows, num_columns = data.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# Count and percentage of missing values per column (in original order)
missing_values_count = data.isnull().sum()
missing_values_percent = data.isnull().mean() * 100
missing_values_df = pd.DataFrame({
    'Missing Count': missing_values_count,
    'Missing Percentage (%)': missing_values_percent
})

print("\nCount and Percentage of missing values per column:")
print(missing_values_df)

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

end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds ({(end_time - start_time)/60:.2f} minutes)")