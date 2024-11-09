import pandas as pd

# Load the CSV file
df = pd.read_csv('us_accidents_sample.csv')

# Count the number of missing values per row
missing_data_per_row = df.isnull().sum(axis=1)

# Count how many rows have 1, 2, 3, ... missing values
missing_counts = missing_data_per_row.value_counts().sort_index()

# Calculate the percentage of rows with missing 1, 2, 3, ... columns
total_rows = len(df)

# Iterate through the counts of missing values per row
for missing, count in missing_counts.items():
    percentage = (count / total_rows) * 100
    column_label = 'column' if missing == 1 else 'columns'
    
    # Print the percentage of rows with missing columns
    print(f"Missing {missing} {column_label}: {percentage:.5f}% of rows.")
    
    # Identify the most common missing columns for this number of missing values
    if missing > 0:
        # Get the rows where exactly 'missing' columns are NaN
        rows_with_missing = missing_data_per_row[missing_data_per_row == missing]
        
        # Get the columns that are missing in these rows
        missing_columns = df.columns[df.iloc[rows_with_missing.index].isnull().any()]
        
        # Calculate the percentage of missing columns
        column_missing_percent = {}
        for col in missing_columns:
            # Count how many rows in 'rows_with_missing' have a missing value for this column
            missing_in_col = rows_with_missing[df.loc[rows_with_missing.index, col].isnull()].count()
            column_missing_percent[col] = (missing_in_col / count) * 100
        
        # Sort the columns by most common missing
        sorted_missing_columns = sorted(column_missing_percent.items(), key=lambda x: x[1], reverse=True)
        
        # Print the top missing columns
        top_columns = sorted_missing_columns[:(missing + 2)]  # Get x+2 columns
        top_columns_str = ', '.join([f"'{col}' - {pct:.1f}%" for col, pct in top_columns])
        print(f"  Most common: {top_columns_str}")
