import xgboost as xgb
import shap
import pandas as pd

# List of IDs to exclude
excluded_ids = ['A-4810425', 'A-5053641', 'A-5399002', 'A-4014778', 'A-4574829']  # Replace with actual IDs you want to exclude

# File path
file_path = 'us_accidents.csv'

# Initialize variables for finding the max row
max_value = -float('inf')
max_row = None

# Iterate through the CSV file in chunks
chunk_size = 100000  # Adjust based on memory limits
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    # Filter out rows with IDs in the excluded_ids list
    chunk = chunk[~chunk['ID'].isin(excluded_ids)]
    
    # Drop rows with missing 'Start_Time' or 'End_Time' values
    chunk = chunk.dropna(subset=['Start_Time', 'End_Time'])

    # Convert 'Start_Time' and 'End_Time' to datetime and calculate 'Affected_Time'
    chunk['Start_Time'] = pd.to_datetime(chunk['Start_Time'].str.split('.').str[0], errors='coerce')
    chunk['End_Time'] = pd.to_datetime(chunk['End_Time'].str.split('.').str[0], errors='coerce')
    chunk['Affected_Time'] = (chunk['End_Time'] - chunk['Start_Time']).dt.total_seconds()
    
    # Drop rows where 'Affected_Time' could not be calculated due to datetime parsing issues
    chunk = chunk.dropna(subset=['Affected_Time'])

    # Find the maximum 'Affected_Time' in the current chunk
    if not chunk.empty:
        chunk_max_value = chunk['Affected_Time'].max()
        if chunk_max_value > max_value and chunk_max_value < 1000000:
            max_value = chunk_max_value
            max_row = chunk[chunk['Affected_Time'] == chunk_max_value].iloc[0]

# Check if max_row was found and save it to CSV
if max_row is not None:
    max_row.to_frame().T.to_csv('max_affected_time_row.csv', index=False)
else:
    print("No valid 'Affected_Time' value found in the dataset.")
