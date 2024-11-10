import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'us_accidents_sample_cleaned.csv'
df = pd.read_csv(file_path)

# Mapping of wind directions to the new categories
wind_direction_map = {
    'N': 'N', 
    'North': 'N', 
    'NW': 'N', 
    'NNE': 'N', 
    'NNW': 'N', 

    'E': 'E', 
    'East': 'E',
    'NE': 'E', 
    'ENE': 'E', 
    'ESE': 'E', 

    'S': 'S', 
    'South': 'S', 
    'SE': 'S', 
    'SSE': 'S', 
    'SSW': 'S', 

    'W': 'W', 
    'West': 'W',
    'SW': 'W', 
    'WNW': 'W', 
    'WSW': 'W', 

    'CALM': 'Calm', 
    'Calm': 'Calm', 

    'VAR': 'Variable', 
    'Variable': 'Variable',

    None: 'NA'
}

# Apply the mapping to the 'Wind_Direction' column
df['Wind_Direction'] = df['Wind_Direction'].map(wind_direction_map)

# Get the unique values in the 'Wind_Direction' column and sort them alphabetically
unique_values = sorted(df['Wind_Direction'].dropna().unique())

# Count of NA (missing) values
na_count = df['Wind_Direction'].isna().sum()

# Print the unique values
print("Unique values for 'Wind_Direction':")
print(unique_values)

# Print the length of unique values (excluding NA)
print("\nNumber of unique values (excluding NA):", len(unique_values))

# Print the count for each unique value
print("\nCount of each unique value:")
print(f"NA: {na_count}")

for value in unique_values:
    count = df['Wind_Direction'].value_counts().get(value, 0)
    print(f"{value}: {count}")
