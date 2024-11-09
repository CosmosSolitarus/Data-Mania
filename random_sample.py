import pandas as pd

# Load the CSV file
csv_file = 'us_accidents.csv'
df = pd.read_csv(csv_file)

# Remove the specified columns
df = df.drop(columns=['Severity', 'Country', 'Description', 'Turning_Loop'], errors='ignore')

# Randomly select n rows
sample_data = df.sample(n=250000, random_state=1)

# Save the sample to a new CSV file
sample_data.to_csv('us_accidents_sample_MINI.csv', index=False)
