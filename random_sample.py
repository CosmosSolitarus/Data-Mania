import pandas as pd
import time

start_time = time.time()

# Load the CSV file
csv_file = 'us_accidents.csv'
df = pd.read_csv(csv_file)

# Randomly select n rows
sample_data = df.sample(n=200000, random_state=1)

# Save the sample to a new CSV file
sample_data.to_csv('us_accidents_sample.csv', index=False)

end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds ({(end_time - start_time)/60:.2f} minutes)")

print("Random Sampling Complete.")
