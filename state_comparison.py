import pandas as pd

# Load your data
df = pd.read_csv('us_accidents_sample_cleaned.csv')

# Count of 1's in 'State_LA' and 'State_SC'
state_la_count = df['State_LA'].sum()
state_sc_count = df['State_SC'].sum()

# Averages of 'Affected_Time' and 'Affected_Distance' for rows where 'State_LA' == 1
state_la_affected_time_avg = df.loc[df['State_LA'] == 1, 'Affected_Time'].mean()
state_la_affected_distance_avg = df.loc[df['State_LA'] == 1, 'Affected_Distance'].mean()

# Averages of 'Affected_Time' and 'Affected_Distance' for rows where 'State_SC' == 1
state_sc_affected_time_avg = df.loc[df['State_SC'] == 1, 'Affected_Time'].mean()
state_sc_affected_distance_avg = df.loc[df['State_SC'] == 1, 'Affected_Distance'].mean()

# Print results
print(f"Count of 1's in 'State_LA': {state_la_count}")
print(f"Count of 1's in 'State_SC': {state_sc_count}")
print(f"Average 'Affected_Time' for 'State_LA': {state_la_affected_time_avg}")
print(f"Average 'Affected_Distance' for 'State_LA': {state_la_affected_distance_avg}")
print(f"Average 'Affected_Time' for 'State_SC': {state_sc_affected_time_avg}")
print(f"Average 'Affected_Distance' for 'State_SC': {state_sc_affected_distance_avg}")
