import numpy as np
import pandas as pd

# Load the CSV file
df = pd.read_csv("us_accidents_sample.csv")

# Step 1: Remove unnecessary columns
columns_to_remove = ['ID', 'Severity', 'End_Lat', 'End_Lng', 'Description', 'County', 
                     'Zipcode', 'Country', 'Weather_Timestamp', 'Timezone', 'Airport_Code', 'Turning_Loop']
df.drop(columns=columns_to_remove, inplace=True)

# Step 2: Rename columns
rename_columns = {
    'Start_Lat': 'Latitude',
    'Start_Lng': 'Longitude',
    'Distance(mi)': 'Affected_Distance',
    'Temperature(F)': 'Temperature',
    'Wind_Chill(F)': 'Wind_Chill',
    'Humidity(%)': 'Humidity',
    'Pressure(in)': 'Pressure',
    'Visibility(mi)': 'Visibility',
    'Wind_Speed(mph)': 'Wind_Speed',
    'Precipitation(in)': 'Precipitation'
}
df.rename(columns=rename_columns, inplace=True)

# Step 3: Create 'Affected_Time' column in seconds
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df['Affected_Time'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds()

# Step 4: Extract time-based columns
df['Day_of_Month'] = df['Start_Time'].dt.day
df['Month_of_Year'] = df['Start_Time'].dt.month
df['Day_of_Week'] = df['Start_Time'].dt.weekday + 1  # Monday = 1, Sunday = 7
df['Hour_of_Day'] = df['Start_Time'].dt.hour + 1
df['Minute_of_Hour'] = df['Start_Time'].dt.minute + 1
df['Second_of_Minute'] = df['Start_Time'].dt.second + 1

# Remove 'Start_Time' and 'End_Time' columns after extracting the information
df.drop(columns=['Start_Time', 'End_Time'], inplace=True)

# Step 5: Set precision for numerical columns and ensure trailing zeros
df['Latitude'] = df['Latitude'].map(lambda x: f"{x:.6f}")
df['Longitude'] = df['Longitude'].map(lambda x: f"{x:.6f}")
df['Affected_Distance'] = df['Affected_Distance'].map(lambda x: f"{x:.3f}")
df['Pressure'] = df['Pressure'].map(lambda x: f"{x:.2f}")
df['Precipitation'] = df['Precipitation'].map(lambda x: f"{x:.2f}")
df['Temperature'] = df['Temperature'].map(lambda x: f"{x:.1f}")
df['Wind_Chill'] = df['Wind_Chill'].map(lambda x: f"{x:.1f}")
df['Wind_Speed'] = df['Wind_Speed'].map(lambda x: f"{x:.1f}")
df['Humidity'] = df['Humidity'].map(lambda x: f"{x:.0f}")
df['Visibility'] = df['Visibility'].map(lambda x: f"{x:.0f}")

# Step 6: Replace all missing values with 'NA'
df.fillna('NA', inplace=True)

# Step 7: Convert boolean columns ('False'/'True') to 0/1
bool_columns = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 
                'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']
for col in bool_columns:
    df[col] = np.where(df[col] == 'True', 1, 0)

# Step 8: Convert twilight columns ('Night'/'Day') to 0/1
twilight_columns = ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
for col in twilight_columns:
    df[col] = np.where(df[col] == 'Day', 1, 0)

# Step 9: Convert source column ('Source1' / 'Source2') to 0/1
df['Source'] = np.where(df['Source'] == 'Source2', 1, 0)

# Step 10: Remove leading spaces in 'Street' column
df['Street'] = df['Street'].str.lstrip()

# Save the cleaned data to a new CSV file
df.to_csv("us_accidents_sample_cleaned.csv", index=False)
