import pandas as pd

# Load the max row data
max_row = pd.read_csv('max_affected_time_row_cleaned.csv')

# Full list of required columns
required_columns = [
    'Affected_Distance', 'Affected_Time', 'Source', 'Latitude', 'Longitude', 
    'Temperature', 'Humidity', 'Pressure', 'Visibility', 'Wind_Speed', 'Precipitation', 
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 
    'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Sunrise_Sunset', 
    'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Percentage_of_Year', 
    'Percentage_of_Day', 'Holiday', 'After_Holiday',
    
    # State columns - includes DC; excludes HI/AK
    *['State_' + state for state in ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 
                                     'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 
                                     'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 
                                     'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 
                                     'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']],
    
    # Wind direction columns
    'WindDir_N', 'WindDir_E', 'WindDir_S', 'WindDir_W', 'WindDir_Calm', 'WindDir_Variable',

    # Weather condition columns
    'Weather_Clear', 'Weather_Cloudy', 'Weather_Fog', 'Weather_Heavy Rain', 'Weather_Light Rain', 
    'Weather_Rain', 'Weather_Snow',

    # Day columns
    'Day_Monday', 'Day_Tuesday', 'Day_Wednesday', 'Day_Thursday', 'Day_Friday', 
    'Day_Saturday', 'Day_Sunday'
]

# Ensure all required columns are in max_row with correct order
for col in required_columns:
    if col not in max_row.columns:
        max_row[col] = 0  # Add missing column with default value 0

# Reorder the columns to match the required order
max_row = max_row[required_columns]

# Save the updated max row to ensure it's correctly formatted
max_row.to_csv('max_affected_time_row_cleaned.csv', index=False)
