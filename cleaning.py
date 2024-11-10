import numpy as np
import pandas as pd
from workalendar.usa import UnitedStates
import time
import gc

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

def process_chunk(chunk, holidays_dict, wind_direction_map, weather_condition_reverse_map):
    # Step 1: Remove unnecessary columns
    columns_to_remove = ['ID', 'Severity', 'End_Lat', 'End_Lng', 'Description', 
                        'County', 'City', 'Zipcode', 'Country', 'Weather_Timestamp', 
                        'Timezone', 'Airport_Code', 'Wind_Chill(F)', 'Turning_Loop']
    chunk.drop(columns=[col for col in columns_to_remove if col in chunk.columns], inplace=True)

    # Step 2: Rename columns
    rename_columns = {
        'Start_Lat': 'Latitude',
        'Start_Lng': 'Longitude',
        'Distance(mi)': 'Affected_Distance',
        'Temperature(F)': 'Temperature',
        'Humidity(%)': 'Humidity',
        'Pressure(in)': 'Pressure',
        'Visibility(mi)': 'Visibility',
        'Wind_Speed(mph)': 'Wind_Speed',
        'Precipitation(in)': 'Precipitation'
    }
    chunk.rename(columns=rename_columns, inplace=True)

    # Step 3: Map and combine 'Weather_Condition' and 'Wind_Direction'
    chunk['Wind_Direction'] = chunk['Wind_Direction'].map(wind_direction_map).fillna('')
    chunk['Weather_Condition'] = chunk['Weather_Condition'].map(weather_condition_reverse_map).fillna('')

    # Step 4: Set 'Wind_Speed' to 0 where 'Wind_Direction' is 'Calm' and 'Wind_Speed' is blank
    chunk.loc[(chunk['Wind_Direction'] == 'Calm') & (chunk['Wind_Speed'].isna()), 'Wind_Speed'] = 0

    # Step 5: Set 'Precipitation' to 0 where 'Weather_Condition' is 'Clear', 'Cloudy', or 'Fog' and 'Precipitation' is blank
    chunk.loc[(chunk['Weather_Condition'].isin(['Clear', 'Cloudy', 'Fog'])) & (chunk['Precipitation'].isna()), 'Precipitation'] = 0

    # Step 6: Handle datetime columns and create 'Affected_Time'
    chunk['Start_Time'] = chunk['Start_Time'].str.split('.').str[0]
    chunk['End_Time'] = chunk['End_Time'].str.split('.').str[0]
    chunk['Start_Time'] = pd.to_datetime(chunk['Start_Time'])
    chunk['End_Time'] = pd.to_datetime(chunk['End_Time'])
    chunk['Affected_Time'] = (chunk['End_Time'] - chunk['Start_Time']).dt.total_seconds()

    # Step 7: Extract time-based columns
    chunk['Day_of_Month'] = chunk['Start_Time'].dt.day
    chunk['Month_of_Year'] = chunk['Start_Time'].dt.month
    chunk['Day_of_Week'] = chunk['Start_Time'].dt.weekday + 1
    chunk['Hour_of_Day'] = chunk['Start_Time'].dt.hour
    chunk['Minute_of_Hour'] = chunk['Start_Time'].dt.minute
    chunk['Second_of_Minute'] = chunk['Start_Time'].dt.second

    # Step 8: Calculate 'Percentage_of_Year'
    def percentage_of_year(row):
        day_of_year = pd.Timestamp(year=row['Start_Time'].year, month=row['Month_of_Year'], day=row['Day_of_Month']).day_of_year
        days_in_year = 366 if ((row['Start_Time'].year % 4 == 0) and (row['Start_Time'].year % 100 != 0 or row['Start_Time'].year % 400 == 0)) else 365
        return day_of_year / days_in_year

    chunk['Percentage_of_Year'] = chunk.apply(percentage_of_year, axis=1)

    # Step 9: Calculate 'Percentage_of_Day'
    def percentage_of_day(row):
        total_seconds_in_day = 86400
        seconds_in_day = (row['Hour_of_Day'] * 3600) + (row['Minute_of_Hour'] * 60) + row['Second_of_Minute']
        return seconds_in_day / total_seconds_in_day

    chunk['Percentage_of_Day'] = chunk.apply(percentage_of_day, axis=1)

    # Step 10: Drop time-related columns
    chunk.drop(columns=['Day_of_Month', 'Month_of_Year', 'Hour_of_Day', 'Minute_of_Hour', 'Second_of_Minute'], inplace=True)

    # Step 11: Add 'Holiday' and 'After_Holiday' columns
    chunk['Holiday'] = chunk['Start_Time'].apply(lambda x: x.date() in holidays_dict[x.year]).astype(int)
    chunk['After_Holiday'] = chunk['Start_Time'].apply(lambda x: (x - pd.Timedelta(days=1)).date() in holidays_dict[x.year]).astype(int)

    # Step 12: Drop 'Start_Time' and 'End_Time'
    chunk.drop(columns=['Start_Time', 'End_Time'], inplace=True)

    # Step 13: Set precision for numerical columns
    numeric_columns = {
        'Percentage_of_Year': 6,
        'Percentage_of_Day': 6,
        'Latitude': 6,
        'Longitude': 6,
        'Affected_Distance': 3,
        'Pressure': 2,
        'Precipitation': 2,
        'Temperature': 1,
        'Wind_Speed': 1,
        'Humidity': 0,
        'Visibility': 0
    }

    for col, decimals in numeric_columns.items():
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        formatter = "{{:.{}f}}".format(decimals)
        chunk[col] = chunk[col].apply(lambda x: formatter.format(x).zfill(decimals + 2) if pd.notna(x) else None)

    # Step 14: One-hot-encode categorical columns
    day_of_week_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    chunk['Day_of_Week'] = chunk['Day_of_Week'].map(day_of_week_map)
    chunk = pd.get_dummies(chunk, columns=['State', 'Wind_Direction', 'Weather_Condition', 'Day_of_Week'], prefix=['State', 'WindDir', 'Weather', 'Day'], drop_first=False)

    # Convert boolean columns to 0 or 1
    chunk = chunk.apply(lambda x: x.map({True: 1, False: 0}) if x.dtype == bool else x)

    # Map day/night columns
    for col in ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']:
        chunk[col] = chunk[col].map({'Day': 1, 'Night': 0})

    # Map 'Source' column
    chunk['Source'] = chunk['Source'].map({'Source1': 0, 'Source2': 1})

    # Move 'Affected_Distance' and 'Affected_Time' to the leftmost columns
    cols = ['Affected_Distance', 'Affected_Time'] + [col for col in chunk.columns if col not in ['Affected_Distance', 'Affected_Time']]
    chunk = chunk[cols]

    # Remove rows with NaNs and drop 'Street' column
    chunk.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    chunk.dropna(inplace=True)
    chunk.drop(columns=['Street'], inplace=True)

    # Enforce required columns
    for column in required_columns:
        if column not in chunk.columns:
            chunk[column] = 0  # Default to 0 or a suitable value for missing columns

    chunk = chunk.reindex(columns=required_columns)

    return chunk[required_columns]  # Ensure the chunk has the exact columns in the correct order

def main():
    start_time = time.time()
    
    # Initialize constants and mappings as before
    chunk_size = 100000
    start_year = 2015
    end_year = 2024
    
    # Initialize the workalendar object only for holiday calculation
    cal = UnitedStates()
    
    # Rest of the mappings initialization remains the same
    wind_direction_map = {
        'N': 'N', 'North': 'N', 'NW': 'N', 'NNE': 'N', 'NNW': 'N',
        'E': 'E', 'East': 'E', 'NE': 'E', 'ENE': 'E', 'ESE': 'E',
        'S': 'S', 'South': 'S', 'SE': 'S', 'SSE': 'S', 'SSW': 'S',
        'W': 'W', 'West': 'W', 'SW': 'W', 'WNW': 'W', 'WSW': 'W',
        'CALM': 'Calm', 'Calm': 'Calm',
        'VAR': 'Variable', 'Variable': 'Variable',
        None: 'NA'
    }
    
    weather_condition_map = {
        "Clear": ["Clear", "Fair", "Fair / Windy"],
        "Cloudy": ["Cloudy", "Cloudy / Windy", "Scattered Clouds", "Overcast", "Partly Cloudy",
                "Partly Cloudy / Windy", "Mostly Cloudy", "Mostly Cloudy / Windy"],
        "Fog": ["Fog", "Light Freezing Fog", "Patches of Fog", "Haze", "Haze / Windy", "Mist", 
                "Shallow Fog", "Smoke"],
        "Rain": ["Heavy Drizzle", "N/A Precipitation", "Rain", "Rain / Windy", 
                "Light Thunderstorms and Rain"],
        "Heavy Rain": ["Heavy Rain", "Heavy T-Storm", "Heavy T-Storm / Windy", 
                    "Heavy Thunderstorms and Rain", "T-Storm", "T-Storm / Windy", 
                    "Thunderstorm", "Thunderstorms and Rain", "Thunder", 
                    "Thunder in the Vicinity"],
        "Light Rain": ["Light Drizzle", "Showers in the Vicinity", "Light Freezing Rain", 
                    "Light Rain", "Light Rain / Windy", "Light Rain with Thunder", 
                    "Drizzle", "Drizzle and Fog"],
        "Snow": ["Snow", "Snow / Windy", "Snow and Sleet", "Heavy Snow", "Heavy Snow / Windy", 
                "Light Snow", "Light Snow / Windy", "Wintry Mix", "Hail", 
                "Blowing Snow / Windy"]
    }

    weather_condition_reverse_map = {cond: key for key, cond_list in weather_condition_map.items() for cond in cond_list}

    # Pre-calculate holidays using cal
    holidays_dict = {}
    for year in range(start_year, end_year + 1):
        holidays_list = cal.holidays(year)
        holidays_dict[year] = set([holiday[0] for holiday in holidays_list])
    
    # Process the file in chunks
    first_chunk = True
    for chunk in pd.read_csv("us_accidents.csv", chunksize=chunk_size):
        processed_chunk = process_chunk(chunk, holidays_dict, wind_direction_map, weather_condition_reverse_map)
        
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        processed_chunk.to_csv("us_accidents_cleaned.csv", mode=mode, index=False, header=header)
        first_chunk = False
        
        del processed_chunk
        gc.collect()

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds ({(end_time - start_time)/60:.2f} minutes)")

if __name__ == "__main__":
    main()