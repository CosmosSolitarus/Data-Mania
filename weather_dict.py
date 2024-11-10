import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'us_accidents_sample_cleaned.csv'
df = pd.read_csv(file_path)

# Define category mappings
categories = {
    "CLEAR": ["Clear", "Fair", "Fair / Windy"],
    "CLOUDY": [
        "Cloudy", "Cloudy / Windy", "Scattered Clouds", "Overcast", 
        "Partly Cloudy", "Partly Cloudy / Windy", "Mostly Cloudy", "Mostly Cloudy / Windy"
    ],
    "FOG": [
        "Fog", "Light Freezing Fog", "Patches of Fog", "Haze", 
        "Haze / Windy", "Mist", "Shallow Fog", "Smoke"
    ],
    "RAIN": ["Heavy Drizzle", "N/A Precipitation", "Rain", "Rain / Windy", "Light Thunderstorms and Rain"],
    "HEAVY RAIN": [
        "Heavy Rain", "Heavy T-Storm", "Heavy T-Storm / Windy", 
        "Heavy Thunderstorms and Rain", "T-Storm", "T-Storm / Windy", 
        "Thunderstorm", "Thunderstorms and Rain", "Thunder", "Thunder in the Vicinity"
    ],
    "LIGHT RAIN": [
        "Light Drizzle", "Showers in the Vicinity", "Light Freezing Rain", 
        "Light Rain", "Light Rain / Windy", "Light Rain with Thunder", "Drizzle", "Drizzle and Fog"
    ],
    "SNOW": [
        "Snow", "Snow / Windy", "Snow and Sleet", "Heavy Snow", "Heavy Snow / Windy", 
        "Light Snow", "Light Snow / Windy", "Wintry Mix", "Hail", "Blowing Snow / Windy"
    ]
}

# Create a new column 'Weather_Category' based on the mappings
df['Weather_Category'] = df['Weather_Condition'].map(lambda x: next(
    (key for key, values in categories.items() if x in values), 'NA' if pd.isna(x) else x
))

# Get the counts for each main weather category
category_counts = df['Weather_Category'].value_counts().sort_index()

# Print results
print("Unique Weather Categories and Counts:\n")
for category, count in category_counts.items():
    print(f"{category}: {count}")
