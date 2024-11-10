import xgboost as xgb
import pandas as pd

# Load model from file
def load_model(model_filename):
    model = xgb.Booster()
    model.load_model(model_filename)
    print(f"Model loaded from {model_filename}")
    return model

# Load feature importances from CSV (optional)
def load_feature_importances(importance_filename):
    importance_df = pd.read_csv(importance_filename)
    print(f"Feature importances loaded from {importance_filename}")
    return importance_df

# Using the loaded model
def main():
    model_filename = "best_model_time.xgb"
    feature_importance_filename = "feature_importances_time.csv"
    
    # Load model and feature importances
    model = load_model(model_filename)
    feature_importances = load_feature_importances(feature_importance_filename)
    
    # Test an example input for prediction
    example_input = pd.DataFrame({
        "Latitude": [39.865147],
        "Longitude": [-84.058723],

        "Temperature": [36.9],
        "Humidity": [91],
        "Pressure": [29.68],
        "Visibility": [10],
        "Wind_Speed": [0.0],
        "Precipitation": [0.02],

        "Amenity": [0],
        "Bump": [0],
        "Crossing": [0],
        "Give_Way": [0],
        "Junction": [0],
        "No_Exit": [0],
        "Railway": [0],
        "Roundabout": [0],
        "Station": [0],
        "Stop": [0],
        "Traffic_Calming": [0],
        "Traffic_Signal": [0],

        "Sunrise_Sunset": [0.0],
        "Civil_Twilight": [0.0],
        "Nautical_Twilight": [0.0],
        "Astronomical_Twilight": [0.0],

        "Percentage_of_Year": [0.106557],
        "Percentage_of_Day": [0.240278],

        "Holiday": [0],
        "After_Holiday": [0],

        "State_AL": [0], "State_AR": [0], "State_AZ": [0], "State_CA": [0],
        "State_CO": [0], "State_CT": [0], "State_DC": [0], "State_DE": [0],
        "State_FL": [0], "State_GA": [0], "State_IA": [0], "State_ID": [0],
        "State_IL": [0], "State_IN": [0], "State_KS": [0], "State_KY": [0],
        "State_LA": [0], "State_MA": [0], "State_MD": [0], "State_ME": [0],
        "State_MI": [0], "State_MN": [0], "State_MO": [0], "State_MS": [0],
        "State_MT": [0], "State_NC": [0], "State_ND": [0], "State_NE": [0],
        "State_NH": [0], "State_NJ": [0], "State_NM": [0], "State_NV": [0],
        "State_NY": [0], "State_OH": [1], "State_OK": [0], "State_OR": [0],
        "State_PA": [0], "State_RI": [0], "State_SC": [0], "State_SD": [0],
        "State_TN": [0], "State_TX": [0], "State_UT": [0], "State_VA": [0],
        "State_VT": [0], "State_WA": [0], "State_WI": [0], "State_WV": [0],
        "State_WY": [0],

        "WindDir_N": [0], "WindDir_E": [0], "WindDir_S": [0], "WindDir_W": [0],
        "WindDir_Calm": [1], "WindDir_Variable": [0],
        "Weather_Clear": [0], "Weather_Cloudy": [0], "Weather_Fog": [0],
        "Weather_Heavy Rain": [0], "Weather_Light Rain": [1], "Weather_Rain": [0],
        "Weather_Snow": [0],

        "Day_Monday": [1], "Day_Tuesday": [0], "Day_Wednesday": [0],
        "Day_Thursday": [0], "Day_Friday": [0], "Day_Saturday": [0],
        "Day_Sunday": [0]
    })

    example_dmatrix = xgb.DMatrix(example_input)
    prediction = model.predict(example_dmatrix)
    print("Prediction for example input:", prediction[0])
    
    # Display loaded feature importances
    print("Feature Importances:")
    print(feature_importances)

if __name__ == "__main__":
    main()