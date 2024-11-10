import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import random

class ModelOptimizer:
    def __init__(self, model_filename: str):
        self.model = self.load_model(model_filename)
        
        # Get the model's feature names
        self.feature_names = self.model.feature_names
        if self.feature_names is None:
            print("Warning: Model doesn't have feature names stored. Using default order.")
            # Use the order from your training data here
            self.feature_names = [
                'Latitude', 'Longitude', 'Temperature', 'Humidity', 'Pressure', 
                'Visibility', 'Wind_Speed', 'Precipitation', 'Amenity', 'Bump', 
                'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 
                'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 
                'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 
                'Astronomical_Twilight', 'Percentage_of_Year', 'Percentage_of_Day', 
                'Holiday', 'After_Holiday', "Highway", 'State_AL', 'State_AR', 'State_AZ', 
                'State_CA', 'State_CO', 'State_CT', 'State_DC', 'State_DE', 
                'State_FL', 'State_GA', 'State_IA', 'State_ID', 'State_IL', 
                'State_IN', 'State_KS', 'State_KY', 'State_LA', 'State_MA', 
                'State_MD', 'State_ME', 'State_MI', 'State_MN', 'State_MO', 
                'State_MS', 'State_MT', 'State_NC', 'State_ND', 'State_NE', 
                'State_NH', 'State_NJ', 'State_NM', 'State_NV', 'State_NY', 
                'State_OH', 'State_OK', 'State_OR', 'State_PA', 'State_RI', 
                'State_SC', 'State_SD', 'State_TN', 'State_TX', 'State_UT', 
                'State_VA', 'State_VT', 'State_WA', 'State_WI', 'State_WV', 
                'State_WY', 'WindDir_N', 'WindDir_E', 'WindDir_S', 'WindDir_W', 
                'WindDir_Calm', 'WindDir_Variable', 'Weather_Clear', 
                'Weather_Cloudy', 'Weather_Fog', 'Weather_Heavy Rain', 
                'Weather_Light Rain', 'Weather_Rain', 'Weather_Snow', 'Day_Monday', 
                'Day_Tuesday', 'Day_Wednesday', 'Day_Thursday', 'Day_Friday', 
                'Day_Saturday', 'Day_Sunday'
            ]
        
        print("\nModel feature names:")
        print(self.feature_names)
        
        # Define feasible ranges for continuous variables
        self.continuous_ranges = {
            "Latitude": (24.396308, 49.384358),      # Approximate continental US bounds
            "Longitude": (-125.000000, -66.934570),  # Approximate continental US bounds
            "Temperature": (-30, 120),               # Reasonable F range for continental US
            "Humidity": (0, 100),
            "Pressure": (27, 32),                    # Reasonable inHg range
            "Visibility": (0, 10),
            "Wind_Speed": (0, 200),
            "Precipitation": (0, 10),
            "Percentage_of_Year": (0, 1),
            "Percentage_of_Day": (0, 1)
        }
        
        # Define mutually exclusive categorical groups
        self.categorical_groups = {
            "State": [f"State_{state}" for state in [
                'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ',
                'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD',
                'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
            ]],
            "WindDir": ["WindDir_N", "WindDir_E", "WindDir_S", "WindDir_W", 
                       "WindDir_Calm", "WindDir_Variable"],
            "Weather": ["Weather_Clear", "Weather_Cloudy", "Weather_Fog",
                       "Weather_Heavy Rain", "Weather_Light Rain", "Weather_Rain",
                       "Weather_Snow"],
            "Day": ["Day_Monday", "Day_Tuesday", "Day_Wednesday",
                    "Day_Thursday", "Day_Friday", "Day_Saturday",
                    "Day_Sunday"]
        }
        
        # Define boolean features
        self.boolean_features = [
            "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
            "No_Exit", "Railway", "Roundabout", "Station", "Stop",
            "Traffic_Calming", "Traffic_Signal", "Sunrise_Sunset",
            "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
            "Holiday", "After_Holiday", "Highway"
        ]

    @staticmethod
    def load_model(model_filename: str) -> xgb.Booster:
        model = xgb.Booster()
        model.load_model(model_filename)
        print(f"Model loaded from {model_filename}")
        return model

    def generate_random_sample(self) -> pd.DataFrame:
        """Generate a random sample respecting all constraints"""
        # Initialize all features to 0
        data = {feature: [0] for feature in self.feature_names}
        
        # Generate continuous variables
        for feature, (min_val, max_val) in self.continuous_ranges.items():
            data[feature] = [random.uniform(min_val, max_val)]
        
        # Generate boolean variables
        for feature in self.boolean_features:
            data[feature] = [random.choice([0, 1])]
        
        # Handle mutually exclusive categorical groups
        for group_name, features in self.categorical_groups.items():
            chosen_feature = random.choice(features)
            for feature in features:
                data[feature] = [1 if feature == chosen_feature else 0]
        
        # Create DataFrame with exact feature order
        return pd.DataFrame(data, columns=self.feature_names)

    def predict_sample(self, sample_df: pd.DataFrame) -> float:
        """Make prediction for a single sample"""
        dmatrix = xgb.DMatrix(sample_df)
        prediction = self.model.predict(dmatrix)
        return prediction[0]

    def random_search(self, n_iterations: int, top_k: int = 5) -> List[Tuple[pd.DataFrame, float]]:
        """Perform random search and return top k results"""
        results = []
        
        for i in range(n_iterations):
            sample_df = self.generate_random_sample()
            prediction = self.predict_sample(sample_df)
            results.append((sample_df, prediction))
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1} iterations. Current best: {max(results, key=lambda x: x[1])[1]}")
        
        # Sort by prediction value and return top k
        top_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        return top_results

    @staticmethod
    def print_result(sample_df: pd.DataFrame, prediction: float):
        """Pretty print a single result"""
        print(f"\nPrediction: {prediction}")
        print("\nKey feature values:")
        
        # Print non-zero categorical variables
        for column in sample_df.columns:
            value = sample_df[column].iloc[0]
            if value == 1 and ("State_" in column or "Weather_" in column or 
                             "WindDir_" in column or "Day_" in column):
                print(f"{column}: {value}")
        
        # Print continuous variables
        continuous_vars = ["Temperature", "Humidity", "Pressure", "Visibility", 
                         "Wind_Speed", "Precipitation"]
        for var in continuous_vars:
            print(f"{var}: {sample_df[var].iloc[0]:.2f}")
        
        # Print non-zero boolean features
        boolean_features = ["Amenity", "Bump", "Crossing", "Give_Way",
                          "Junction", "No_Exit", "Railway", "Roundabout",
                          "Station", "Stop", "Traffic_Calming", 
                          "Traffic_Signal", "Holiday", "After_Holiday", "Highway"]
        for feature in boolean_features:
            value = sample_df[feature].iloc[0]
            if value == 1:
                print(f"{feature}: {value}")

def main():
    # Initialize optimizer
    model_filename = "best_model_distance.xgb"
    optimizer = ModelOptimizer(model_filename)
    
    # Perform random search
    n_iterations = 1000
    top_k = 5
    print(f"\nPerforming random search with {n_iterations} iterations...")
    top_results = optimizer.random_search(n_iterations, top_k)
    
    # Print top results
    print(f"\nTop {top_k} configurations found:")
    for i, (sample_df, prediction) in enumerate(top_results, 1):
        print(f"\n--- Result {i} ---")
        optimizer.print_result(sample_df, prediction)

if __name__ == "__main__":
    main()