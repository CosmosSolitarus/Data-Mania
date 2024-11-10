import xgboost as xgb
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import optuna

# Save model to file
def save_model(model, model_filename):
    model.save_model(model_filename)
    print(f"Model saved to {model_filename}")

# Save feature importances to CSV
def save_feature_importances(model, importance_filename):
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df.to_csv(importance_filename, index=False)
    print(f"Feature importances saved to {importance_filename}")

# Hyperparameter tuning function
def tune_hyperparameters(X, y, n_trials=50):
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0)
        }
        
        dmatrix = xgb.DMatrix(X, label=y)
        cv_results = xgb.cv(
            params=params,
            dtrain=dmatrix,
            num_boost_round=1000,
            nfold=5,
            early_stopping_rounds=10,
            metrics='rmse',
            as_pandas=True,
            seed=0
        )
        return cv_results['test-rmse-mean'].min()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    return study.best_params

# Training function with early stopping
def train_model(X, y, params, n_splits=5, num_boost_round=1000, early_stopping_rounds=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    dmatrix = xgb.DMatrix(X, label=y)
    
    cv_results = xgb.cv(
        params=params,
        dtrain=dmatrix,
        num_boost_round=num_boost_round,
        folds=kf,
        early_stopping_rounds=early_stopping_rounds,
        metrics='rmse',
        as_pandas=True,
        seed=0
    )
    
    model = xgb.train(params, dmatrix, num_boost_round=cv_results.shape[0]) # Use best boosting rounds
    return model, cv_results

# Feature importance visualization function
def plot_feature_importance(model):
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("Feature Importances:")
    print(importance_df)
    importance_df.plot(kind='bar', x='Feature', y='Importance', title='Feature Importances')
    plt.show()

# Main function
def main():
    # Hyperparameters and other constants
    n_splits = 5
    num_boost_round = 1000
    early_stopping_rounds = 10
    
    # Load data
    data = pd.read_csv("us_accidents_sample_cleaned.csv")
    X = data.drop(columns=["Affected_Time", "Affected_Distance", "Source"])
    y = data["Affected_Time"]
    
    # Tune hyperparameters
    tuned_params = tune_hyperparameters(X, y)
    tuned_params.update({'tree_method': 'hist', 'device': 'cuda', 'eval_metric': 'rmse'})
    
    # Train model
    model, cv_results = train_model(X, y, tuned_params, n_splits, num_boost_round, early_stopping_rounds)
    
    # Save the model and feature importances
    model_filename = "best_model.xgb"
    feature_importance_filename = "feature_importances.csv"
    save_model(model, model_filename)
    save_feature_importances(model, feature_importance_filename)

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

    # Display cross-validation results and feature importance
    print("Cross-validation results:")
    print(cv_results)
    plot_feature_importance(model)
    
if __name__ == "__main__":
    main()

