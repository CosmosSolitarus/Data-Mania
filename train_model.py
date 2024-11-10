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
    
    model = xgb.train(params, dmatrix, num_boost_round=cv_results.shape[0])
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
    model_filename = "best_model_distance.xgb"
    feature_importance_filename = "feature_importances_distance.csv"
    save_model(model, model_filename)
    save_feature_importances(model, feature_importance_filename)

    # Display cross-validation results and feature importance
    print("Cross-validation results:")
    print(cv_results)
    plot_feature_importance(model)
    
if __name__ == "__main__":
    main()

