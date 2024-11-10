import xgboost as xgb
import shap
import pandas as pd

# Load the trained model
model = xgb.XGBRegressor()
model.load_model('best_model_time2.xgb')  # Replace with your actual model filename

# Load the max row data
max_row = pd.read_csv('max_affected_time_row_cleaned.csv')

# Drop target and unnecessary columns ('Affected_Time', 'Affected_Distance', 'Source') from the row
X_max_row = max_row.drop(columns=['Affected_Time', 'Affected_Distance', 'Source'])


# Load a small sample of your dataset as background data for SHAP, if available
background_data = pd.read_csv('us_accidents_sample_cleaned.csv', nrows=1000).drop(columns=['Affected_Time', 'Affected_Distance', 'Source'])

# Create a SHAP explainer with background data
explainer = shap.Explainer(model, background_data, check_additivity=False)

# Calculate SHAP values for the max row
shap_values = explainer(X_max_row)

# Plot the SHAP values for the max row using a force plot
shap.initjs()
shap.force_plot(shap_values[0])

# Calculate and display the percentage contribution of each feature to the prediction
shap_contributions = shap_values.values[0]
total_contribution = sum(abs(shap_contributions))

# Calculate percentage contributions
percentage_contributions = {
    X_max_row.columns[i]: abs(shap_contributions[i]) / total_contribution * 100 for i in range(len(shap_contributions))
}

# Sort and display the percentage contributions
sorted_percentage_contributions = sorted(percentage_contributions.items(), key=lambda x: x[1], reverse=True)

print("Feature contributions in percentage:")
for feature, percentage in sorted_percentage_contributions:
    print(f"{feature}: {percentage:.2f}%")
