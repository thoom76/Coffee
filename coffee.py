#!/bin/python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

categorical_schema = {
    'coffee_machine': {'9BARISTA': 0},
    'coffee_grinder': {'PIETRO': 0},
    'stove_type': {'CERAMIC': 0},
    'coffee_category': {
        'HUILA-COLOMBIA;PINK_BOURBON;ANAEROBIC_DOUBLE_FERMENTATION;LIGHT': 0,
        'JUKIA_PARK-UGANDA;SL-14;NATURAL_ANAEROBIC;MEDIUM': 1
    },
    'water_filtered': {False: 0}
}

inverted_categorical_schema = {
    category: {v: k for k, v in values.items()}
    for category, values in categorical_schema.items()
}

# Load from file
df = pd.read_csv('coffee_data.csv', sep=';')

# Combine coffee-related columns into a single category so the model won't fuck-up
df['coffee_category'] = df['coffee_origin'] + ';' + df['coffee_variety'] + ';' + \
                        df['coffee_processing_type'] + ';' + df['coffee_roast_level']
# Drop the original columns since they are now combined
df.drop(['coffee_origin', 'coffee_variety', 'coffee_processing_type', 'coffee_roast_level'], axis=1, inplace=True)
# Drop those columns since they are not really accurately measured..
df.drop(['heat_setting', 'water_weight'], axis=1, inplace=True)

# List of columns to encode
label_columns = [*categorical_schema]

# Encoding categorical columns
for col in label_columns:
    df[col] = [categorical_schema[col][value] for value in df[col]]

# Define Inputs and Outputs
inputs = ['coffee_machine', 'coffee_grinder', 'stove_type', 'coffee_category', 'coffee_roasted_days_ago', 'coffee_dose', 'grind_size', 'water_filtered']
outputs = ['pre-infusion_time', 'extraction_time', 'total_brew_time', 'taste_score', 'bitterness_score', 'sourness_score', 'aroma_intensity', 'crema_quality']

# Normalize numerical features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[inputs])

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

# Add PCA results to the dataframe
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Detect outliers using IQR on PCA components
Q1 = df[['PCA1', 'PCA2']].quantile(0.25)
Q3 = df[['PCA1', 'PCA2']].quantile(0.75)
IQR = Q3 - Q1

# Define outliers as points outside 1.5 * IQR
outlier_mask = (
    (df['PCA1'] < (Q1['PCA1'] - 1.5 * IQR['PCA1'])) | (df['PCA1'] > (Q3['PCA1'] + 1.5 * IQR['PCA1'])) |
    (df['PCA2'] < (Q1['PCA2'] - 1.5 * IQR['PCA2'])) | (df['PCA2'] > (Q3['PCA2'] + 1.5 * IQR['PCA2']))
)

# Filter out outliers
df_cleaned = df[~outlier_mask]

# Print the number of outliers and the cleaned dataset size
print(f"Number of outliers detected: {outlier_mask.sum()}")
print(f"Original dataset size: {df.shape[0]}, Cleaned dataset size: {df_cleaned.shape[0]}")

# Plot the PCA components to visualize potential outliers
plt.figure(figsize=(10, 8))

# Define normal points and outliers
normal_points = df[~outlier_mask]
outliers = df[outlier_mask]

# Scatter plot for normal points
plt.scatter(normal_points['PCA1'], normal_points['PCA2'], alpha=0.7, label='Normal Points', c='blue')

# Scatter plot for outliers
plt.scatter(outliers['PCA1'], outliers['PCA2'], alpha=0.9, label='Outliers', c='red', marker='x', s=100)

# Add title, labels, and legend
plt.title('PCA Components - Outlier Visualization', fontsize=16)
plt.xlabel('PCA1', fontsize=14)
plt.ylabel('PCA2', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

X = df_cleaned[inputs]
y = df_cleaned[outputs]

# Model for hyperparameter tweaking
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [i for i in range(50, 200, 10)],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
}
loo = LeaveOneOut()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=loo, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Calculate LOOCV MSE as percentage error
y_true = y.values  # Actual values
y_pred = grid_search.predict(X)  # Predicted values

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_true - y_pred) ** 2)

# Calculate the mean of the actual values (for percentage calculation)
y_true_mean = np.mean(y_true)

# Calculate the percentage error
percentage_error = (mse / y_true_mean) * 100

print(f"Best Parameters: {best_params}")
print(f"Best Mean CV Error: {best_score}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Percentage Error: {percentage_error}%")
print()

# Train the model
# TODO: what amount of estimators do we need? 
best_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    bootstrap=best_params['bootstrap'],
    random_state=42
)
errors = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    errors.append(np.mean((y_test.values - y_pred) ** 2))

def objective(params):
    # Extract parameters from the input list
    coffee_machine, coffee_grinder, stove_type, coffee_category, coffee_roasted_days_ago, coffee_dose, grind_size, water_filtered = params

    # Create the input dataframe for prediction
    input_data = {
        'coffee_machine': [coffee_machine],
        'coffee_grinder': [coffee_grinder],
        'stove_type': [stove_type],
        # 'heat_setting': [heat_setting],
        'coffee_category': [coffee_category],
        'coffee_roasted_days_ago': [coffee_roasted_days_ago],
        'coffee_dose': [coffee_dose],
        'grind_size': [grind_size],
        'water_filtered': [water_filtered],
        # 'water_weight': [water_weight]
    }

    input_df = pd.DataFrame(input_data)

    # Get model predictions (all outputs for the given input)
    prediction = best_model.predict(input_df)

    # The first value in the prediction is taste_score and the second is bitterness_score
    pre_infusion_time = prediction[0][0]
    extraction_time = prediction[0][1]
    total_brew_time = prediction[0][2]
    taste_score = prediction[0][3]  
    bitterness_score = prediction[0][4]
    sourness_score = prediction[0][5]
    aronma_intensity = prediction[0][6]
    creama_quality = prediction[0][7] 

    sourness_penalty = sourness_score ** 1.55 if sourness_score > 5.5 else 0 
    bitterness_penalty = bitterness_score ** 1.55 if bitterness_score > 6.0 else 0

    # (minimizing the negative is equivalent to maximizing the positive)
    return -((0.9 * taste_score) + (0.1 * creama_quality)) * 5 + sourness_penalty + bitterness_penalty

# TODO: We can limit this to only contain the devices / coffee etc what we currently have, but still train the model on all the data :)
search_space = [
    Categorical([*categorical_schema['coffee_machine'].values()], name='coffee_machine'),
    Categorical([*categorical_schema['coffee_grinder'].values()], name='coffee_grinder'),
    Categorical([*categorical_schema['stove_type'].values()], name='stove_type'),
    # Integer(80, 100, name='heat_setting'),
    Categorical([categorical_schema['coffee_category']['JUKIA_PARK-UGANDA;SL-14;NATURAL_ANAEROBIC;MEDIUM']], name='coffee_category'),
    # Only the amount of days that we've available for now
    Integer(6, 30, name='coffee_roasted_days_ago'),
    Real(13, 22, name='coffee_dose'),
    # It's possible to grind up to 8, but this doesn't make any sence.
    Real(0.6, 1.2, name='grind_size'),
    Categorical([*categorical_schema['water_filtered'].values()], name='water_filtered'),
    # Real(110, 130, name='water_weight')
]

# Run Bayesian Optimization
# TODO: what amount of n_calls do we need?
result = gp_minimize(objective, search_space, n_calls=100, random_state=42)

print("Best coffee brew settings found:")
for param_name, param_value in zip(search_space, result.x):
    if param_name.name in inverted_categorical_schema:
        param_value = inverted_categorical_schema[param_name.name][param_value]
    print(f"{param_name.name}: {param_value}")

print("Optimized Objective Value:", result.fun)







