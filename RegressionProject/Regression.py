#191805019 - Emircan Karagöz
#191805029 - Onur Doğan
#191805063 - Emre Karataş


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("BTCUSDT.csv")

print("\nInformation of the Data Set:")
print(data.info())

print("First Five Row:")
print(data.head())

print("Basic Statistics of Numerical Variables:")
print(data.describe())

print("\nIncomplete Data Check:")
print(data.isnull().sum())
print("\n")

# Histogram
next_close_values = data['next close']

plt.hist(next_close_values, bins=10)
plt.xlabel('Next Close Values')
plt.ylabel('Frequency')
plt.title('Next Histogram by Next Close Values')
plt.show()



data['time'] = pd.to_datetime(data['time'])

plt.figure(figsize=(12,6))
plt.plot(data['time'], data['close'])
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Closing Price Changes Over Time')
plt.show()




# Date of the selected day
selected_day = "2023-04-12"

# Convert data to date format
data['time'] = pd.to_datetime(data['time'])

# Filter and copy the selected day's data
selected_data = data[data['time'].dt.date == pd.to_datetime(selected_day).date()].copy()

# Convert time tags to time format
selected_data['time'] = selected_data['time'].dt.strftime('%H:%M')

# Chart creation
plt.figure(figsize=(12, 6))
plt.plot(selected_data['time'], selected_data['open'], label='Opening Price')
plt.plot(selected_data['time'], selected_data['close'], label='Closing Price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f"Comparison of Opening and Closing Prices - {selected_day}")
plt.legend()
plt.xticks(rotation=45)

plt.show()


plt.figure(figsize=(12, 6))

# RSI_14 
plt.subplot(1, 3, 1)
plt.hist(data['RSI_14'], bins=20, color='teal')
plt.title('RSI_14 Distribution')
plt.xlabel('RSI_14')
plt.ylabel('Count')

# MACD_14_20_9 
plt.subplot(1, 3, 2)
plt.hist(data['MACD_14_20_9'], bins=20, color='orange')
plt.title('MACD_14_20_9 Distribution')
plt.xlabel('MACD_14_20_9')
plt.ylabel('Count')

# ADX_14 
plt.subplot(1, 3, 3)
plt.hist(data['ADX_14'], bins=20, color='purple')
plt.title('ADX_14 Distribution')
plt.xlabel('ADX_14')
plt.ylabel('Count')

plt.tight_layout()
plt.show()



features = ['time', 'open', 'close', 'volume', 'RSI_14', 'MACD_14_20_9', 'ADX_14']
target = 'next close'

X = data[features]
y = data[target]

# Remove the time column
X = X.drop('time', axis=1)

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test setlerini oluştur
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Regresyon algoritmalarını tanımla
regressors = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor()
]

# Create dictionaries to store R2 and MAE values
r2_scores = {}
mae_scores = {}

# Apply regression algorithms
for regressor in regressors:
    model_name = regressor.__class__.__name__
    model = regressor.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2_scores[model_name] = r2_score(y_test, y_pred)
    mae_scores[model_name] = mean_absolute_error(y_test, y_pred)
    
    print(f"{model_name}:")
    print("R2 skoru:", r2_scores[model_name])
    print("MAE skoru:", mae_scores[model_name])
    print("\n")

# Create a comparison table of training results
train_results = pd.DataFrame({'Algorithm': list(r2_scores.keys()), 'R2 Score': list(r2_scores.values()), 'MAE Score': list(mae_scores.values())})
print("Training Results:")
print(train_results)
print("\n")

# Create dictionaries to store R2 and MAE values for Test and Train
test_compare = {}

# Apply regression algorithms
for regressor in regressors:
    model_name = regressor.__class__.__name__
    model = regressor.fit(X_train, y_train)
    
    # Predict on test data
    y_pred_test = model.predict(X_test)
    
    # Create a DataFrame to compare actual test values and predicted test values
    compare_df = pd.DataFrame({
        'Actual Test Value': y_test,
        'Predicted Test Value': y_pred_test
    })
    
    # Store the DataFrame for the current model in the dictionary
    test_compare[model_name] = compare_df

# Print the test compare tables for each model
for model_name, compare_df in test_compare.items():
    print(f"Test Compare Table - {model_name}:")
    print(compare_df)
    print("\n")

# Find the best model
best_model_index = max(range(len(regressors)), key=lambda i: r2_scores[regressors[i].__class__.__name__])
best_model = regressors[best_model_index]

best_model_name = best_model.__class__.__name__
print("Best Model:", best_model_name)
print("R2 score:", r2_scores[best_model_name])
print("MAE score:", mae_scores[best_model_name])


# Hyperparameter tuning for the best training algorithm
if best_model_name != 'LinearRegression':
    params = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.01, 0.001]
    }

    grid_search = GridSearchCV(best_model, params, cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nBest Model (after hyperparameter tuning):", best_model.__class__.__name__)
    print("R2 score (after hyperparameter tuning):", r2_score(y_test, y_pred))
    print("MAE score (after hyperparameter tuning):", mean_absolute_error(y_test, y_pred))
    print("\n")
else:
    print("\nUsing a model that does not require hyperparameter tuning: LinearRegression")
    print("Best model:", best_model_name)
    print("R2 score:", r2_scores[best_model_name])
    print("MAE score:", mae_scores[best_model_name])


# Plotting Residual scatter plot 
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

plt.scatter(y_test, residuals)
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Scatter Plot')
plt.show()