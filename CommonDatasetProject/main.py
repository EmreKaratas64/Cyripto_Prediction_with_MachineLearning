#191805019 - Emircan Karagöz
#191805029 - Onur Doğan
#191805063 - Emre Karataş


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer



# Veri setini yükle
dataset = pd.read_csv('indoor_data_HAACS.csv')

print("\nInformation of the Data Set:")
print(dataset.info())

print("First Five Row:")
print(dataset.head())

print("Basic Statistics of Numerical Variables:")
print(dataset.describe())

print("\nIncomplete Data Check:")
print(dataset.isnull().sum())
print("\n")


# Plotting histograms for X and Y coordinates
sns.histplot(data=dataset, x='X (Numeric)', kde=True)
plt.title('Distribution of X (Numeric)')
plt.show()

sns.histplot(data=dataset, x='Y (Numeric)', kde=True)
plt.title('Distribution of Y (Numeric)')
plt.show()

# Checking the class distribution for the Floor target variable
sns.countplot(data=dataset, x='Floor (Categoric)')
plt.title('Class Distribution for Floor')
plt.show()

# Creating the correlation matrix of X and Y variables
correlation_matrix = dataset[['X (Numeric)', 'Y (Numeric)']].corr()

# Visualizing the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix: X and Y')
plt.show()

# Data Preprocessing
# Filing Incomplete Data
imputer = SimpleImputer(strategy='mean')  
dataset[['X (Numeric)', 'Y (Numeric)']] = imputer.fit_transform(dataset[['X (Numeric)', 'Y (Numeric)']])

# Scaling
numerical_features = ['X (Numeric)', 'Y (Numeric)']
scaler = StandardScaler()
dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

# Separating input features and output columns
X = dataset.drop(columns=['X (Numeric)', 'Y (Numeric)', 'Floor (Categoric)'])
Y_X = dataset['X (Numeric)']
Y_Y = dataset['Y (Numeric)']
Y_Floor = dataset['Floor (Categoric)']

# Create training and test data sets
X_train, X_test, Y_X_train, Y_X_test, Y_Y_train, Y_Y_test, Y_Floor_train, Y_Floor_test = train_test_split(
    X, Y_X, Y_Y, Y_Floor, test_size=0.2, random_state=42)

# Declare Regression Algorithm
regressors = [
    LinearRegression(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    GradientBoostingRegressor(),
]
# Defining variables to follow the best performing regression algorithms
best_r2_x_regression = 0
best_r2_y_regression = 0
best_regression_algorithm_x = None
best_regression_algorithm_y = None

# Define data frames to create tables
training_results = pd.DataFrame(columns=['Algorithm', 'R2 Score (X Regression)', 'R2 Score (Y Regression)'])
test_results = pd.DataFrame({'Actual Value (X)': Y_X_test, 'Actual Value (Y)': Y_Y_test})

# Train and test regression algorithms for X
for regressor in regressors:
    regressor.fit(X_train, Y_X_train)
    Y_X_pred_train = regressor.predict(X_train)
    Y_X_pred_test = regressor.predict(X_test)
    r2_x_regression_train = r2_score(Y_X_train, Y_X_pred_train)
    r2_x_regression_test = r2_score(Y_X_test, Y_X_pred_test)

    training_results = pd.concat([training_results, pd.DataFrame({
        'Algorithm': [regressor.__class__.__name__],
        'R2 Score (X Regression)': [r2_x_regression_train],
        'R2 Score (Y Regression)': [r2_x_regression_test]
    })], ignore_index=True)

    # Update the best performing regression algorithm according to the performance on the training set
    if r2_x_regression_train > best_r2_x_regression:
        best_r2_x_regression = r2_x_regression_train
        best_regression_algorithm_x = regressor
    test_results[regressor.__class__.__name__ + ' (X)'] = Y_X_pred_test

# Train and test regression algorithms for Y
for regressor in regressors:
    regressor.fit(X_train, Y_Y_train)
    Y_Y_pred_train = regressor.predict(X_train)
    Y_Y_pred_test = regressor.predict(X_test)
    r2_y_regression_train = r2_score(Y_Y_train, Y_Y_pred_train)
    r2_y_regression_test = r2_score(Y_Y_test, Y_Y_pred_test)

    test_results[regressor.__class__.__name__ + ' (Y)'] = Y_Y_pred_test

    # Update the best performing regression algorithm
    if r2_y_regression_test > best_r2_y_regression:
        best_r2_y_regression = r2_y_regression_test
        best_regression_algorithm_y = regressor

# Results
print("Training Results:")
print(training_results)

print("\nTest Results:")
print(test_results)

# Calculate scores using the best performing regression algorithms
Y_X_pred = best_regression_algorithm_x.predict(X_test)
Y_Y_pred = best_regression_algorithm_y.predict(X_test)

# Define the classification model
classifier_Floor = XGBClassifier()

# Train and test the classification model
classifier_Floor.fit(X_train, Y_Floor_train)
Y_Floor_pred = classifier_Floor.predict(X_test)

# Calculate performance
accuracy_floor_classification = classifier_Floor.score(X_test, Y_Floor_test)
project_performance_score = best_r2_x_regression * best_r2_y_regression * accuracy_floor_classification

# Evaluate
print("\nBest Regression Algorithm (X Regression):", best_regression_algorithm_x.__class__.__name__)
print("Best R2 Score (X Regression):", best_r2_x_regression)

print("\nBest Regression Algorithm (Y Regression):", best_regression_algorithm_y.__class__.__name__)
print("Best R2 Score (Y Regression):", best_r2_y_regression)

print("\nAccuracy (Floor Classification):", accuracy_floor_classification)
print("Project Performance Score:", project_performance_score)

