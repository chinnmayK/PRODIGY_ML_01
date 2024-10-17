import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train = pd.read_csv('train.csv')

# Select features: Square footage, number of bedrooms, number of bathrooms
# Square footage = GrLivArea
# Total number of bathrooms = FullBath + HalfBath
train['TotalBath'] = train['FullBath'] + (train['HalfBath'] * 0.5)
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath']

# Select target: SalePrice
X = train[features]
y = train['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on the training set
y_train_pred = model.predict(X_train)

# Calculate performance metrics on training set
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print(f'Training Mean Squared Error: {mse_train:.2f}')
print(f'Training Mean Absolute Error: {mae_train:.2f}')
print(f'Training R-squared: {r2_train:.2f}')

# Predictions on the test set
y_test_pred = model.predict(X_test)

# Calculate performance metrics on test set
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Test Mean Squared Error: {mse_test:.2f}')
print(f'Test Mean Absolute Error: {mae_test:.2f}')
print(f'Test R-squared: {r2_test:.2f}')

# Plot: Actual vs Predicted prices
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
