# House Price Prediction using Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset

df = pd.read_csv("train.csv")

print(df.head())
print("\nShape:", df.shape)

# Select Important Features

features = [
    'OverallQual',
    'GrLivArea',
    'TotalBsmtSF',
    'GarageCars',
    'GarageArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd',
    'LotArea'
]


X = df[features]
y = df['SalePrice']

# Handle Missing Values

X = X.fillna(X.median())

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Train Model

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# Prediction

y_pred = model.predict(X_test)

# Evaluation

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n----------------------")
print("MODEL PERFORMANCE")
print("----------------------")

print("MAE :", round(mae,2))            # Mean Absolute Error
print("MSE :", round(mse,2))            # Mean Squared Error
print("RMSE :", round(rmse,2))          # Root Mean Squared Error
print("R2 Score :", round(r2,4))        # R-Squared

# Actual vs Predicted Graph

plt.figure(figsize=(8,6))

plt.scatter(y_test,y_pred)

plt.xlabel("Actual House Price")

plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted Prices")


plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()

# Residual Plot

residual = y_test-y_pred


plt.figure(figsize=(8,6))

sns.histplot(
    residual,
    bins=30,
    kde=True
)

plt.title("Residual Distribution")

plt.xlabel("Error")

plt.show()

# Predict New House

new_house = pd.DataFrame({

    'OverallQual':[8],
    'GrLivArea':[2000],
    'TotalBsmtSF':[1000],
    'GarageCars':[2],
    'GarageArea':[500],
    'YearBuilt':[2010],
    '1stFlrSF':[1200],
    '2ndFlrSF':[800],
    'FullBath':[2],
    'BedroomAbvGr':[3],
    'TotRmsAbvGrd':[7],
    'LotArea':[8000]

})

new_house = scaler.transform(new_house)

price = model.predict(new_house)

print("\nPredicted House Price:")

print(f"${price[0]:,.2f}")
