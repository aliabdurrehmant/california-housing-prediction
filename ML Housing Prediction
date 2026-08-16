import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=35)
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(prediction[:10])
print("\n")
print(y_test[:10])
mse = mean_squared_error(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
print("MSE:", mse)
print("MAE:", mae)
print("R²:", r2)
model = RandomForestRegressor(random_state=50)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\n")
print(predictions[:10])
print("\n")
print(y_test[:10])
print(predictions[:10])
print("\n")
print(y_test[:10])
print("MSE:", mean_squared_error(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))
print("R²:", r2_score(y_test, predictions))
import matplotlib.pyplot as plt
models = ["Linear (baseline)", "Linear + features", "Random Forest"]
r2_scores = [0.601, 0.658, 0.809]
plt.bar(models, r2_scores)
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Model Comparison: R² Score")
plt.ylim([0, 1])
plt.show()

