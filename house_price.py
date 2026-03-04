import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Create dataset
data = {
    "Size": [1000,1500,2000,2500,3000,1200,1800,2200,2700,3200,
             1400,1600,2100,2600,3100,1300,1700,2300,2800,3300],
    "Bedrooms": [2,3,3,4,5,2,3,4,4,5,
                 2,3,3,4,5,2,3,4,4,5],
    "Price": [50,75,90,120,150,60,85,110,130,160,
              70,80,95,125,155,65,88,115,135,165]
}

df = pd.DataFrame(data)

# Separate features and target
X = df[["Size", "Bedrooms"]]
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- Linear Regression -----
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_score = r2_score(y_test, lr_pred)

# ----- Ridge Regression -----
from sklearn.model_selection import GridSearchCV

# Define alpha values
param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}

ridge = Ridge()

grid = GridSearchCV(ridge, param_grid, cv=5, scoring="r2")
grid.fit(X_train_scaled, y_train)

print("Best Alpha:", grid.best_params_)
print("Best Cross-Validation R²:", grid.best_score_)

# Evaluate best model on test data
best_model = grid.best_estimator_
test_score = best_model.score(X_test_scaled, y_test)

print("Test R² with Best Alpha:", test_score)

# Save the trained model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save the scaler too (VERY IMPORTANT)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")

# Predict new house
new_house = pd.DataFrame([[1800, 3]], columns=["Size", "Bedrooms"])
new_house_scaled = scaler.transform(new_house)

prediction = best_model.predict(new_house_scaled)
print("Predicted Price (Ridge):", prediction[0])

# Visualization
plt.scatter(df["Size"], df["Price"])
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("House Size vs Price")
plt.savefig("price_plot.png")
plt.show()