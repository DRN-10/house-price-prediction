import pandas as pd
import pickle

# Load model
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# New house data
new_house = pd.DataFrame([[2000, 3]], columns=["Size", "Bedrooms"])

# Scale
new_house_scaled = scaler.transform(new_house)

# Predict
prediction = model.predict(new_house_scaled)

print("Predicted Price:", prediction[0])