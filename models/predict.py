import joblib
import pandas as pd
import numpy as np
import os

from utils import get_data_path 
from sklearn.preprocessing import StandardScaler


model = joblib.load("saved_models/catboost_monsoon.pkl")

input_data = pd.DataFrame([{
    "Total Area (sqft)": 1200,
    "Number of Occupants": 4,
    "Number of Floors": 2,
    "Orientation": "E"  
}])

input_data = pd.get_dummies(
    input_data,
    columns=["Orientation"]
)

expected_features = model.feature_names_
for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_features]

prediction = model.predict(input_data)

print("Predicted electricity bill:", prediction[0])
