import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "Energy_data.csv")


def load_and_preprocess(target_column):
    df = pd.read_csv(get_data_path())

    features = [
        'Total Area (sqft)',
        'Number of Occupants',
        'Number of Floors',
        'Orientation'
    ]
    
    X = df[features]
    y = df[target_column]
    
    X = pd.get_dummies(X, columns=['Orientation'], drop_first=False)

    feature_names = X.columns.tolist()
    
    X = X.fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    ), feature_names
