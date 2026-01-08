import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def get_data_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "Energy_data.csv")


def load_and_preprocess(target_column):
    df = pd.read_csv(get_data_path())

    if 'Orientation' in df.columns:
        le = LabelEncoder()
        df['Orientation'] = le.fit_transform(df['Orientation'].astype(str))

    features = [
        'Total Area (sqft)',
        'Number of Occupants',
        'Number of Floors',
        'Orientation'
    ]

    X = df[features].fillna(df[features].mean())
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
