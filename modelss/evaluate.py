import joblib
import pandas as pd
import os

FEATURE_NAMES = [
    'Total Area (sqft)',
    'Number of Occupants',
    'Number of Floors',
    'Orientation'
]

def evaluate_model(model_path, season_name):
    print(f"\n Evaluation for {season_name} Model")

    model = joblib.load(model_path)

    importance = model.get_feature_importance()
    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    print(df)


if __name__ == "__main__":
    evaluate_model(
        model_path="saved_models/catboost_monsoon.pkl",
        season_name="Monsoon"
    )

    evaluate_model(
        model_path="saved_models/catboost_summer.pkl",
        season_name="Summer"
    )
