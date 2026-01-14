import pandas as pd


def evaluate_model(model, X_test, y_test, feature_names, season_name):
    print(f"\nEvaluation for {season_name} Model")

    importance = model.get_feature_importance()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

    r2 = model.score(X_test, y_test)
    print(f"\n{season_name} R² Score: {r2:.4f}")

    return importance_df
