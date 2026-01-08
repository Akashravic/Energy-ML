from catboost import CatBoostRegressor
from utils import load_and_preprocess
from sklearn.metrics import r2_score
import joblib
import os

X_train, X_test, y_train, y_test = load_and_preprocess(
    target_column="KSEB bill in monsoon"
)

model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"Monsoon R² Score: {r2:.4f}")

# Save model
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/catboost_monsoon.pkl")
