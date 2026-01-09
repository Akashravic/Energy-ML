from catboost import CatBoostRegressor
from utils import load_and_preprocess
from sklearn.metrics import r2_score
import joblib
import os

from sklearn.model_selection import KFold, cross_val_score
import numpy as np

X_train, X_test, y_train, y_test = load_and_preprocess(
    target_column="KSEB bill in summer"
)

model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kf,
    scoring='r2'
)

print("Summer CV R2 scores:", cv_scores)
print("Mean CV R2:", np.mean(cv_scores))
print("Std Dev:", np.std(cv_scores))


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"Summer R² Score: {r2:.4f}")


os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/catboost_summer.pkl")
