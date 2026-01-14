import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_and_preprocess


(data, feature_names) = load_and_preprocess("KSEB bill in monsoon")
X_train, X_test, y_train, y_test = data


model = joblib.load("saved_models/catboost_monsoon.pkl")


y_pred = model.predict(X_test)
residuals = y_test - y_pred


plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Analysis - Monsoon Model")
plt.show()
