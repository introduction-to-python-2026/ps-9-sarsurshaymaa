# -*- coding: utf-8 -*-

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ===============================
# Load dataset
# ===============================
df = pd.read_csv("parkinsons.csv")
df = df.dropna()

# ===============================
# Select features
# ===============================
features = ["MDVP:Fo(Hz)", "MDVP:Jitter(%)"]
X = df[features]
y = df["status"]

# ===============================
# Split data
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Pipeline: Scaling + SVM
# ===============================
model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
])

# ===============================
# Train
# ===============================
model.fit(X_train, y_train)

# ===============================
# Evaluate (local check)
# ===============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Validation accuracy:", accuracy)

# ===============================
# Save model
# ===============================
joblib.dump(model, "my_model.joblib")
