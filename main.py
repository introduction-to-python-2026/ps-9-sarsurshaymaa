# -*- coding: utf-8 -*-

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
# Train model (NO scaling!)
# ===============================
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# ===============================
# Local check
# ===============================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ===============================
# Save model
# ===============================
joblib.dump(model, "my_model.joblib")
