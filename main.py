# -*- coding: utf-8 -*-
"""
Parkinson's Disease Classification
Based on parkinsons.csv dataset
"""

# ===============================
# 1. Imports
# ===============================
import pandas as pd
import joblib
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ===============================
# 2. Load the dataset
# ===============================
df = pd.read_csv("parkinsons.csv")
df = df.dropna()

print("Dataset head:")
print(df.head())

# ===============================
# 3. Select features
# ===============================
# Input features
X = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]

# Output label
y = df['status']

# Visualization (optional – comment out if running headless)
fig = px.scatter(
    df,
    x='MDVP:Fo(Hz)',
    y='MDVP:Jitter(%)',
    color='status',
    title='Parkinsons Dataset Feature Scatter'
)
fig.show()

# ===============================
# 4. Scale the data
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 5. Split the data
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# 6. Choose and train model
# ===============================
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ===============================
# 7. Evaluate accuracy
# ===============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.3f}")

# ===============================
# 8. Save the model
# ===============================
model_path = "my_model.joblib"
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")

