import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# LOAD DATA
df = pd.read_csv("../datasets/patient_data.csv")

print("Dataset shape:", df.shape)

# ==============================
# FIX ALL COLUMNS EXACTLY
# ==============================

# Convert target
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES':1, 'NO':0})

# Convert GENDER
if 'GENDER' in df.columns:
    df['GENDER'] = df['GENDER'].map({'M':1, 'F':0})

# Convert remaining YES/NO columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].map({'YES':1, 'NO':0, 'Yes':1, 'No':0})

# Fill missing values
df = df.fillna(0)

# ==============================
# SPLIT
# ==============================
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

print("Features shape:", X.shape)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("../data_model/patient_model.pkl")

# ==============================
# PREDICT
# ==============================
y_pred = model.predict(X)

# ==============================
# ACCURACY
# ==============================
accuracy = accuracy_score(y, y_pred)

print("\n✅ Accuracy:", round(accuracy * 100, 2), "%")
