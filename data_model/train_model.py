import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("../datasets/patient_data.csv")

# ==============================
# CLEAN DATA
# ==============================

# Convert target
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES':1, 'NO':0})

# Convert gender
if 'GENDER' in df.columns:
    df['GENDER'] = df['GENDER'].map({'M':1, 'F':0})

# Convert YES/NO
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].map({'YES':1, 'NO':0, 'Yes':1, 'No':0})

# Fill missing
df = df.fillna(0)

# ==============================
# SPLIT DATA
# ==============================
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TRAIN MODEL
# ==============================
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ==============================
# EVALUATE
# ==============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ New Model Accuracy:", round(accuracy * 100, 2), "%")

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, "../data_model/patient_model.pkl")

print("✅ Model saved successfully!")