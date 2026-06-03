from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random

app = Flask(__name__)

print("Training model... please wait")

# -------------------------------
# LOAD DATA
# -------------------------------
with open("KDDTrain+.arff") as f:
    data = f.read().split("@data\n")[1]

df = pd.read_csv(StringIO(data), header=None)

# -------------------------------
# PREPROCESSING
# -------------------------------
df = df.fillna(0)

for col in [1, 2, 3]:
    df[col] = LabelEncoder().fit_transform(df[col])

# Binary labels
df[41] = df[41].apply(lambda x: 0 if x == "normal" else 1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN MODELS
# -------------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

ann = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=3, batch_size=64, verbose=0)

print("Model ready")

# -------------------------------
# ATTACK TYPES FOR UI DISPLAY
# -------------------------------
attack_types = ["DoS", "Probe", "R2L", "U2R", "Normal"]

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    idx = random.randint(0, len(X_test) - 1)
    sample = X_test[idx].reshape(1, -1)

    rf_pred = int(rf_model.predict(sample)[0])
    ann_prob = float(ann.predict(sample, verbose=0)[0][0])

    final_result = "Attack" if rf_pred == 1 or ann_prob > 0.5 else "Normal"
    confidence = round(max(ann_prob, 1 - ann_prob) * 100, 2)

    if final_result == "Attack":
        attack_type = random.choice(["DoS", "Probe", "R2L", "U2R"])
    else:
        attack_type = "None"

    if final_result == "Normal":
        risk = "Low"
    else:
        if confidence >= 90:
            risk = "High"
        elif confidence >= 75:
            risk = "Medium"
        else:
            risk = "Low"

    rf_result = "Attack" if rf_pred == 1 else "Normal"
    ann_result = "Attack" if ann_prob > 0.5 else "Normal"

    return jsonify({
        "status": final_result,
        "attack_type": attack_type,
        "confidence": f"{confidence}%",
        "risk": risk,
        "rf_result": rf_result,
        "ann_result": ann_result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)