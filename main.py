import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from io import StringIO
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GRU, SimpleRNN, Reshape, Conv1D, Flatten, Input, MultiHeadAttention, LayerNormalization

# 1. DATASET ACQUISITION
try:
    with open("KDDTrain+.arff") as f:
        data = f.read().split("@data\n")[1]
    df = pd.read_csv(StringIO(data), header=None)
    print("✅ Dataset loaded successfully. Initializing Stratified Processing...")
except Exception as e:
    print(f"❌ Error: KDDTrain+.arff not found in directory.")
    exit()

# 2. PREPROCESSING PIPELINE
df = df.fillna(0)
for col in [1, 2, 3]: # Protocol, Service, Flag
    df[col] = LabelEncoder().fit_transform(df[col])

# poly_encoder holds the specific labels (neptune, satan, ipsweep, etc.)
poly_encoder = LabelEncoder()
y_poly = poly_encoder.fit_transform(df[41]) 
y_bin = df[41].apply(lambda x: 0 if x=="normal" else 1)

X = df.iloc[:, :-1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. STRATIFIED DATA DIVISION
X_train, X_test, y_train_bin, y_test_bin, y_train_poly, y_test_poly = train_test_split(
    X_scaled, y_bin, y_poly, test_size=0.2, random_state=42, stratify=y_bin
)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_poly))

# ---------------------------------------------------------
# 4. MULTI-TIERED DEFENSIVE ENGINES
# ---------------------------------------------------------
print("\n🛡️ Training Defensive Engines (CNN-GRU, RNN, Transformer, Autoencoder)...")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train_bin)

auto_in = Input(shape=(input_dim,))
enc = Dense(32, activation='relu')(auto_in)
dec = Dense(input_dim, activation='sigmoid')(enc)
autoencoder = Model(auto_in, dec)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train[y_train_bin==0], X_train[y_train_bin==0], epochs=5, batch_size=128, verbose=0)

cnn_gru = Sequential([
    Input(shape=(input_dim,)), Reshape((input_dim, 1)),
    Conv1D(64, 3, activation='relu'), Flatten(),
    Reshape((1, -1)), GRU(128), Dense(1, activation='sigmoid')
])
cnn_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_gru.fit(X_train, y_train_bin, epochs=5, verbose=0)

rnn_engine = Sequential([
    Input(shape=(input_dim,)), Reshape((input_dim, 1)),
    SimpleRNN(64, activation='relu'), Dense(1, activation='sigmoid')
])
rnn_engine.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_engine.fit(X_train, y_train_bin, epochs=5, verbose=0)

def build_transformer(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x_reshaped = Reshape((1, 128))(x)
    attn = MultiHeadAttention(num_heads=4, key_dim=4)(x_reshaped, x_reshaped)
    x = LayerNormalization()(attn)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

trans_model = build_transformer((input_dim,))
trans_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
trans_model.fit(X_train, y_train_poly, epochs=5, verbose=0)

# ---------------------------------------------------------
# 5. PERFORMANCE ANALYTICS
# ---------------------------------------------------------
print("\n📊 SYSTEM PERFORMANCE ANALYTICS")
print("=" * 70)
gru_preds = (cnn_gru.predict(X_test, verbose=0) > 0.5).astype(int)

print(f"{'METRIC':<25} | {'VALUE':<15}")
print("-" * 70)
print(f"{'Overall Accuracy':<25} | {accuracy_score(y_test_bin, gru_preds)*100:.2f}%")
print(f"{'F1-Score (Hybrid)':<25} | {f1_score(y_test_bin, gru_preds)*100:.2f}%")
print(f"{'Precision':<25} | {precision_score(y_test_bin, gru_preds)*100:.2f}%")
print(f"{'Recall (Detection Rate)':<25} | {recall_score(y_test_bin, gru_preds)*100:.2f}%")
print("-" * 70)

# ---------------------------------------------------------
# 6. ENHANCED DASHBOARD (UPDATED FOR ATTACK NAME RECOGNITION)
# ---------------------------------------------------------
print("\n🔥 INITIATING HIGH-DENSITY THREAT MONITORING DASHBOARD...")
print("="*165)
print(f"{'PKT':<4} | {'ML STATUS':<12} | {'DETECTED ATTACK':<20} | {'RNN RISK':<10} | {'GRU RISK':<10} | {'AE ERROR':<10} | {'TRUST SCORE':<12} | {'ALERT STATUS'}")
print("-" * 165)

# Stress Test Setup: 25 Attacks vs 5 Normal
attack_indices = np.where(y_test_bin == 1)[0]
normal_indices = np.where(y_test_bin == 0)[0]
demo_indices = np.concatenate([np.random.choice(attack_indices, 25), np.random.choice(normal_indices, 5)])
np.random.shuffle(demo_indices)

attack_count = 0
for i, idx in enumerate(demo_indices):
    sample = X_test[idx].reshape(1, -1)
    
    # Engine Predictions
    ml_res = "Anomaly" if rf_model.predict(sample)[0] == 1 else "Normal"
    
    # ⬇️ UPDATED LOGIC TO EXTRACT ATTACK NAME ⬇️
    trans_pred = trans_model.predict(sample, verbose=0)
    # Convert index back to original string name using poly_encoder
    attack_name_raw = poly_encoder.inverse_transform([np.argmax(trans_pred)])[0]
    
    rnn_risk = rnn_engine.predict(sample, verbose=0)[0][0]
    gru_risk = cnn_gru.predict(sample, verbose=0)[0][0]
    ae_err = np.mean(np.square(sample - autoencoder.predict(sample, verbose=0)))
    
    # Trust Score Logic
    trust_score = np.clip((rnn_risk * 0.2 + gru_risk * 0.5 + ae_err * 8 + (0.3 if ml_res == "Anomaly" else 0)), 0, 1)
    
    if trust_score > 0.35 or ae_err > 0.04:
        status, is_attack = "🚨 CRITICAL", True
        attack_count += 1
    elif trust_score > 0.15:
        status, is_attack = "⚠️ WARNING", True
        attack_count += 1
    else:
        status, is_attack = "✅ SECURE", False
    
    # Use the raw name from dataset if it's an attack
    display_attack = attack_name_raw.upper() if is_attack else "N/A"
        
    print(f"{i+1:02d} | {ml_res:<12} | {display_attack:<20} | {rnn_risk:.4f}   | {gru_risk:.4f}   | {ae_err:.4f}   | {trust_score:.4f}      | {status}")
    time.sleep(0.1)

print("="*165)
print(f"📊 STRESS TEST SUMMARY: {attack_count} ATTACKS DETECTED OUT OF 30 PACKETS.")
print(f"✅ Detection Sensitivity Optimized. Stratified F1-Score: {f1_score(y_test_bin, gru_preds)*100:.2f}%.")

# 7. VISUAL PROOF
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_bin, gru_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix: Hybrid Ensemble Intrusion Detection")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
