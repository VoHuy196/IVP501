import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ========== PATHS ==========
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ========== LOAD ==========
print("Loading features...")
X         = np.load(os.path.join(DATA_DIR, "X.npy"), allow_pickle=True)
y_plant   = np.load(os.path.join(DATA_DIR, "y_plant.npy"), allow_pickle=True)
y_disease = np.load(os.path.join(DATA_DIR, "y_disease.npy"), allow_pickle=True)

print("Original shape:", X.shape)
print("Data type     :", X.dtype)

# ========== SCALING ==========
print("Fitting StandardScaler...")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.astype(dtype=np.float32)
print("Scaled shape  :", X_scaled.shape)

# ========== SAVE ==========
np.save(os.path.join(DATA_DIR, "X_scaled.npy"),  X_scaled)
np.save(os.path.join(DATA_DIR, "y_plant.npy"),   y_plant)
np.save(os.path.join(DATA_DIR, "y_disease.npy"), y_disease)

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

print("Saved:")
print("  data/X_scaled.npy")
print("  data/y_plant.npy")
print("  data/y_disease.npy")
print("  models/scaler.pkl")
