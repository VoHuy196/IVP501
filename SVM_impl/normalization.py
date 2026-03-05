import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ========== LOAD ==========
print("Loading features: ...")
X = np.load("X.npy")
y = np.load("y.npy")

print("Original shape:", X.shape)
print("Data type:", X.dtype)

# ========== SCALING ==========
print("Fitting StandardScaler...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.astype(dtype=np.float32)
print("Scaled shape:", X_scaled.shape)

# ========== SAVE ==========
np.save("X_scaled.npy", X_scaled)
np.save("y.npy", y)

joblib.dump(scaler, "scaler.pkl")

print("Saved:")
print(" - X_scaled.npy")
print(" - y.npy")
print(" - scaler.pkl")
