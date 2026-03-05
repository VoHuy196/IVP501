import numpy as np
from sklearn.model_selection import train_test_split

# ========== LOAD ==========
print("Loading scaled dataset...")
X = np.load("X_scaled.npy")
y = np.load("y.npy")

print("Full dataset shape:", X.shape)

# ========== SPLIT ==========
print("Splitting 70% train / 30% test ...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ========== SAVE ==========
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Saved:")
print(" - X_train.npy")
print(" - X_test.npy")
print(" - y_train.npy")
print(" - y_test.npy")