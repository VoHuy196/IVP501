import numpy as np
import os
from sklearn.model_selection import train_test_split

# ========== PATHS ==========
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ========== LOAD ==========
print("Loading scaled dataset...")
X         = np.load(os.path.join(DATA_DIR, "X_scaled.npy"))
y_plant   = np.load(os.path.join(DATA_DIR, "y_plant.npy"))
y_disease = np.load(os.path.join(DATA_DIR, "y_disease.npy"))

print("Full dataset shape :", X.shape)
print("y_plant  classes   :", len(set(y_plant)))
print("y_disease classes  :", len(set(y_disease)))

# ========== SPLIT — PLANT ==========
print("\nSplitting plant labels  (70% train / 30% test) ...")
X_train_plant, X_test_plant, y_train_plant, y_test_plant = train_test_split(
    X, y_plant,
    test_size=0.3,
    stratify=y_plant,
    random_state=42
)
print("  Train shape:", X_train_plant.shape)
print("  Test  shape:", X_test_plant.shape)

# ========== SPLIT — DISEASE ==========
print("\nSplitting disease labels (70% train / 30% test) ...")
X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(
    X, y_disease,
    test_size=0.3,
    stratify=y_disease,
    random_state=42
)
print("  Train shape:", X_train_disease.shape)
print("  Test  shape:", X_test_disease.shape)

# ========== SAVE ==========
np.save(os.path.join(DATA_DIR, "X_train_plant.npy"),   X_train_plant)
np.save(os.path.join(DATA_DIR, "X_test_plant.npy"),    X_test_plant)
np.save(os.path.join(DATA_DIR, "y_train_plant.npy"),   y_train_plant)
np.save(os.path.join(DATA_DIR, "y_test_plant.npy"),    y_test_plant)

np.save(os.path.join(DATA_DIR, "X_train_disease.npy"), X_train_disease)
np.save(os.path.join(DATA_DIR, "X_test_disease.npy"),  X_test_disease)
np.save(os.path.join(DATA_DIR, "y_train_disease.npy"), y_train_disease)
np.save(os.path.join(DATA_DIR, "y_test_disease.npy"),  y_test_disease)

print("\nSaved:")
print("  data/X_train_plant.npy  /  data/X_test_plant.npy")
print("  data/y_train_plant.npy  /  data/y_test_plant.npy")
print("  data/X_train_disease.npy / data/X_test_disease.npy")
print("  data/y_train_disease.npy / data/y_test_disease.npy")
