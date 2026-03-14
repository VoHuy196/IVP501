import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

# ========== PATHS ==========
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ========== LOAD ==========
print("Loading scaled dataset...")
X         = np.load(os.path.join(DATA_DIR, "X_scaled.npy"), allow_pickle=True)
y_plant   = np.load(os.path.join(DATA_DIR, "y_plant.npy"), allow_pickle=True)
y_disease = np.load(os.path.join(DATA_DIR, "y_disease.npy"), allow_pickle=True)

with open(os.path.join(BASE_DIR, "label_map.json"), "r", encoding="utf-8") as f:
    label_map = json.load(f)

plant_to_idx     = label_map["plant"]
disease_by_plant = label_map["disease_by_plant"]
idx_to_plant     = {v: k for k, v in plant_to_idx.items()}

print("Full dataset shape :", X.shape)
print("y_plant  classes   :", len(set(y_plant)))

# ========== SPLIT — PLANT (global) ==========
print("\nSplitting plant labels  (70% train / 30% test) ...")
X_train_plant, X_test_plant, y_train_plant, y_test_plant = train_test_split(
    X, y_plant,
    test_size=0.3,
    stratify=y_plant,
    random_state=42
)
np.save(os.path.join(DATA_DIR, "X_train_plant.npy"),  X_train_plant)
np.save(os.path.join(DATA_DIR, "X_test_plant.npy"),   X_test_plant)
np.save(os.path.join(DATA_DIR, "y_train_plant.npy"),  y_train_plant)
np.save(os.path.join(DATA_DIR, "y_test_plant.npy"),   y_test_plant)
print(f"  Train shape: {X_train_plant.shape}  |  Test shape: {X_test_plant.shape}")

# ========== SPLIT — DISEASE per plant ==========
print("\nSplitting disease labels per plant (70% train / 30% test) ...")

disease_data_dir = os.path.join(DATA_DIR, "disease_per_plant")
os.makedirs(disease_data_dir, exist_ok=True)

for p_idx, plant_name in sorted(idx_to_plant.items()):
    mask   = (y_plant == p_idx)
    X_p    = X[mask]
    y_d_p  = y_disease[mask]

    n_classes = len(set(y_d_p))
    print(f"\n  [{plant_name}]  samples={X_p.shape[0]}, disease_classes={n_classes}")

    if X_p.shape[0] < 10 or n_classes < 2:
        print(f"    [SKIP] Not enough data or only 1 disease class.")
        continue

    # Need at least 2 samples per class for stratified split
    class_counts = {c: int(np.sum(y_d_p == c)) for c in set(y_d_p)}
    can_stratify = all(v >= 2 for v in class_counts.values())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_p, y_d_p,
        test_size=0.3,
        stratify=y_d_p if can_stratify else None,
        random_state=42
    )

    safe_name = plant_name.replace(",", "").replace(" ", "_").replace("(", "").replace(")", "")
    np.save(os.path.join(disease_data_dir, f"X_train_{safe_name}.npy"), X_tr)
    np.save(os.path.join(disease_data_dir, f"X_test_{safe_name}.npy"),  X_te)
    np.save(os.path.join(disease_data_dir, f"y_train_{safe_name}.npy"), y_tr)
    np.save(os.path.join(disease_data_dir, f"y_test_{safe_name}.npy"),  y_te)
    print(f"    Saved: train={X_tr.shape[0]}, test={X_te.shape[0]}")

print("\nAll splits saved to:")
print("  data/X_train_plant.npy / data/X_test_plant.npy")
print("  data/y_train_plant.npy / data/y_test_plant.npy")
print("  data/disease_per_plant/X_train_<Plant>.npy  (per-plant disease splits)")
