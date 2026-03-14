import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "input_vectors"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input directory : {RAW_DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# ========== LOAD ==========
print("Loading features: ...\n")

required_files = {
    "X": RAW_DATA_DIR / "X.npy",
    "y_plant": RAW_DATA_DIR / "y_plant.npy",
    "y_disease": RAW_DATA_DIR / "y_disease.npy",
}

missing_files = [str(path) for path in required_files.values() if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "Missing extracted files from feature_extraction.py:\n"
        + "\n".join(missing_files)
    )

X = np.load(required_files["X"])
y_plant = np.load(required_files["y_plant"])
y_disease = np.load(required_files["y_disease"])

# ========== SPLIT ==========
print("Splitting 70% train / 30% test ...\n")

X_train, X_test, y_plant_train, y_plant_test, y_disease_train, y_disease_test = train_test_split(
    X, 
    y_plant, 
    y_disease,
    test_size=0.3,
    stratify=y_plant,
    random_state=42
)

print("Splitting completed \n")

# ========== SCALING ==========
print("Fitting StandardScaler ...\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = X_train_scaled.astype(dtype=np.float32)
X_test_scaled = X_test_scaled.astype(dtype=np.float32)

print("Scaling completed \n")

# ========== DIMENSIONALITY REDUCTION ==========
print("Applying PCA ...\n")

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Reducing completed \n")

# ========== SAVE ==========
print(f"Saving split datasets to: {OUTPUT_DIR}\n")

# Train
np.save(OUTPUT_DIR / "X_train_pca.npy", X_train_pca)
np.save(OUTPUT_DIR / "y_plant_train.npy", y_plant_train)
np.save(OUTPUT_DIR / "y_disease_train.npy", y_disease_train)

# Test
np.save(OUTPUT_DIR / "X_test_pca.npy", X_test_pca)
np.save(OUTPUT_DIR / "y_plant_test.npy", y_plant_test)
np.save(OUTPUT_DIR / "y_disease_test.npy", y_disease_test)

# Reproducible
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
joblib.dump(pca, OUTPUT_DIR / "pca.pkl")

print("[DONE] Saved")
