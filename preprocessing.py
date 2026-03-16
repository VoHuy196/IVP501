import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
import os
from pathlib import Path

# ==========================================
# ĐÃ SỬA: Tự động trỏ vào thư mục input_vectors của code hiện tại
# ==========================================
BASE_DIR = Path(__file__).parent
feature_dir = BASE_DIR / "input_vectors"

if not os.path.exists(feature_dir):
    os.makedirs(feature_dir)
    print(f"Created directory: {feature_dir}")

# ========== LOAD ==========
print("Loading features: ...\n")
# Chú ý: Đảm bảo bạn đã chạy file feature_extraction của SVM trước để có 3 file này nhé
X = np.load(os.path.join(feature_dir, "X_features.npy"), allow_pickle=True)
y_plant = np.load(os.path.join(feature_dir, "Y_plant.npy"), allow_pickle=True)
y_disease = np.load(os.path.join(feature_dir, "Y_disease.npy"), allow_pickle=True)

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
print(f"Saving split datasets to: {feature_dir}\n")

# Train
np.save(os.path.join(feature_dir, "X_train_pca.npy"), X_train_pca)
np.save(os.path.join(feature_dir, "y_plant_train.npy"), y_plant_train)
np.save(os.path.join(feature_dir, "y_disease_train.npy"), y_disease_train)

# Test
np.save(os.path.join(feature_dir, "X_test_pca.npy"), X_test_pca)
np.save(os.path.join(feature_dir, "y_plant_test.npy"), y_plant_test)
np.save(os.path.join(feature_dir, "y_disease_test.npy"), y_disease_test)

# Reproducible
joblib.dump(scaler, os.path.join(feature_dir, "scaler.pkl"))
joblib.dump(pca, os.path.join(feature_dir, "pca.pkl"))

print("[DONE] Saved")