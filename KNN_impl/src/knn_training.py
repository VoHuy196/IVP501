import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
import datetime
import json
import pandas as pd

# ========== PATHS ==========
# Points to the KNN_impl directory
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Reuse dataset and label map from RandomForest_impl for convenience
RF_DIR = os.path.join(BASE_DIR, "..", "RandomForest_impl")
DATA_DIR = os.path.join(RF_DIR, "data")

# ========== LOAD LABEL MAP ==========
with open(os.path.join(BASE_DIR, "label_map.json"), "r", encoding="utf-8") as f:
    label_map = json.load(f)

plant_names = {v: k for k, v in label_map["plant"].items()}
disease_names = {v: k for k, v in label_map["disease"].items()}

# ========== HELPER FUNCTION ==========
def run_knn(task_name, X_train, y_train, X_test, y_test, target_names, k_values, log_file):
    best_model = None
    best_f1 = -1
    best_k = None

    for k in k_values:
        print(f"\n{'='*55}")
        print(f"[{task_name}] Training KNN with n_neighbors = {k} ...")

        # Initialize KNN model
        model = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
        model.fit(X_train, y_train)
        print("Training finished. Predicting...")

        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Accuracy : {acc:.4f} | F1 Score : {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_k = k

        # --- Write log entry ---
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = pd.DataFrame([{
            "Time": timestamp, "Task": task_name, "n_neighbors": k,
            "Accuracy": acc, "F1 Score": f1,
            "Train Size": X_train.shape[0], "Features": X_train.shape[1]
        }])
        
        if not os.path.isfile(log_file):
            log_entry.to_csv(log_file, index=False)
        else:
            log_entry.to_csv(log_file, mode="a", header=False, index=False)

        # --- Write txt report ---
        reports_dir = os.path.join(LOGS_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        time_str = datetime.datetime.now().strftime("%H%M%S")
        report_name = os.path.join(reports_dir, f"report_knn_{task_name}_k{k}_{time_str}.txt")
        
        with open(report_name, "w", encoding="utf-8") as f:
            f.write(f"EXPERIMENT REPORT - {timestamp}\nTask: {task_name}\n")
            f.write(f"Model: KNeighborsClassifier (n_neighbors={k}, weights='distance')\n")
            f.write("=" * 40 + "\nCLASSIFICATION REPORT:\n")
            f.write(report)
            f.write("\nCONFUSION MATRIX:\n")
            f.write(np.array2string(cm))
            
    return best_f1, best_model, best_k

# ========== MAIN SCRIPT ==========
print("Generating dummy train/test data...")

# Generate dummy data for testing
np.random.seed(42)
n_samples = 1000
n_features = 1864
n_plant_classes = len(plant_names)
n_disease_classes = len(disease_names)

# Create random features
X = np.random.randn(n_samples, n_features).astype(np.float32)
y_plant = np.random.randint(0, n_plant_classes, n_samples)
y_disease = np.random.randint(0, n_disease_classes, n_samples)

# Split into train/test
from sklearn.model_selection import train_test_split
X_train_plant, X_test_plant, y_train_plant, y_test_plant = train_test_split(
    X, y_plant, test_size=0.3, random_state=42, stratify=y_plant
)
X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(
    X, y_disease, test_size=0.3, random_state=42, stratify=y_disease
)

plant_labels = [plant_names[i] for i in range(len(plant_names))]
disease_labels = [disease_names[i] for i in range(len(disease_names))]

# Set up K values (number of neighbors) to experiment with
k_values = [3, 5, 7, 9, 11]
log_file = os.path.join(LOGS_DIR, "training_history_knn.csv")

# 1. Train plant classification model
best_plant_f1, best_plant_model, best_plant_k = run_knn(
    "plant", X_train_plant, y_train_plant, X_test_plant, y_test_plant, plant_labels, k_values, log_file
)

# 2. Train disease classification model
best_disease_f1, best_disease_model, best_disease_k = run_knn(
    "disease", X_train_disease, y_train_disease, X_test_disease, y_test_disease, disease_labels, k_values, log_file
)

# Save the best models
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(best_plant_model, os.path.join(MODELS_DIR, f"knn_best_plant_k{best_plant_k}.pkl"))
joblib.dump(best_disease_model, os.path.join(MODELS_DIR, f"knn_best_disease_k{best_disease_k}.pkl"))

print(f"\n{'='*55}")
print(f"Done! Best models have been saved to the models/ directory.")