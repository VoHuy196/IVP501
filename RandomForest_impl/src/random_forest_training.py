import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib
import os
import datetime
import json
import pandas as pd

# ========== PATHS ==========
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

# ========== LOAD LABEL MAP ==========
with open(os.path.join(BASE_DIR, "label_map.json"), "r", encoding="utf-8") as f:
    label_map = json.load(f)

plant_names   = {v: k for k, v in label_map["plant"].items()}
disease_names = {v: k for k, v in label_map["disease"].items()}

# ========== HELPER ==========
def run_random_forest(task_name, X_train, y_train, X_test, y_test,
                      target_names, n_estimators_values, depth_values, log_file):

    best_model  = None
    best_f1     = -1
    best_params = {}

    for n_est in n_estimators_values:
        for max_depth in depth_values:
            depth_label = str(max_depth) if max_depth is not None else "full"
            param_label = f"n{n_est}_depth{depth_label}"

            print(f"\n{'='*55}")
            print(f"[{task_name}]  RandomForest  n_estimators={n_est}, max_depth={depth_label} ...")

            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_depth,
                criterion="gini",
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            model.fit(X_train, y_train)
            print("Training completed.")

            y_pred = model.predict(X_test)

            f1     = f1_score(y_test, y_pred, average="weighted")
            acc    = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=target_names)
            cm     = confusion_matrix(y_test, y_pred)

            print(f"Accuracy : {acc:.4f}")
            print(f"F1 Score : {f1:.4f}")
            print("\nClassification Report:")
            print(report)
            print("Confusion Matrix:")
            print(cm)

            if f1 > best_f1:
                best_f1     = f1
                best_model  = model
                best_params = {"n_estimators": n_est, "max_depth": depth_label}

            # --- Log ---
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = pd.DataFrame([{
                "Time"         : timestamp,
                "Task"         : task_name,
                "n_estimators" : n_est,
                "max_depth"    : depth_label,
                "criterion"    : model.criterion,
                "Accuracy"     : acc,
                "F1 Score"     : f1,
                "Train Size"   : X_train.shape[0],
                "Features"     : X_train.shape[1]
            }])
            if not os.path.isfile(log_file):
                log_entry.to_csv(log_file, index=False)
            else:
                log_entry.to_csv(log_file, mode="a", header=False, index=False)

            # --- Report ---
            reports_dir = os.path.join(LOGS_DIR, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            time_str    = datetime.datetime.now().strftime("%H%M%S")
            report_name = os.path.join(reports_dir,
                            f"report_rf_{task_name}_{param_label}_{time_str}.txt")
            with open(report_name, "w", encoding="utf-8") as f:
                f.write(f"EXPERIMENT REPORT - {timestamp}\n")
                f.write(f"Task  : {task_name}\n")
                f.write(f"Model : RandomForestClassifier "
                        f"(n_estimators={n_est}, max_depth={depth_label}, "
                        f"criterion={model.criterion})\n")
                f.write("=" * 40 + "\n")
                f.write("CLASSIFICATION REPORT:\n")
                f.write(report)
                f.write("\nCONFUSION MATRIX:\n")
                f.write(np.array2string(cm))
                f.write("\n\nFEATURE IMPORTANCES (Top 20):\n")
                importances = model.feature_importances_
                top20_idx   = np.argsort(importances)[::-1][:20]
                for rank, idx in enumerate(top20_idx, 1):
                    f.write(f"  {rank:2d}. f{idx:<6d}  importance={importances[idx]:.6f}\n")
            print(f"[OK] Report saved -> '{report_name}'")

    return best_f1, best_model, best_params

# ========== LOAD DATA ==========
print("Loading train/test datasets...")

X_train_plant    = np.load(os.path.join(DATA_DIR, "X_train_plant.npy"))
X_test_plant     = np.load(os.path.join(DATA_DIR, "X_test_plant.npy"))
y_train_plant    = np.load(os.path.join(DATA_DIR, "y_train_plant.npy"))
y_test_plant     = np.load(os.path.join(DATA_DIR, "y_test_plant.npy"))

X_train_disease  = np.load(os.path.join(DATA_DIR, "X_train_disease.npy"))
X_test_disease   = np.load(os.path.join(DATA_DIR, "X_test_disease.npy"))
y_train_disease  = np.load(os.path.join(DATA_DIR, "y_train_disease.npy"))
y_test_disease   = np.load(os.path.join(DATA_DIR, "y_test_disease.npy"))

print("Plant   train shape:", X_train_plant.shape)
print("Disease train shape:", X_train_disease.shape)

# ========== TARGET NAMES ==========
plant_labels   = [plant_names[i]   for i in range(len(plant_names))]
disease_labels = [disease_names[i] for i in range(len(disease_names))]

# ========== HYPERPARAMETER GRID ==========
n_estimators_values = [50, 100, 200]
depth_values        = [10, 20, None]
log_file            = os.path.join(LOGS_DIR, "training_history_rf.csv")

# ========== TASK 1: Plant classification ==========
best_plant_f1, best_plant_model, best_plant_params = run_random_forest(
    "plant", X_train_plant, y_train_plant,
    X_test_plant, y_test_plant,
    plant_labels, n_estimators_values, depth_values, log_file
)

# ========== TASK 2: Disease classification ==========
best_disease_f1, best_disease_model, best_disease_params = run_random_forest(
    "disease", X_train_disease, y_train_disease,
    X_test_disease, y_test_disease,
    disease_labels, n_estimators_values, depth_values, log_file
)

# ========== SAVE BEST MODELS ==========
p_label = f"n{best_plant_params['n_estimators']}_depth{best_plant_params['max_depth']}"
d_label = f"n{best_disease_params['n_estimators']}_depth{best_disease_params['max_depth']}"

joblib.dump(best_plant_model,   os.path.join(MODELS_DIR, f"rf_best_plant_{p_label}.pkl"))
joblib.dump(best_disease_model, os.path.join(MODELS_DIR, f"rf_best_disease_{d_label}.pkl"))

print(f"\n{'='*55}")
print(f"Best Plant   F1 : {best_plant_f1:.4f}  -> models/rf_best_plant_{p_label}.pkl")
print(f"Best Disease F1 : {best_disease_f1:.4f}  -> models/rf_best_disease_{d_label}.pkl")
print(f"Log saved       : {log_file}")
