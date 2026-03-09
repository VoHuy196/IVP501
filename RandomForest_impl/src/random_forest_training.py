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

plant_to_idx     = label_map["plant"]
disease_by_plant = label_map["disease_by_plant"]   # {plant_name: {disease_name: local_idx}}
idx_to_plant     = {v: k for k, v in plant_to_idx.items()}

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
            report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
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

X_train_plant = np.load(os.path.join(DATA_DIR, "X_train_plant.npy"))
X_test_plant  = np.load(os.path.join(DATA_DIR, "X_test_plant.npy"))
y_train_plant = np.load(os.path.join(DATA_DIR, "y_train_plant.npy"))
y_test_plant  = np.load(os.path.join(DATA_DIR, "y_test_plant.npy"))

print("Plant train shape:", X_train_plant.shape)

# ========== HYPERPARAMETER GRID ==========
n_estimators_values = [50, 100, 200]
depth_values        = [10, 20, None]
log_file            = os.path.join(LOGS_DIR, "training_history_rf.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

# ========== TASK 1: Plant classification ==========
plant_labels = [idx_to_plant[i] for i in range(len(idx_to_plant))]

best_plant_f1, best_plant_model, best_plant_params = run_random_forest(
    "plant", X_train_plant, y_train_plant,
    X_test_plant, y_test_plant,
    plant_labels, n_estimators_values, depth_values, log_file
)

p_label = f"n{best_plant_params['n_estimators']}_depth{best_plant_params['max_depth']}"
joblib.dump(best_plant_model, os.path.join(MODELS_DIR, f"rf_best_plant_{p_label}.pkl"))
print(f"\nBest Plant F1 : {best_plant_f1:.4f}  -> models/rf_best_plant_{p_label}.pkl")

# ========== TASK 2: Per-plant disease classification ==========
print(f"\n{'='*55}")
print("Training per-plant disease classifiers ...")

disease_data_dir    = os.path.join(DATA_DIR, "disease_per_plant")
disease_models_dir  = os.path.join(MODELS_DIR, "disease_per_plant")
os.makedirs(disease_models_dir, exist_ok=True)

best_disease_summary = {}

for p_idx, plant_name in sorted(idx_to_plant.items()):
    safe_name = plant_name.replace(",", "").replace(" ", "_").replace("(", "").replace(")", "")

    X_tr_path = os.path.join(disease_data_dir, f"X_train_{safe_name}.npy")
    X_te_path = os.path.join(disease_data_dir, f"X_test_{safe_name}.npy")
    y_tr_path = os.path.join(disease_data_dir, f"y_train_{safe_name}.npy")
    y_te_path = os.path.join(disease_data_dir, f"y_test_{safe_name}.npy")

    if not os.path.isfile(X_tr_path):
        print(f"\n[SKIP] No split data found for plant: {plant_name}")
        continue

    X_tr = np.load(X_tr_path)
    X_te = np.load(X_te_path)
    y_tr = np.load(y_tr_path)
    y_te = np.load(y_te_path)

    # Build target names for this plant's diseases (local indices → names)
    d_map        = disease_by_plant[plant_name]            # {disease_name: local_idx}
    idx_to_d     = {v: k for k, v in d_map.items()}
    disease_lbls = [idx_to_d[i] for i in range(len(idx_to_d))]

    print(f"\n{'='*55}")
    print(f"Plant: {plant_name}  |  diseases: {disease_lbls}")

    task_name = f"disease_{safe_name}"
    best_f1, best_model, best_params = run_random_forest(
        task_name, X_tr, y_tr, X_te, y_te,
        disease_lbls, n_estimators_values, depth_values, log_file
    )

    d_label = f"n{best_params['n_estimators']}_depth{best_params['max_depth']}"
    model_path = os.path.join(disease_models_dir, f"rf_disease_{safe_name}_{d_label}.pkl")
    joblib.dump(best_model, model_path)

    best_disease_summary[plant_name] = {"f1": best_f1, "model_file": model_path, "params": best_params}
    print(f"  Best F1 : {best_f1:.4f}  -> {model_path}")

# ========== SUMMARY ==========
print(f"\n{'='*55}")
print(f"Best Plant   F1 : {best_plant_f1:.4f}  -> models/rf_best_plant_{p_label}.pkl")
print("\nBest Disease F1 per plant:")
for plant_name, info in best_disease_summary.items():
    print(f"  {plant_name:<35s}  F1={info['f1']:.4f}  -> {info['model_file']}")
print(f"\nLog saved: {log_file}")
