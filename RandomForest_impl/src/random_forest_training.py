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
INPUT_DIR  = os.path.join(BASE_DIR, "input_vectors")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
GLOBAL_LOG_FILE = os.path.join(LOGS_DIR, "training_history_all.csv")
# ========== LOAD LABEL MAP ==========
with open(os.path.join(BASE_DIR, "label_map.json"), "r", encoding="utf-8") as f:
    label_map = json.load(f)

plant_to_idx     = label_map["plant"]
disease_by_plant = label_map["disease_by_plant"]   # {plant_name: {disease_name: local_idx}}
idx_to_plant     = {v: k for k, v in plant_to_idx.items()}

# ========== HELPER ==========
def run_random_forest(task_name, X_train, y_train, X_test, y_test,
                      target_names, n_estimators_values, depth_values, log_file, labels=None):

    best_model  = None
    best_f1     = -1
    best_params = {}
    task_entries = []   # collect all entries for this task

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
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            model.fit(X_train, y_train)
            print("Training completed.")

            y_pred = model.predict(X_test)

            f1     = f1_score(y_test, y_pred, average="weighted")
            acc    = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test,
                y_pred,
                labels=labels,
                target_names=target_names,
                zero_division=0
            )
            cm     = confusion_matrix(y_test, y_pred, labels=labels)

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

            # --- Log (per-task CSV) ---
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = {
                "Time"         : timestamp,
                "Task"         : task_name,
                "n_estimators" : n_est,
                "max_depth"    : depth_label,
                "criterion"    : model.criterion,
                "Accuracy"     : acc,
                "F1 Score"     : f1,
                "Train Size"   : X_train.shape[0],
                "Features"     : X_train.shape[1]
            }
            log_entry = pd.DataFrame([entry])
            if not os.path.isfile(log_file):
                log_entry.to_csv(log_file, index=False)
            else:
                log_entry.to_csv(log_file, mode="a", header=False, index=False)

            task_entries.append(entry)

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

    # --- Mark best & append to global log ---
    global_rows = []
    for e in task_entries:
        row = dict(e)
        row["is_best"] = (
            e["n_estimators"] == best_params.get("n_estimators") and
            e["max_depth"]    == best_params.get("max_depth")
        )
        global_rows.append(row)

    global_df = pd.DataFrame(global_rows)
    if not os.path.isfile(GLOBAL_LOG_FILE):
        global_df.to_csv(GLOBAL_LOG_FILE, index=False)
    else:
        global_df.to_csv(GLOBAL_LOG_FILE, mode="a", header=False, index=False)

    return best_f1, best_model, best_params


# ========== LOAD DATA ==========
print("Loading train/test datasets...")

X_train_plant = np.load(os.path.join(INPUT_DIR, "X_train_pca.npy"))
X_test_plant  = np.load(os.path.join(INPUT_DIR, "X_test_pca.npy"))
y_train_plant = np.load(os.path.join(INPUT_DIR, "y_plant_train.npy"))
y_test_plant  = np.load(os.path.join(INPUT_DIR, "y_plant_test.npy"))
y_train_disease = np.load(os.path.join(INPUT_DIR, "y_disease_train.npy"))
y_test_disease  = np.load(os.path.join(INPUT_DIR, "y_disease_test.npy"))

print("Plant train shape:", X_train_plant.shape)

# ========== HYPERPARAMETER GRID ==========
n_estimators_values = [50, 100, 200, 300]
depth_values        = [10, 20, 30, None]
log_file            = os.path.join(LOGS_DIR, "training_history_rf.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

# ========== TASK 1: Plant classification ==========
plant_label_indices = sorted(idx_to_plant.keys())
plant_labels = [idx_to_plant[i] for i in plant_label_indices]

best_plant_f1, best_plant_model, best_plant_params = run_random_forest(
    "plant", X_train_plant, y_train_plant,
    X_test_plant, y_test_plant,
    plant_labels, n_estimators_values, depth_values, log_file,
    labels=plant_label_indices
)

p_label = f"n{best_plant_params['n_estimators']}_depth{best_plant_params['max_depth']}"
joblib.dump(best_plant_model, os.path.join(MODELS_DIR, f"rf_best_plant_{p_label}.pkl"))
print(f"\nBest Plant F1 : {best_plant_f1:.4f}  -> models/rf_best_plant_{p_label}.pkl")

# ========== TASK 2: Per-plant disease classification ==========
print(f"\n{'='*55}")
print("Training per-plant disease classifiers ...")

disease_models_dir  = os.path.join(MODELS_DIR, "disease_per_plant")
os.makedirs(disease_models_dir, exist_ok=True)

best_disease_summary = {}

for p_idx, plant_name in sorted(idx_to_plant.items()):
    safe_name = plant_name.replace(",", "").replace(" ", "_").replace("(", "").replace(")", "")

    train_mask = (y_train_plant == p_idx)
    test_mask = (y_test_plant == p_idx)
    if not np.any(train_mask) or not np.any(test_mask):
        print(f"\n[SKIP] No input_vectors samples for plant: {plant_name}")
        continue

    X_tr = X_train_plant[train_mask]
    X_te = X_test_plant[test_mask]
    y_tr = y_train_disease[train_mask]
    y_te = y_test_disease[test_mask]

    # Build target names for this plant's diseases (local indices → names)
    d_map        = disease_by_plant[plant_name]            # {disease_name: local_idx}
    idx_to_d     = {v: k for k, v in d_map.items()}
    disease_label_indices = sorted(idx_to_d.keys())
    disease_lbls = [idx_to_d[i] for i in disease_label_indices]

    print(f"\n{'='*55}")
    print(f"Plant: {plant_name}  |  diseases: {disease_lbls}")

    task_name = f"disease_{safe_name}"
    best_f1, best_model, best_params = run_random_forest(
        task_name, X_tr, y_tr, X_te, y_te,
        disease_lbls, n_estimators_values, depth_values, log_file,
        labels=disease_label_indices
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
