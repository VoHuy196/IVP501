import numpy as np
import joblib
import datetime
import pandas as pd
import gc
import sys
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, ParameterGrid

# ========== 1. HYPERPARAMETERS CONFIGURATION ==========
N_ESTIMATORS_GRID = [50, 100, 200, 300]
MAX_DEPTH_GRID = [10, 20, 30, None]

BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "input_vectors"
report_base_dir = BASE_DIR / "reports"
report_base_dir.mkdir(parents=True, exist_ok=True)

global_log_file = report_base_dir / "global_training_history.csv"
crash_log_file = report_base_dir / "crash_log.txt"
label_map_file = BASE_DIR / "label_map.json"

# ========== 2. LOAD DATASET ==========
print("Loading datasets ...")
try:
    X_train_all = np.load(data_dir / "X_train_pca.npy", mmap_mode='r')
    X_test_all = np.load(data_dir / "X_test_pca.npy", mmap_mode='r')
    y_plant_train = np.load(data_dir / "y_plant_train.npy")
    y_plant_test = np.load(data_dir / "y_plant_test.npy")
    y_disease_train = np.load(data_dir / "y_disease_train.npy")
    y_disease_test = np.load(data_dir / "y_disease_test.npy")
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

# ========== 2.1 LOAD LABEL MAP ==========
plant_idx_to_name = {}
plant_idx_to_disease_names = {}

if label_map_file.exists():
    with open(label_map_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    plant_name_to_idx = label_map.get("plant", {})
    disease_by_plant = label_map.get("disease_by_plant", {})

    plant_idx_to_name = {int(idx): name for name, idx in plant_name_to_idx.items()}
    for plant_name, disease_map in disease_by_plant.items():
        plant_idx = plant_name_to_idx.get(plant_name)
        if plant_idx is None:
            continue
        plant_idx_to_disease_names[int(plant_idx)] = {
            int(d_idx): d_name for d_name, d_idx in disease_map.items()
        }


def plant_name_from_idx(plant_idx):
    return plant_idx_to_name.get(int(plant_idx), f"plant_{int(plant_idx)}")


def disease_name_from_idx(plant_idx, disease_idx):
    return plant_idx_to_disease_names.get(int(plant_idx), {}).get(
        int(disease_idx), f"disease_{int(disease_idx)}"
    )


def depth_to_label(max_depth):
    return str(max_depth) if max_depth is not None else "full"


def create_rf_model(n_estimators, max_depth):
    return RandomForestClassifier(
        criterion="gini",
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=0,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )


def build_report_assets(task, y_true, y_pred):
    report_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if task == "ROOT_PLANT_SPECIES":
        report_names = [plant_name_from_idx(idx) for idx in report_labels]
    else:
        report_names = [disease_name_from_idx(task, idx) for idx in report_labels]

    class_report_text = classification_report(
        y_true,
        y_pred,
        labels=report_labels,
        target_names=report_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=report_labels)
    cm_table = pd.DataFrame(
        cm,
        index=[f"true:{name}" for name in report_names],
        columns=[f"pred:{name}" for name in report_names],
    )
    return class_report_text, cm_table


def write_report(report_path, task_label, n_estimators, depth_label, train_acc, test_acc, f1, class_report_text, cm_table, importances, is_best):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            f"TASK: {task_label} | n_estimators: {n_estimators} | max_depth: {depth_label} | is_best: {is_best}\n"
        )
        f.write(f"Train_Acc: {train_acc:.4f} | Test_Acc: {test_acc:.4f} | F1: {f1:.4f}\n\n")
        f.write(class_report_text)
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write(cm_table.to_string())
        f.write("\n\nFEATURE IMPORTANCES (Top 20):\n")
        top20_idx = np.argsort(importances)[::-1][:20]
        for rank, idx in enumerate(top20_idx, 1):
            f.write(f"  {rank:2d}. f{idx:<6d}  importance={importances[idx]:.6f}\n")

unique_plants = np.unique(y_plant_train)
all_tasks = ["ROOT_PLANT_SPECIES"] + list(unique_plants)

# ========== 3. TRAINING LOOP ==========

for task in all_tasks:
    task_label = "ROOT_PLANT_SPECIES" if task == "ROOT_PLANT_SPECIES" else plant_name_from_idx(task)

    print(f"\n{'='*60}")
    print(f"TASK: {task_label}")
    print(f"{'='*60}")
    task_dir = report_base_dir / task_label.replace(" ", "_").lower()
    task_dir.mkdir(parents=True, exist_ok=True)

    # ----- Select dataset -----
    if task == "ROOT_PLANT_SPECIES":
        X_tr, y_tr = X_train_all, y_plant_train
        X_te, y_te = X_test_all, y_plant_test
    else:
        train_mask = (y_plant_train == task)
        test_mask = (y_plant_test == task)

        X_tr = X_train_all[train_mask]
        y_tr = y_disease_train[train_mask]

        X_te = X_test_all[test_mask]
        y_te = y_disease_test[test_mask]

        if len(np.unique(y_tr)) < 2:
            print(f"-> Skipping {task_label}: Insufficient classes.")
            continue
    print(f"Samples: {len(X_tr)} | Classes: {len(np.unique(y_tr))}")

    # ========== 4. HYPERPARAMETER SEARCH ==========
    try:
        base_model = create_rf_model(n_estimators=100, max_depth=None)

        param_grid = {
            "n_estimators": N_ESTIMATORS_GRID,
            "max_depth": MAX_DEPTH_GRID
        }
        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        print("Running cross-validation to find best hyperparameters...")

        grid.fit(X_tr, y_tr)
        model = grid.best_estimator_
        best_n_estimators = grid.best_params_["n_estimators"]
        best_max_depth = grid.best_params_["max_depth"]

        print(f"Best n_estimators: {best_n_estimators}, Best max_depth: {best_max_depth}")

        # ========== 5. EVALUATION ==========
        y_pred = model.predict(X_te)
        train_acc = model.score(X_tr, y_tr)
        test_acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average='weighted')
        class_report_text, cm_table = build_report_assets(task, y_te, y_pred)

        print(f"Done! Train: {train_acc:.4f} | Test: {test_acc:.4f} | F1: {f1:.4f}")

        # Save Model
        depth_label = depth_to_label(best_max_depth)
        model_save_path = task_dir / f"model_{task_label.lower()}_n{best_n_estimators}_depth{depth_label}.pkl"
        joblib.dump(model, model_save_path)

        # ========== 6. SAVE REPORT ==========
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        best_report_filename = f"Report_{task_label}_BEST_n{best_n_estimators}_depth{depth_label}.txt"
        write_report(
            task_dir / best_report_filename,
            task_label,
            best_n_estimators,
            depth_label,
            train_acc,
            test_acc,
            f1,
            class_report_text,
            cm_table,
            model.feature_importances_,
            is_best=True,
        )

        # Save one report for each hyperparameter pair that was tried.
        for params in ParameterGrid(param_grid):
            n_estimators = params["n_estimators"]
            max_depth = params["max_depth"]
            depth_label_trial = depth_to_label(max_depth)

            is_best_combo = (
                n_estimators == best_n_estimators and max_depth == best_max_depth
            )

            if is_best_combo:
                trial_model = model
            else:
                trial_model = create_rf_model(n_estimators=n_estimators, max_depth=max_depth)
                trial_model.fit(X_tr, y_tr)

            y_pred_trial = trial_model.predict(X_te)
            train_acc_trial = trial_model.score(X_tr, y_tr)
            test_acc_trial = accuracy_score(y_te, y_pred_trial)
            f1_trial = f1_score(y_te, y_pred_trial, average='weighted')
            class_report_trial, cm_table_trial = build_report_assets(task, y_te, y_pred_trial)

            trial_report_filename = (
                f"Report_{task_label}_TRY_n{n_estimators}_depth{depth_label_trial}.txt"
            )
            write_report(
                task_dir / trial_report_filename,
                task_label,
                n_estimators,
                depth_label_trial,
                train_acc_trial,
                test_acc_trial,
                f1_trial,
                class_report_trial,
                cm_table_trial,
                trial_model.feature_importances_,
                is_best=is_best_combo,
            )

            if not is_best_combo:
                del trial_model

            del y_pred_trial

        # ========== 7. LOG METRICS ==========
        log_entry = pd.DataFrame([{
            "Time": timestamp,
            "Task": task_label,
            "TaskId": task if task == "ROOT_PLANT_SPECIES" else int(task),
            "Best n_estimators": best_n_estimators,
            "Best max_depth": depth_label,
            "Train_Acc": train_acc,
            "Test_Acc": test_acc,
            "F1 score": f1
        }])
        log_entry.to_csv(global_log_file, mode='a', header=not global_log_file.exists(), index=False)

        del model, y_pred
        gc.collect()

    except Exception as e:
        print(f"[FAILED] Task {task_label}: {e}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(crash_log_file, "a") as f:
            f.write(f"{timestamp} - Task: {task_label} - Error: {str(e)}\n")
        continue

print("\n[FINISH] All hierarchical models trained.")

