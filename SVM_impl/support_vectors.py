import numpy as np
import os
import joblib
import datetime
import pandas as pd
import gc
import sys
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# ========== 1. HYPERPARAMETERS CONFIGURATION ==========
C_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

data_dir = r"D:\project_ML\IVP501\SVM_impl\input_vectors"
report_base_dir = r"D:\project_ML\IVP501\SVM_impl\reports"
os.makedirs(report_base_dir, exist_ok=True)

global_log_file = os.path.join(report_base_dir, "global_training_history.csv")

# ========== 2. LOAD DATASET ==========
print("Loading datasets ...")
try:
    X_train_all = np.load(os.path.join(data_dir, "X_train_pca.npy"), mmap_mode='r')
    X_test_all = np.load(os.path.join(data_dir, "X_test_pca.npy"), mmap_mode='r')
    y_plant_train = np.load(os.path.join(data_dir, "y_plant_train.npy"))
    y_plant_test = np.load(os.path.join(data_dir, "y_plant_test.npy"))
    y_disease_train = np.load(os.path.join(data_dir, "y_disease_train.npy"))
    y_disease_test = np.load(os.path.join(data_dir, "y_disease_test.npy"))
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

unique_plants = np.unique(y_plant_train)
all_tasks = ["ROOT_PLANT_SPECIES"] + list(unique_plants)

# ========== 3. TRAINING LOOP ==========

for task in all_tasks:

    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}")
    task_dir = os.path.join(report_base_dir, task.replace(" ", "_").lower())
    os.makedirs(task_dir, exist_ok=True)

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
            print(f"-> Skipping {task}: Insufficient classes.")
            continue
    print(f"Samples: {len(X_tr)} | Classes: {len(np.unique(y_tr))}")

    # ========== 4. HYPERPARAMETER SEARCH ==========
    try:
        base_model = LinearSVC(
            max_iter=2000,
            dual=False,
            tol=1e-3,
            class_weight='balanced',
            random_state=42,
        )
        
        param_grid = {"C": C_GRID}
        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        print("Running cross-validation to find best C...")

        grid.fit(X_tr, y_tr)
        model = grid.best_estimator_
        best_c = grid.best_params_["C"]

        print(f"Best C found: {best_c}")

        # ========== 5. EVALUATION ==========
        y_pred = model.predict(X_te)
        train_acc = model.score(X_tr, y_tr)
        test_acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average='weighted')

        print(f"Done! Train: {train_acc:.4f} | Test: {test_acc:.4f} | F1: {f1:.4f}")

        # Save Results
        model_save_path = os.path.join(task_dir, f"model_{task.lower()}_C{best_c}.pkl")
        joblib.dump(model, model_save_path)

        # ========== 6. SAVE REPORT ==========
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_filename = f"Report_{task}_C{best_c}.txt"
        with open(os.path.join(task_dir, report_filename), "w", encoding="utf-8") as f:
            f.write(f"TASK: {task} | C: {best_c}\n")
            f.write(classification_report(y_te, y_pred))
        
        # ========== 7. LOG METRICS ==========
        log_entry = pd.DataFrame([{
            "Time": timestamp,
            "Task": task,
            "Best C": best_c,
            "Train_Acc": train_acc,
            "Test_Acc": test_acc,
            "F1 score": f1
        }])
        log_entry.to_csv(global_log_file, mode='a', header=not os.path.exists(global_log_file), index=False)

        del model, y_pred
        gc.collect()

    except Exception as e:
        print(f"[FAILED] Task {task}, C={best_c}: {e}")
        with open("crash_log.txt", "a") as f:
            f.write(f"{timestamp} - Task: {task}, C: {best_c} - Error: {str(e)}\n")
        continue

print("\n[FINISH] All hierarchical models trained.")
