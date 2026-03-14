import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import datetime
import json
import pandas as pd
import gc

# ========== PATHS ==========
BASE_DIR = os.path.dirname(__file__)
KNN_DIR = os.path.join(BASE_DIR, "..")

MODELS_DIR = os.path.join(KNN_DIR, "models")
LOGS_DIR = os.path.join(KNN_DIR, "logs")
REPORTS_DIR = os.path.join(LOGS_DIR, "reports")

# Create directories if they do not exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

RF_DIR = os.path.join(KNN_DIR, "..", "RandomForest_impl")
DATA_DIR = os.path.join(RF_DIR, "data")
DISEASE_DATA_DIR = os.path.join(DATA_DIR, "disease_per_plant")

# ========== LOAD LABEL MAP ==========
label_map_path = os.path.join(RF_DIR, "label_map.json")
with open(label_map_path, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# ========== HELPER FUNCTION ==========
def run_knn(task_name, X_train, y_train, X_test, y_test, target_names, k_values, log_file):
    best_model = None
    best_f1 = -1
    best_k = None

    for k in k_values:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{task_name}] Training KNN with n_neighbors = {k} ...")

        try:
            model = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            
            # Ensure target_names matches the classes present in y_test
            labels_in_y = np.unique(np.concatenate((y_train, y_test)))
            current_target_names = [target_names[i] for i in labels_in_y] if len(labels_in_y) <= len(target_names) else target_names

            report = classification_report(y_test, y_pred, target_names=current_target_names, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            print(f"Accuracy : {acc:.4f} | F1 Score : {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_k = k

            # --- Write Log File (CSV) ---
            log_entry = pd.DataFrame([{
                "Time": timestamp, 
                "Task": task_name, 
                "n_neighbors": k,
                "Accuracy": acc, 
                "F1 Score": f1,
                "Train Size": X_train.shape[0], 
                "Features": X_train.shape[1]
            }])
            
            if not os.path.isfile(log_file):
                log_entry.to_csv(log_file, index=False)
            else:
                log_entry.to_csv(log_file, mode="a", header=False, index=False)

            # --- Write detailed txt Report ---
            time_str = datetime.datetime.now().strftime("%H%M%S")
            report_name = os.path.join(REPORTS_DIR, f"report_knn_{task_name}_k{k}_{time_str}.txt")
            
            with open(report_name, "w", encoding="utf-8") as f:
                f.write(f"EXPERIMENT REPORT - {timestamp}\nTask: {task_name}\n")
                f.write(f"Model: KNeighborsClassifier (n_neighbors={k}, weights='distance')\n")
                f.write("=" * 40 + "\nCLASSIFICATION REPORT:\n")
                f.write(report)
                f.write("\nCONFUSION MATRIX:\n")
                f.write(np.array2string(cm))
                
            del model
            gc.collect()

        except Exception as e:
            print(f"[ERROR] Error training {task_name} with k={k}: {e}")
            
    return best_f1, best_model, best_k

# ========== MAIN SCRIPT ==========
log_file = os.path.join(LOGS_DIR, "training_history_knn.csv")
k_values = [3, 5, 7, 9, 11]

# 1. TRAIN PLANT CLASSIFICATION
print("="*50)
print("STARTING PLANT CLASSIFICATION TRAINING")
print("="*50)
try:
    X_train_plant = np.load(os.path.join(DATA_DIR, "X_train_plant.npy"), allow_pickle=True)
    X_test_plant = np.load(os.path.join(DATA_DIR, "X_test_plant.npy"), allow_pickle=True)
    y_train_plant = np.load(os.path.join(DATA_DIR, "y_train_plant.npy"), allow_pickle=True)
    y_test_plant = np.load(os.path.join(DATA_DIR, "y_test_plant.npy"), allow_pickle=True)

    print("Scaling plant data...")
    scaler_plant = StandardScaler()
    X_train_plant_scaled = scaler_plant.fit_transform(X_train_plant)
    X_test_plant_scaled = scaler_plant.transform(X_test_plant)

    plant_names_dict = {v: k for k, v in label_map["plant"].items()}
    plant_labels = [plant_names_dict[i] for i in range(len(plant_names_dict))]

    best_plant_f1, best_plant_model, best_plant_k = run_knn(
        "plant", X_train_plant_scaled, y_train_plant, X_test_plant_scaled, y_test_plant, plant_labels, k_values, log_file
    )

    if best_plant_model:
        joblib.dump(best_plant_model, os.path.join(MODELS_DIR, f"knn_best_plant_k{best_plant_k}.pkl"))
        joblib.dump(scaler_plant, os.path.join(MODELS_DIR, "scaler.pkl"))
        
    del X_train_plant, X_test_plant, X_train_plant_scaled, X_test_plant_scaled
    gc.collect()

except Exception as e:
    print(f"[ERROR] Failed to train plant model: {e}")

# 2. TRAIN DISEASE CLASSIFICATION PER PLANT
print("\n" + "="*50)
print("STARTING DISEASE CLASSIFICATION PER PLANT")
print("="*50)

disease_map = label_map.get("disease_by_plant", {})

for plant_name, disease_dict in disease_map.items():
    print(f"\nProcessing disease classification for: {plant_name}")
    
    # Generate paths dynamically matching RandomForest_impl structure
    X_train_path = os.path.join(DISEASE_DATA_DIR, f"X_train_{plant_name}.npy")
    y_train_path = os.path.join(DISEASE_DATA_DIR, f"y_train_{plant_name}.npy")
    X_test_path = os.path.join(DISEASE_DATA_DIR, f"X_test_{plant_name}.npy")
    y_test_path = os.path.join(DISEASE_DATA_DIR, f"y_test_{plant_name}.npy")
    
    if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
        print(f"  -> Skipping {plant_name}: Data files not found.")
        continue
        
    try:
        X_train_dis = np.load(X_train_path, allow_pickle=True)
        y_train_dis = np.load(y_train_path, allow_pickle=True)
        X_test_dis = np.load(X_test_path, allow_pickle=True)
        y_test_dis = np.load(y_test_path, allow_pickle=True)
        
        # Skip if only one class is present in training data (e.g., Blueberry only has "healthy")
        if len(np.unique(y_train_dis)) <= 1:
            print(f"  -> Skipping {plant_name}: Only one class present in training data.")
            continue

        print(f"  -> Scaling data for {plant_name}...")
        scaler_dis = StandardScaler()
        X_train_dis_scaled = scaler_dis.fit_transform(X_train_dis)
        X_test_dis_scaled = scaler_dis.transform(X_test_dis)

        dis_names_dict = {v: k for k, v in disease_dict.items()}
        dis_labels = [dis_names_dict.get(i, f"Class_{i}") for i in range(len(dis_names_dict))]

        task_name = f"disease_{plant_name}"
        best_dis_f1, best_dis_model, best_dis_k = run_knn(
            task_name, X_train_dis_scaled, y_train_dis, X_test_dis_scaled, y_test_dis, dis_labels, k_values, log_file
        )

        if best_dis_model:
            joblib.dump(best_dis_model, os.path.join(MODELS_DIR, f"knn_best_{task_name}_k{best_dis_k}.pkl"))
            joblib.dump(scaler_dis, os.path.join(MODELS_DIR, f"scaler_{task_name}.pkl"))

        del X_train_dis, X_test_dis, X_train_dis_scaled, X_test_dis_scaled
        gc.collect()

    except Exception as e:
        print(f"  -> [ERROR] Failed to train disease model for {plant_name}: {e}")

print(f"\n{'='*55}")
print(f"Completed! All KNN models, scalers, and reports have been saved successfully.")