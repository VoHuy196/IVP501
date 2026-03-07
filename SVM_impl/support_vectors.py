import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.decomposition import PCA
import joblib
import os
import datetime
import pandas as pd
import gc

# ========== REPORT DIR ==========
report_dir = "reports"
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

log_file = os.path.join(report_dir, "training_history.csv")
c_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# ========== DIMENSIONALITY REDUCTION ==========
pca = PCA(n_components=0.95)

# ========== LOAD DATA ==========
print("Loading train/test datasets...")

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ========== TRAINING MODEL ==========
print("Training SVM...")
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("Train shape after PCA:", X_train_pca.shape)
print("Test shape after PCA:", X_test_pca.shape)

for c_val in c_list:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Running SVM with C = {c_val}")

    try:
        model = LinearSVC(C=c_val, max_iter=1000, dual=False, verbose=1, tol=1e-2)
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        train_acc = model.score(X_train_pca, y_train)
        test_acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Done! Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {f1:.4f}")

        log_entry = pd.DataFrame([{
            "Time": timestamp,
            "C":    c_val,
            "Train accuracy": train_acc,
            "Test accuracy": test_acc,
            "F1 Score": f1,
            "Train Size": X_train_pca.shape[0],
            "Features": X_train_pca.shape[1]
        }])

        if not os.path.isfile(log_file):
            log_entry.to_csv(log_file, index=False)
        else:
            log_entry.to_csv(log_file, mode='a', header=False, index=False)

        # Report to txt
        report_name = os.path.join(report_dir, f"report_C{c_val}_{datetime.datetime.now().strftime('%H%M%S')}.txt")
        with open(report_name, "w", encoding="utf-8") as f:
            f.write(f"EXPERIMENT REPORT - {timestamp}\n")
            f.write(f"Model: LinearSVC (C={c_val})\n")
            f.write("="*40 + "\n")
            f.write("CLASSIFICATION REPORT:\n")
            f.write(f"Train Accuracy: {train_acc}\n")
            f.write(f"Test Accuracy: {test_acc}\n")
            f.write("-" * 30 + "\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\nCONFUSION MATRIX:\n")
            f.write(np.array2string(confusion_matrix(y_test, y_pred)))

        # ========== SAVE MODEL ==========
        joblib.dump(model, os.path.join(report_dir, f"model_C{c_val}.pkl"))

        del model
        del y_pred
        gc.collect()

    except Exception as e:
        print(f"[ERROR] Error at C={c_val}: {e}")
        continue

print(f"\n[DONE] All experiments are saved into {report_dir}")

