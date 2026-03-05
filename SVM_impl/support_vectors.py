import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib
import os
import datetime
import pandas as pd

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

C_value = 10.0   # safe baseline value

model = LinearSVC(C=C_value, max_iter=5000, verbose=1, dual=False)
model.fit(X_train, y_train)

print("Training completed.")

# ========== PREDICTION ==========
print("Running prediction...")

y_pred = model.predict(X_test)

# ========== EVALUATION ==========
print("\nF1 Score:")
f1 = f1_score(y_test, y_pred)
print(f1)

acc = accuracy_score(y_test, y_pred)
print(acc)

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ========== SAVE MODEL ==========
joblib.dump(model, "svm_baseline.pkl")

print("\nModel saved: svm_baseline.pkl")

# ========== LOGGING ==========
log_file = "training_history.csv"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log_entry = pd.DataFrame([{
    "Time": timestamp,
    "C":    C_value,
    "Accuracy": acc,
    "F1 Score": f1,
    "Train Size": X_train.shape[0],
    "Features": X_train.shape[1]
}])

if not os.path.isfile(log_file):
    log_entry.to_csv(log_file, index=False)
else:
    log_entry.to_csv(log_file, mode='a', header=False, index=False)

# Report to txt
report_name = f"report_C{C_value}_{datetime.datetime.now().strftime('%H%M%S')}.txt"
with open(report_name, "w", encoding="utf-8") as f:
    f.write(f"EXPERIMENT REPORT - {timestamp}\n")
    f.write(f"Model: LinearSVC (C={C_value})\n")
    f.write("="*40 + "\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write(report)
    f.write("\nCONFUSION MATRIX:\n")
    f.write(np.array2string(cm))

print(f"\n[OK] Result is saved into '{log_file}' and '{report_name}'")

# ========== SAVE MODEL ==========
model_filename = "svm_baseline.pkl"
joblib.dump(model, model_filename)
print(f"Model saved: {model_filename}")

