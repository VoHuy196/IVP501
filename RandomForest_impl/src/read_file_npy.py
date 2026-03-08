import numpy as np
import os

# ========== PATHS ==========
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Load data from files
X_train         = np.load(os.path.join(DATA_DIR, "X.npy"))
y_plant_train   = np.load(os.path.join(DATA_DIR, "y_plant.npy"))
y_disease_train = np.load(os.path.join(DATA_DIR, "y_disease.npy"))

# Hiển thị kích thước tổng thể (số lượng mẫu và số lượng đặc trưng)
print("Kích thước của X_train:", X_train.shape)
print("Kích thước của y_plant_train:", y_plant_train.shape)
print("Kích thước của y_disease_train:", y_disease_train.shape)

# In thử 2 mẫu dữ liệu đầu tiên để xem cấu trúc thực tế
print("\n--- 2 dòng dữ liệu X_train đầu tiên ---")
print(X_train[:2])
X_train_first_line = X_train[0]
print("\n--- Cấu trúc của dòng dữ liệu đầu tiên ---")
print("Số lượng đặc trưng:", len(X_train_first_line))
print("Giá trị đặc trưng đầu tiên:", X_train_first_line[:5])

print("\n--- 2 nhãn y_train đầu tiên ---")
print(y_plant_train[:2])
print(y_disease_train[:2])