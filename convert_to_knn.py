import numpy as np
import json
import os
from pathlib import Path

# Trỏ đường dẫn tới thư mục chứa dữ liệu của SVM và nhà mới của KNN
BASE_DIR = Path(__file__).parent
svm_dir = BASE_DIR / "SVM_impl" / "input_vectors"
knn_data_dir = BASE_DIR / "KNN_impl" / "data"
knn_base_dir = BASE_DIR / "KNN_impl"

os.makedirs(knn_data_dir, exist_ok=True)

def main():
    print("Dang doc du lieu tu SVM...")
    try:
        X = np.load(svm_dir / "X_features.npy")
        Y_plant_str = np.load(svm_dir / "Y_plant.npy")
        Y_disease_str = np.load(svm_dir / "Y_disease.npy")
    except FileNotFoundError:
        print("[LOI] Khong tim thay du lieu SVM. Ban chac chan da chay trich xuat SVM chua?")
        return

    print("Dang ma hoa Chu (String) sang So (Integer) cho KNN...")
    
    # 1. Tạo từ điển cho Loại Cây
    all_plants = sorted(list(set(Y_plant_str)))
    plant_to_idx = {name: idx for idx, name in enumerate(all_plants)}

    # 2. Tạo từ điển cho Bệnh (Dựa theo từng cây)
    disease_by_plant = {}
    for plant in all_plants:
        # Lấy tất cả các bệnh thuộc về cây này
        diseases = sorted(list(set(Y_disease_str[Y_plant_str == plant])))
        disease_by_plant[plant] = {name: idx for idx, name in enumerate(diseases)}

    # 3. Ép kiểu dữ liệu đồng loạt
    y_plant_int = np.array([plant_to_idx[p] for p in Y_plant_str], dtype=np.int32)
    y_disease_int = np.array([disease_by_plant[p][d] for p, d in zip(Y_plant_str, Y_disease_str)], dtype=np.int32)

    print("Dang luu du lieu vao thu muc cua KNN...")
    # Lưu Mảng
    np.save(knn_data_dir / "X.npy", X)
    np.save(knn_data_dir / "y_plant.npy", y_plant_int)
    np.save(knn_data_dir / "y_disease.npy", y_disease_int)

    # Lưu Từ điển JSON
    label_map = {
        "plant": plant_to_idx,
        "disease_by_plant": disease_by_plant
    }
    with open(knn_base_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print("\n[HOAN TAT] Du lieu da duoc chuyen doi va san sang cho KNN huan luyen.")

if __name__ == "__main__":
    main()