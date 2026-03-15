import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from skimage import color
from skimage.transform import resize

"""
Extract Color Histogram features
"""
def extract_color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])

    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()

    return np.concatenate([h_hist, s_hist, v_hist])

"""
Extract texture features
choose HOG
"""
def extract_hog(image):
    image_resized = resize(image, (128, 128))
    gray = color.rgb2gray(image_resized)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

"""
Extract Shape features
choose Contour area
feature size = 4
"""
def extract_shape(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros(4)
    
    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0

    return np.array([area, perimeter, aspect_ratio, solidity])

"""
Final feature vector
Expect 96 color
~1764 (HOG)
4 shape
~1864 features per image
"""
def final_vector(image):
    color_feat = extract_color_histogram(image)
    texture_feat = extract_hog(image)
    shape_feat = extract_shape(image)

    return np.concatenate([color_feat, texture_feat, shape_feat])


def extract_features(image):
    """Shared inference entrypoint that reuses the training feature pipeline."""
    return final_vector(image).astype(np.float32)

# ========== FOLDER NAME PARSER ==========
# Format: PlantName___DiseaseName
def parse_folder(folder_name):
    parts = folder_name.split("___")
    if len(parts) != 2:
        return None, None
    return parts[0].strip(), parts[1].strip()

def build_dataset(dataset_path):
    X         = []
    y_plant   = []   # integer label -> which plant
    y_disease = []   # integer label -> local disease index within its plant

    print("Scanning dataset folders...")

    all_pairs = []
    for f in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, f)):
            plant, disease = parse_folder(f)
            if plant is not None:
                all_pairs.append((plant, disease))

    all_plants = sorted({p for p, d in all_pairs})
    plant_to_idx = {name: idx for idx, name in enumerate(all_plants)}

    # Build per-plant disease maps (disease_by_plant[plant] = {disease_name: local_idx})
    disease_by_plant = {}
    for plant in all_plants:
        diseases_for_plant = sorted({d for p, d in all_pairs if p == plant})
        disease_by_plant[plant] = {name: idx for idx, name in enumerate(diseases_for_plant)}

    print(f"Plant classes ({len(all_plants)}): {all_plants}")
    for plant, d_map in disease_by_plant.items():
        print(f"  {plant}: {list(d_map.keys())}")

    for folder_name in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        plant, disease = parse_folder(folder_name)
        if plant is None:
            print(f"  [SKIP] Cannot parse: {folder_name}")
            continue

        p_idx = plant_to_idx[plant]
        d_idx = disease_by_plant[plant][disease]   # local index within plant

        print(f"Processing '{folder_name}' -> plant={plant}({p_idx}), disease={disease}({d_idx})")

        for img_name in tqdm(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            image    = cv2.imread(img_path)
            if image is None:
                continue

            X.append(final_vector(image))
            y_plant.append(p_idx)
            y_disease.append(d_idx)

    X         = np.array(X,         dtype=np.float32)
    y_plant   = np.array(y_plant,   dtype=np.int32)
    y_disease = np.array(y_disease, dtype=np.int32)

    print("\nFinal dataset shape :", X.shape)
    print("y_plant  shape      :", y_plant.shape)
    print("y_disease shape     :", y_disease.shape)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "X.npy"),         X)
    np.save(os.path.join(data_dir, "y_plant.npy"),   y_plant)
    np.save(os.path.join(data_dir, "y_disease.npy"), y_disease)

    label_map = {
        "plant": plant_to_idx,
        "disease_by_plant": disease_by_plant
    }
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(base_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print("  data/X.npy")
    print("  data/y_plant.npy  (global plant index)")
    print("  data/y_disease.npy (local disease index per plant)")
    print("  label_map.json  (hierarchical: disease_by_plant)")


if __name__ == "__main__":
    dataset_path = r"D:\00.vanfolder\000.fsb\09.ComputerVision\demo\leafDisease\dataset\train\color"
    build_dataset(dataset_path)


