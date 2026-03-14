import os
import cv2 
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.feature import hog
from skimage import color
from skimage.transform import resize

feature_dir = Path(r"D:\project_ML\IVP501\SVM_impl\input_vectors")

if not os.path.exists(feature_dir):
    os.makedirs(feature_dir)
    print(f"Created directory: {feature_dir}")

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

# ========== LOAD DATASET ==========
def main():
    dataset_path = r"D:\project_ML\dataset\plantvillage dataset\color"
    X = []
    Y_plant = []        #   labels of species
    Y_diseased = []     #   labels of specific disease

    all_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    for folder_name in all_folders:
        """
        Separate the species name and disease name from the folder name
        e.g "Apple__Black_rot" -> plant = "Apple", disease = "Black_rot"
        """
        if "__" in folder_name:
            plant, disease = folder_name.split("__")
        else:
            plant = folder_name.split("__")[0]
            disease = folder_name.split("__")[1]

        folder_path = os.path.join(dataset_path, folder_name)
        image_files = os.listdir(folder_path)
        print(f"Processing Folder: {folder_name} (Plant: {plant}, Disease: {disease})")

        for img_name in tqdm(image_files):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            feature_vector = final_vector(image)
            X.append(feature_vector)
            Y_plant.append(plant)
            Y_diseased.append(disease)

        X = np.array(X, dtype=np.float32)
        Y_plant = np.array(Y_plant)
        Y_diseased = np.array(Y_diseased)

        print("Final dataset shape:", X.shape)
        print("Final Y plant dataset shape:", Y_plant.shape)
        print("Final Y diseased dataset shape:", Y_diseased.shape)

        # ========== SAVE ==========
        np.save(os.path.join(feature_dir, "X_features.npy"), X)
        np.save(os.path.join(feature_dir, "Y_plant.npy"), Y_plant)
        np.save(os.path.join(feature_dir, "Y_disease.npy"), Y_diseased)

        print("\n[DONE] Saved!\n")

if __name__ == "__main__":
    main()
    