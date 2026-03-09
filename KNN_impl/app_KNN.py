import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import joblib
import os
from skimage.feature import hog
from skimage import color
from skimage.transform import resize

# ========== PATHS ==========
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Reuse scaler and label map from KNN_impl
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")

# Automatically locate trained KNN model filenames (k may vary between runs)
try:
    PLANT_MODEL_PATH = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.startswith("knn_best_plant")][0]
    DISEASE_MODEL_PATH = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.startswith("knn_best_disease")][0]
except IndexError:
    PLANT_MODEL_PATH = ""
    DISEASE_MODEL_PATH = ""

# ========== FEATURE EXTRACTION (same as RF version) ==========
def extract_color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.normalize(cv2.calcHist([hsv], [0], None, [32], [0, 180]), None).flatten()
    s_hist = cv2.normalize(cv2.calcHist([hsv], [1], None, [32], [0, 256]), None).flatten()
    v_hist = cv2.normalize(cv2.calcHist([hsv], [2], None, [32], [0, 256]), None).flatten()
    return np.concatenate([h_hist, s_hist, v_hist])

def extract_hog(image):
    image_resized = resize(image, (128, 128))
    gray = color.rgb2gray(image_resized)
    features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

def extract_shape(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return np.zeros(4)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    return np.array([area, perimeter, aspect_ratio, solidity])

def extract_features(image):
    feat = np.concatenate([extract_color_histogram(image), extract_hog(image), extract_shape(image)]).astype(np.float32)
    return feat

# ========== APP ==========
class PlantDiseaseKNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detection - K-Nearest Neighbors (KNN)")
        self.root.geometry("1000x580")

        self.cv_image = None
        self.model_plant = None
        self.model_disease = None
        self.scaler = None
        self.plant_names = {}
        self.disease_names = {}
        self.is_model_ready = False

        # --- LEFT PANEL ---
        self.left_frame = tk.Frame(root, width=320, bg="#f5f5f5", padx=20, pady=20)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.left_frame, text="PLANT DISEASE DETECTION", font=("Arial", 13, "bold"), bg="#f5f5f5", wraplength=260).pack(pady=10)
        tk.Label(self.left_frame, text="Model: K-Nearest Neighbors", font=("Arial", 10), bg="#f5f5f5", fg="gray").pack()

        self.btn_load = tk.Button(self.left_frame, text="📂 Select Leaf Image", command=self.load_image, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_load.pack(pady=20, fill=tk.X)

        tk.Label(self.left_frame, text="Plant Type:", font=("Arial", 11, "bold"), bg="#f5f5f5").pack(anchor="w")
        self.lbl_plant = tk.Label(self.left_frame, text="-", font=("Arial", 13, "bold"), bg="#f5f5f5", fg="#2196F3", wraplength=260, justify="left")
        self.lbl_plant.pack(anchor="w", pady=(0, 10))

        tk.Label(self.left_frame, text="Disease:", font=("Arial", 11, "bold"), bg="#f5f5f5").pack(anchor="w")
        self.lbl_disease = tk.Label(self.left_frame, text="-", font=("Arial", 13, "bold"), bg="#f5f5f5", fg="red", wraplength=260, justify="left")
        self.lbl_disease.pack(anchor="w", pady=(0, 10))

        self.lbl_status = tk.Label(self.left_frame, text="Loading models...", font=("Arial", 10), bg="#f5f5f5", fg="orange")
        self.lbl_status.pack(pady=10)

        # --- RIGHT PANEL ---
        self.right_frame = tk.Frame(root, bg="black")
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.canvas_label = tk.Label(self.right_frame, text="Select a leaf image to start.", bg="black", fg="white", font=("Arial", 12))
        self.canvas_label.pack(expand=True)

        self.root.after(300, self.load_models)

    def load_models(self):
        try:
            if not PLANT_MODEL_PATH or not DISEASE_MODEL_PATH:
                raise FileNotFoundError("KNN model not found. Please run the training script first.")

            self.model_plant = joblib.load(PLANT_MODEL_PATH)
            self.model_disease = joblib.load(DISEASE_MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)

            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                label_map = json.load(f)

            self.plant_names = {v: k for k, v in label_map["plant"].items()}
            self.disease_names = {v: k for k, v in label_map["disease"].items()}

            self.is_model_ready = True
            self.lbl_status.config(text="Models loaded. Ready.", fg="green")
            self.canvas_label.config(text="Models ready.\nSelect a leaf image to start.")

        except FileNotFoundError as e:
            messagebox.showerror("Model Load Error", str(e))
            self.lbl_status.config(text="Model file missing!", fg="red")

    def load_image(self):
        if not self.is_model_ready: return
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path: return

        self.cv_image = cv2.imread(file_path)
        if self.cv_image is None: return
        self.display_image(self.cv_image)

        # Extract and normalize features
        feat = extract_features(self.cv_image)
        feat_scaled = self.scaler.transform([feat])

        # Classify using KNN
        plant_idx = self.model_plant.predict(feat_scaled)[0]
        disease_idx = self.model_disease.predict(feat_scaled)[0]

        plant_name = self.plant_names.get(int(plant_idx), f"Unknown({plant_idx})")
        disease_name = self.disease_names.get(int(disease_idx), f"Unknown({disease_idx})")

        self.lbl_plant.config(text=plant_name)
        if disease_name.lower() == "healthy":
            self.lbl_disease.config(text="✅ Healthy", fg="green")
        else:
            self.lbl_disease.config(text=f"⚠️ {disease_name}", fg="red")

        self.lbl_status.config(text="Prediction complete.", fg="blue")

    def display_image(self, img):
        h, w = img.shape[:2]
        new_w = 650
        new_h = int(h * (new_w / w))
        resized = cv2.resize(img, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas_label.config(image=tk_img, text="")
        self.canvas_label.image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseKNNApp(root)
    root.mainloop()