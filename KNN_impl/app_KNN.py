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
# Pointing to RandomForest_impl for the global label_map
RF_DIR = os.path.join(BASE_DIR, "..", "RandomForest_impl")
LABEL_MAP_PATH = os.path.join(RF_DIR, "label_map.json")

# ========== FEATURE EXTRACTION ==========
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
    color_feat = extract_color_histogram(image)
    texture_feat = extract_hog(image)
    shape_feat = extract_shape(image)
    return np.concatenate([color_feat, texture_feat, shape_feat])

# ========== APP ==========
class PlantDiseaseKNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detection - K-Nearest Neighbors (KNN)")
        self.root.geometry("1000x580")

        self.cv_image = None
        self.model_plant = None
        self.scaler_plant = None
        self.plant_names = {}
        self.disease_map = {} # Store the nested dictionary for diseases
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

        self.root.after(300, self.load_initial_models)

    def load_initial_models(self):
        try:
            # 1. Load label map
            if not os.path.exists(LABEL_MAP_PATH):
                raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}")
                
            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            self.plant_names = {v: k for k, v in label_data["plant"].items()}
            self.disease_map = label_data.get("disease_by_plant", {})

            # 2. Locate and load the Plant KNN model
            plant_model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("knn_best_plant")]
            if not plant_model_files:
                raise FileNotFoundError("KNN plant model not found. Run training script first.")
            self.model_plant = joblib.load(os.path.join(MODELS_DIR, plant_model_files[0]))

            # 3. Load the Plant Scaler
            scaler_plant_path = os.path.join(MODELS_DIR, "scaler.pkl")
            if not os.path.exists(scaler_plant_path):
                raise FileNotFoundError("scaler_plant.pkl not found in models directory.")
            self.scaler_plant = joblib.load(scaler_plant_path)

            self.is_model_ready = True
            self.lbl_status.config(text="Plant model loaded. Ready.", fg="green")
            self.canvas_label.config(text="Models ready.\nSelect a leaf image to start.")

        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            self.lbl_status.config(text="Initialization failed!", fg="red")

    def load_image(self):
        if not self.is_model_ready: return
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path: return

        self.cv_image = cv2.imread(file_path)
        if self.cv_image is None: return
        self.display_image(self.cv_image)

        # Clear previous text
        self.lbl_plant.config(text="Analyzing...")
        self.lbl_disease.config(text="-", fg="gray")
        self.root.update()
        
        ai_image = cv2.resize(self.cv_image, (256, 256))

        # Step 1: Extract Base Features
        feat = extract_features(ai_image)

        # Step 2: Predict Plant Type
        feat_scaled_plant = self.scaler_plant.transform([feat])
        plant_idx = self.model_plant.predict(feat_scaled_plant)[0]
        plant_name = self.plant_names.get(int(plant_idx), f"Unknown({plant_idx})")
        self.lbl_plant.config(text=f"🌱 {plant_name.replace('_', ' ')}")

        # Step 3: Predict Disease based on Plant Type
        self.predict_disease_for_plant(feat, plant_name)

    def predict_disease_for_plant(self, original_features, plant_name):
        # Look for the specific disease model and scaler for this plant
        disease_model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f"knn_best_disease_{plant_name}")]
        scaler_disease_path = os.path.join(MODELS_DIR, f"scaler_disease_{plant_name}.pkl")

        if disease_model_files and os.path.exists(scaler_disease_path):
            try:
                model_disease = joblib.load(os.path.join(MODELS_DIR, disease_model_files[0]))
                scaler_disease = joblib.load(scaler_disease_path)

                # Scale features using the SPECIFIC scaler for this plant's disease model
                feat_scaled_disease = scaler_disease.transform([original_features])
                disease_idx = model_disease.predict(feat_scaled_disease)[0]

                # Map index to disease name
                disease_dict = {v: k for k, v in self.disease_map.get(plant_name, {}).items()}
                disease_name = disease_dict.get(int(disease_idx), f"Unknown({disease_idx})")

                if disease_name.lower() == "healthy":
                    self.lbl_disease.config(text="✅ Healthy", fg="green")
                else:
                    self.lbl_disease.config(text=f"⚠️ {disease_name.replace('_', ' ')}", fg="red")
                    
                self.lbl_status.config(text="Prediction complete.", fg="blue")
                
            except Exception as e:
                self.lbl_disease.config(text="Error analyzing disease", fg="red")
                self.lbl_status.config(text=f"Error: {str(e)}", fg="red")
        else:
            # E.g., Blueberry only has healthy images, so no disease model was trained
            self.lbl_disease.config(text="✅ Healthy (No disease model)", fg="green")
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