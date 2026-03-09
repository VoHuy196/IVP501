import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import joblib
import os
import glob
from skimage.feature import hog
from skimage import color
from skimage.transform import resize

# ========== PATHS ==========
BASE_DIR        = os.path.join(os.path.dirname(__file__), "RandomForest_impl")
PLANT_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_best_plant_n200_depthfull.pkl")
SCALER_PATH      = os.path.join(BASE_DIR, "models", "scaler.pkl")
LABEL_MAP_PATH   = os.path.join(BASE_DIR, "label_map.json")
DISEASE_MODELS_DIR = os.path.join(BASE_DIR, "models", "disease_per_plant")

# ========== FEATURE EXTRACTION (same as feature_extraction.py) ==========
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
    if len(contours) == 0:
        return np.zeros(4)
    c            = max(contours, key=cv2.contourArea)
    area         = cv2.contourArea(c)
    perimeter    = cv2.arcLength(c, True)
    x, y, w, h   = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    hull         = cv2.convexHull(c)
    hull_area    = cv2.contourArea(hull)
    solidity     = float(area) / hull_area if hull_area != 0 else 0
    return np.array([area, perimeter, aspect_ratio, solidity])

def extract_features(image):
    feat = np.concatenate([
        extract_color_histogram(image),
        extract_hog(image),
        extract_shape(image)
    ]).astype(np.float32)
    return feat

def plant_to_safe_name(plant_name):
    """Convert plant name to the safe filename suffix used when saving models."""
    return plant_name.replace(",", "").replace(" ", "_").replace("(", "").replace(")", "")


# ========== APP ==========
class PlantDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detection - Random Forest")
        self.root.geometry("1000x580")

        self.cv_image           = None
        self.model_plant        = None
        self.disease_models     = {}   # {plant_name: model}
        self.scaler             = None
        self.plant_names        = {}   # {idx: plant_name}
        self.disease_by_plant   = {}   # {plant_name: {disease_name: local_idx}}
        self.is_model_ready     = False

        # --- LEFT PANEL ---
        self.left_frame = tk.Frame(root, width=320, bg="#f5f5f5", padx=20, pady=20)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.left_frame, text="PLANT DISEASE DETECTION",
                 font=("Arial", 13, "bold"), bg="#f5f5f5", wraplength=260).pack(pady=10)

        tk.Label(self.left_frame, text="Model: Random Forest",
                 font=("Arial", 10), bg="#f5f5f5", fg="gray").pack()

        self.btn_load = tk.Button(self.left_frame, text="📂 Select Leaf Image",
                                  command=self.load_image, bg="#4CAF50", fg="white",
                                  font=("Arial", 12, "bold"), height=2)
        self.btn_load.pack(pady=20, fill=tk.X)

        # Plant result
        tk.Label(self.left_frame, text="Plant Type:",
                 font=("Arial", 11, "bold"), bg="#f5f5f5").pack(anchor="w")
        self.lbl_plant = tk.Label(self.left_frame, text="-",
                                  font=("Arial", 13, "bold"), bg="#f5f5f5", fg="#2196F3",
                                  wraplength=260, justify="left")
        self.lbl_plant.pack(anchor="w", pady=(0, 10))

        # Disease result
        tk.Label(self.left_frame, text="Disease:",
                 font=("Arial", 11, "bold"), bg="#f5f5f5").pack(anchor="w")
        self.lbl_disease = tk.Label(self.left_frame, text="-",
                                    font=("Arial", 13, "bold"), bg="#f5f5f5", fg="red",
                                    wraplength=260, justify="left")
        self.lbl_disease.pack(anchor="w", pady=(0, 10))

        # Status bar
        self.lbl_status = tk.Label(self.left_frame, text="Loading models...",
                                   font=("Arial", 10), bg="#f5f5f5", fg="orange")
        self.lbl_status.pack(pady=10)

        # --- RIGHT PANEL ---
        self.right_frame = tk.Frame(root, bg="black")
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.canvas_label = tk.Label(self.right_frame,
                                     text="Select a leaf image to start.",
                                     bg="black", fg="white", font=("Arial", 12))
        self.canvas_label.pack(expand=True)

        # Load models after UI is rendered
        self.root.after(300, self.load_models)

    # ------------------------------------------------------------------
    def load_models(self):
        """Load plant model, all per-plant disease models, scaler, and label map."""
        try:
            self.model_plant = joblib.load(PLANT_MODEL_PATH)
            self.scaler      = joblib.load(SCALER_PATH)

            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                label_map = json.load(f)

            self.plant_names      = {v: k for k, v in label_map["plant"].items()}
            self.disease_by_plant = label_map["disease_by_plant"]

            # Load all per-plant disease models from models/disease_per_plant/
            loaded = 0
            if os.path.isdir(DISEASE_MODELS_DIR):
                for model_file in glob.glob(os.path.join(DISEASE_MODELS_DIR, "rf_disease_*.pkl")):
                    # filename: rf_disease_<SafePlantName>_n<X>_depth<Y>.pkl
                    # Match back to original plant name
                    for plant_name in self.disease_by_plant:
                        safe = plant_to_safe_name(plant_name)
                        basename = os.path.basename(model_file)
                        if basename.startswith(f"rf_disease_{safe}_"):
                            # Keep only the best (last loaded wins; training saves only best)
                            self.disease_models[plant_name] = joblib.load(model_file)
                            loaded += 1
                            break

            self.is_model_ready = True
            self.lbl_status.config(
                text=f"Models loaded. {loaded} disease models ready.", fg="green")
            self.canvas_label.config(
                text=f"Models ready ({loaded} disease classifiers).\nSelect a leaf image to start.")

        except FileNotFoundError as e:
            messagebox.showerror("Model Not Found", str(e))
            self.lbl_status.config(text="Model file missing!", fg="red")

    # ------------------------------------------------------------------
    def load_image(self):
        """Open file dialog, run hierarchical prediction, show results."""
        if not self.is_model_ready:
            messagebox.showwarning("Not Ready", "Models are still loading. Please wait.")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        self.cv_image = cv2.imread(file_path)
        if self.cv_image is None:
            messagebox.showerror("Error", "Cannot read image file.")
            return
        self.display_image(self.cv_image)

        # Extract + scale features
        feat        = extract_features(self.cv_image)
        feat_scaled = self.scaler.transform([feat])

        # --- Step 1: Predict plant ---
        plant_idx  = int(self.model_plant.predict(feat_scaled)[0])
        plant_name = self.plant_names.get(plant_idx, f"Unknown({plant_idx})")
        self.lbl_plant.config(text=plant_name)

        # --- Step 2: Predict disease using plant-specific model ---
        disease_model = self.disease_models.get(plant_name)
        if disease_model is None:
            # Single-class plant (e.g. Blueberry, Orange) — look up its only disease
            d_map    = self.disease_by_plant.get(plant_name, {})
            if len(d_map) == 1:
                disease_name = next(iter(d_map))
                if disease_name.lower() == "healthy":
                    self.lbl_disease.config(text="✅ Healthy", fg="green")
                else:
                    self.lbl_disease.config(text=f"⚠️ {disease_name}", fg="red")
            else:
                self.lbl_disease.config(text="No disease model for this plant", fg="orange")
            self.lbl_status.config(text="Prediction complete.", fg="blue")
            return

        disease_local_idx = int(disease_model.predict(feat_scaled)[0])

        # Look up disease name from hierarchical map
        d_map        = self.disease_by_plant.get(plant_name, {})
        idx_to_d     = {v: k for k, v in d_map.items()}
        disease_name = idx_to_d.get(disease_local_idx, f"Unknown({disease_local_idx})")

        if disease_name.lower() == "healthy":
            self.lbl_disease.config(text="✅ Healthy", fg="green")
        else:
            self.lbl_disease.config(text=f"⚠️ {disease_name}", fg="red")

        self.lbl_status.config(text="Prediction complete.", fg="blue")

    # ------------------------------------------------------------------
    def display_image(self, img):
        """Resize and show image on the right panel."""
        h, w    = img.shape[:2]
        new_w   = 650
        new_h   = int(h * (new_w / w))
        resized = cv2.resize(img, (new_w, new_h))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tk_img  = ImageTk.PhotoImage(pil_img)
        self.canvas_label.config(image=tk_img, text="")
        self.canvas_label.image = tk_img


if __name__ == "__main__":
    root = tk.Tk()
    app  = PlantDiseaseApp(root)
    root.mainloop()
