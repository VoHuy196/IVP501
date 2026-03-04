import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

class PlantDiseaseLazyLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phát hiện Bệnh trên Lá - Supervised Learning (KNN)")
        self.root.geometry("1000x550")

        self.cv_image = None
        self.model = KNeighborsClassifier(n_neighbors=3) # Thuật toán KNN, xét 3 hàng xóm gần nhất
        self.is_model_ready = False

        # --- GIAO DIỆN KHUNG BÊN TRÁI ---
        self.left_frame = tk.Frame(root, width=300, bg="#f5f5f5", padx=20, pady=20)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.left_frame, text="HỆ THỐNG CHẨN ĐOÁN", font=("Arial", 14, "bold"), bg="#f5f5f5").pack(pady=10)

        self.btn_load = tk.Button(self.left_frame, text="📂 Chọn Ảnh Cần Test", 
                                  command=self.load_test_image, bg="#4CAF50", fg="white", 
                                  font=("Arial", 12, "bold"), height=2)
        self.btn_load.pack(pady=20, fill=tk.X)
        
        self.lbl_result = tk.Label(self.left_frame, text="Đang nạp dữ liệu học...", 
                                   font=("Arial", 14, "bold"), bg="#f5f5f5", fg="orange")
        self.lbl_result.pack(pady=10)

        # --- GIAO DIỆN KHUNG BÊN PHẢI ---
        self.right_frame = tk.Frame(root, bg="black")
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.canvas_label = tk.Label(self.right_frame, text="Vui lòng đợi hệ thống học từ ảnh mẫu...", 
                                     bg="black", fg="white", font=("Arial", 12))
        self.canvas_label.pack(expand=True)

        # TỰ ĐỘNG HỌC NGAY KHI MỞ APP (Dùng After để giao diện hiện lên trước rồi mới học)
        self.root.after(500, self.train_on_the_fly)

    def extract_features(self, image_path):
        """Trích xuất đặc trưng màu sắc (Color Histogram) của 1 bức ảnh"""
        img = cv2.imread(image_path)
        if img is None: return None
        img_resized = cv2.resize(img, (256, 256))
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def train_on_the_fly(self):
        """Hàm tự động đọc ảnh mẫu và huấn luyện siêu tốc"""
        data_dir = "mau_thu_nghiem" # Tên thư mục chứa ảnh mẫu
        
        if not os.path.exists(data_dir):
            messagebox.showerror("Thiếu dữ liệu học", 
                                 "Hãy tạo folder 'mau_thu_nghiem' và để ảnh mẫu vào 2 thư mục con 'khoe' và 'benh'")
            self.lbl_result.config(text="CHƯA HỌC XONG", fg="red")
            return

        X_train = []
        y_train = []
        
        # Đọc ảnh trong thư mục 'khoe' (Nhãn là 0)
        healthy_dir = os.path.join(data_dir, "khoe")
        if os.path.exists(healthy_dir):
            for file in os.listdir(healthy_dir):
                feat = self.extract_features(os.path.join(healthy_dir, file))
                if feat is not None:
                    X_train.append(feat)
                    y_train.append(0)

        # Đọc ảnh trong thư mục 'benh' (Nhãn là 1)
        disease_dir = os.path.join(data_dir, "benh")
        if os.path.exists(disease_dir):
            for file in os.listdir(disease_dir):
                feat = self.extract_features(os.path.join(disease_dir, file))
                if feat is not None:
                    X_train.append(feat)
                    y_train.append(1)

        # Bắt đầu cho máy "học" (Fit dữ liệu vào mô hình KNN)
        if len(X_train) > 0:
            self.model.fit(X_train, y_train)
            self.is_model_ready = True
            self.lbl_result.config(text="Đã học xong! Sẵn sàng.", fg="green")
            self.canvas_label.config(text="Hệ thống đã sẵn sàng.\nHãy bấm chọn ảnh để test.")
        else:
            self.lbl_result.config(text="Không có ảnh mẫu", fg="red")

    def load_test_image(self):
        """Mở ảnh để dự đoán"""
        if not self.is_model_ready:
            messagebox.showwarning("Cảnh báo", "Hệ thống chưa học xong dữ liệu mẫu!")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path: return

        self.cv_image = cv2.imread(file_path)
        self.display_image_on_ui(self.cv_image)
        
        # Trích xuất đặc trưng của ảnh vừa chọn
        test_features = self.extract_features(file_path)
        if test_features is not None:
            # Máy tính dự đoán
            prediction = self.model.predict([test_features])[0]
            
            # In kết quả (0 là khỏe, 1 là bệnh)
            if prediction == 0:
                self.lbl_result.config(text="✅ KHÔNG BỆNH", fg="green")
            else:
                self.lbl_result.config(text="⚠️ BỊ BỆNH", fg="red")

    def display_image_on_ui(self, img):
        """Hiển thị ảnh lên giao diện"""
        h, w = img.shape[:2]
        new_w = 650
        new_h = int(h * (650 / w))
        img_resized = cv2.resize(img, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas_label.config(image=tk_image, text="")
        self.canvas_label.image = tk_image 

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseLazyLearningApp(root)
    root.mainloop()