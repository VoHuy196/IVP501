import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

class PlantDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phần mềm Phát hiện Bệnh trên Lá - Demo Xử lý ảnh")
        self.root.geometry("1200x600")

        # Biến lưu ảnh gốc và ảnh đang xử lý
        self.cv_image = None  # Ảnh dạng OpenCV
        self.display_image = None # Ảnh để hiển thị lên UI

        # --- KHUNG BÊN TRÁI (CONTROLS) ---
        self.left_frame = tk.Frame(root, width=300, bg="#f0f0f0")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Nút chọn ảnh
        self.btn_load = tk.Button(self.left_frame, text="📂 Chọn Ảnh Lá Cây", 
                                  command=self.load_image, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.btn_load.pack(pady=20, fill=tk.X)

        # Label kết quả
        self.lbl_result = tk.Label(self.left_frame, text="Tỷ lệ bệnh: 0.00%", 
                                   font=("Arial", 14), bg="#f0f0f0", fg="blue")
        self.lbl_result.pack(pady=10)

        # Các thanh trượt (Sliders) để chỉnh ngưỡng màu HSV
        # Group 1: Màu sắc (Hue)
        tk.Label(self.left_frame, text="--- Tinh chỉnh Màu (Hue) ---", bg="#f0f0f0").pack()
        self.h_min = self.create_slider("H Min", 0, 179, 0) # Màu vàng/nâu thường bắt đầu từ 0
        self.h_max = self.create_slider("H Max", 0, 179, 35) # Kết thúc ở khoảng 35

        # Group 2: Độ bão hòa (Saturation)
        tk.Label(self.left_frame, text="--- Tinh chỉnh Độ đậm (Sat) ---", bg="#f0f0f0").pack()
        self.s_min = self.create_slider("S Min", 0, 255, 50)
        self.s_max = self.create_slider("S Max", 0, 255, 255)

        # --- KHUNG BÊN PHẢI (HIỂN THỊ ẢNH) ---
        self.right_frame = tk.Frame(root, bg="gray")
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.canvas_label = tk.Label(self.right_frame, text="Vui lòng chọn ảnh", bg="gray", fg="white")
        self.canvas_label.pack(expand=True)

    def create_slider(self, label, min_val, max_val, default):
        """Hàm hỗ trợ tạo thanh trượt nhanh"""
        frame = tk.Frame(self.left_frame, bg="#f0f0f0")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label, width=10, anchor='w', bg="#f0f0f0").pack(side=tk.LEFT)
        scale = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, command=self.process_image)
        scale.set(default)
        scale.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        return scale

    def load_image(self):
        """Mở hộp thoại chọn file"""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        # Đọc ảnh bằng OpenCV
        self.cv_image = cv2.imread(file_path)
        if self.cv_image is None:
            messagebox.showerror("Lỗi", "Không thể đọc file ảnh!")
            return
        
        # Resize ảnh cho vừa khung hình (chiều ngang max 600px)
        h, w = self.cv_image.shape[:2]
        new_w = 600
        new_h = int(h * (600 / w))
        self.cv_image = cv2.resize(self.cv_image, (new_w, new_h))

        # Gọi hàm xử lý lần đầu
        self.process_image()

    def process_image(self, event=None):
        """Hàm xử lý chính (chạy mỗi khi load ảnh hoặc kéo thanh trượt)"""
        if self.cv_image is None:
            return

        # 1. Lấy giá trị từ thanh trượt
        lower_hsv = np.array([self.h_min.get(), self.s_min.get(), 40]) # Value min để 40 để lọc bóng đen
        upper_hsv = np.array([self.h_max.get(), self.s_max.get(), 255])

        # 2. Xử lý ảnh 
        img_display = self.cv_image.copy()
        hsv_img = cv2.cvtColor(img_display, cv2.COLOR_BGR2HSV)
        
        # Tạo mask bệnh
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
        
        # Lọc nhiễu
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Tính tỷ lệ
        total_pixels = img_display.shape[0] * img_display.shape[1]
        disease_pixels = cv2.countNonZero(mask)
        ratio = (disease_pixels / total_pixels) * 100

        # Vẽ viền vùng bệnh
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_display, contours, -1, (0, 0, 255), 2) # Vẽ viền màu ĐỎ

        # 3. Cập nhật UI
        # Cập nhật text kết quả
        self.lbl_result.config(text=f"Tỷ lệ bệnh: {ratio:.2f}%")
        if ratio > 10:
            self.lbl_result.config(fg="red", text=f"Tỷ lệ bệnh: {ratio:.2f}% ")
        else:
            self.lbl_result.config(fg="green")

        # Hiển thị ảnh lên Tkinter
        # Chuyển BGR (OpenCV) sang RGB (PIL)
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)

        # Gán ảnh vào Label
        self.canvas_label.config(image=tk_image, text="")
        self.canvas_label.image = tk_image # Giữ tham chiếu để không bị Garbage Collection xóa mất

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop()