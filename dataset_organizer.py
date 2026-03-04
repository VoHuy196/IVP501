import os
import shutil

# ====== PATHS ======
input_root = r"D:\project_ML\dataset\plantvillage dataset\color"
output_root = r"D:\project_ML\binary_dataset"

healthy_dir = os.path.join(output_root, "healthy")
diseased_dir = os.path.join(output_root, "diseased")

os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(diseased_dir, exist_ok=True)

# ====== PROCESS ======
for folder_name in os.listdir(input_root):
    folder_path = os.path.join(input_root, folder_name)

    if not os.path.isdir(folder_path):
        continue

    # Decide class
    if folder_name.endswith("__healthy"):
        target_dir = healthy_dir
    else:
        target_dir = diseased_dir

    # Copy images
    for img_name in os.listdir(folder_path):
        src = os.path.join(folder_path, img_name)
        dst = os.path.join(target_dir, img_name)

        # Avoid name collision
        if os.path.exists(dst):
            base, ext = os.path.splitext(img_name)
            dst = os.path.join(target_dir, base + "_" + folder_name + ext)

        shutil.copy2(src, dst)

print("Done.")
