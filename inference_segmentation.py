import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# 1. Load mô hình TorchScript
# -------------------------------
model = torch.jit.load("11.pt", map_location='cpu')
model.eval()

# -------------------------------
# 2. Define transform
# -------------------------------
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# -------------------------------
# 3. Hàm ánh xạ màu theo class
# -------------------------------
def decode_segmap(mask):
    label_colors = np.array([
        [2, 0, 0],         # class 0
        [127, 0, 0],       # class 1
        [248, 163, 191]    # class 2
    ])
    rgb_mask = label_colors[mask]
    return rgb_mask.astype(np.uint8)

# -------------------------------
# 4. Duyệt qua thư mục ảnh
# -------------------------------
input_folder = "test_input"  # ← thay tên thư mục chứa ảnh
output_folder = "test_output"  # ← thay tên thư mục lưu ảnh mask
os.makedirs(output_folder, exist_ok=True)

image_files = sorted([f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg", ".jpeg"))])

for file_name in image_files:
    image_path = os.path.join(input_folder, file_name)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()

    # Decode mask
    colored_mask = decode_segmap(pred_mask)
    mask_image = Image.fromarray(colored_mask)

    # Lưu kết quả
    mask_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_mask.png")
    mask_image.save(mask_path)

    # Hiển thị (nếu muốn)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Ảnh gốc")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_image)
    plt.title("Mask dự đoán")
    plt.tight_layout()
    plt.show()
