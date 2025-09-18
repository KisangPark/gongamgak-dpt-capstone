import torch
from transformers import DPTImageProcessor, DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import time
import cv2

import pdb

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and feature extractor
#feature_extractor = DPTImageProcessor.from_pretrained("gillich/dpt-hybrid-midas-safetensor")
#model = DPTForDepthEstimation.from_pretrained("gillich/dpt-hybrid-midas-safetensor").to(device)
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-swinv2-tiny-256")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-swinv2-tiny-256").to(device)
feature_extractor.do_resize = False

cap = cv2.VideoCapture(0) # /dev/video0

try:
    ret, frame = cap.read()
    if not ret:
        raise ValueError("not returned")
except:
    raise ValueError("Exception occured, mola")

# 정사각형 이미지 input 넣기 
if ret:
    # frame.shape = (480, 640, 3)  # (height, width, channels)
    cropped = frame[112:368, 192:448]

rgb_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
image = Image.fromarray(rgb_image) # pil_



# check inference time
time_1 = time.time()

# Prepare image
inputs = feature_extractor(images=image, return_tensors="pt").to(device)

#pdb.set_trace()
# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth


# Convert to NumPy
#depth = predicted_depth.cpu().numpy()
depth = predicted_depth.squeeze(0).detach().cpu().numpy()  # [H, W], float32/64

#pdb.set_trace()

# Normalize for visualization
depth_min = depth.min()
depth_max = depth.max()
depth_vis = (depth - depth_min) / (depth_max - depth_min)

#depth_blocks_max = depth_vis.reshape(4, 64, 4, 64).max(axis=(1, 3))


depth_blocks = depth_vis.reshape(4, 64, 4, 64)  # (4, 64, 4, 64)
result = np.zeros((4, 4), dtype=depth.dtype)

for i in range(4):
    for j in range(4):
        block = depth_blocks[i, :, j, :]  # (64, 64)
        flat = block.ravel()              # (4096,)
        # np.partition: k번째로 작은 값을 빠르게 찾음
        # 10번째 큰 값 = (4096 - 10)번째 작은 값
        kth_val = np.partition(flat, -10)[-10]
        result[i, j] = kth_val

print(result.shape)  # (4, 4)
print(result)

#print(depth_blocks_max.shape)  # (4, 4)
#print(depth_blocks_max)

#pdb.set_trace()
#pdb.set_trace()

# inference time
time_2 = time.time()
time_diff = time_2 - time_1
print(f"Inference time: {time_diff}")

# Show result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth_vis, cmap="inferno")
plt.title("Predicted Depth")
plt.axis("off")

plt.tight_layout()
plt.savefig("./depth_result.png")  # 저장 (확장자는 png, jpg 등 선택 가능)
plt.show()
