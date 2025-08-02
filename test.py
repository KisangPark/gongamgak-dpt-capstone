import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# Load image
# image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dpt_depth_example.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and feature extractor
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)

# Prepare image
inputs = feature_extractor(images=image, return_tensors="pt").to(device)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Resize to original image size
predicted_depth = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],  # (height, width)
    mode="bicubic",
    align_corners=False,
).squeeze()

# Convert to NumPy
depth = predicted_depth.cpu().numpy()

# Normalize for visualization
depth_min = depth.min()
depth_max = depth.max()
depth_vis = (depth - depth_min) / (depth_max - depth_min)

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
plt.show()
