import torch
from transformers import DPTImageProcessor, DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import time
import cv2

# import pdb


class DPT():
    def __init__(self):
        # Check for CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-swinv2-tiny-256")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-swinv2-tiny-256").to(self.device)

        #feature_extractor = DPTImageProcessor.from_pretrained("gillich/dpt-hybrid-midas-safetensor")
        #model = DPTForDepthEstimation.from_pretrained("gillich/dpt-hybrid-midas-safetensor").to(device)

        self.cap = cv2.VideoCapture(0) # /dev/video0

    def capture(self):
        # capture with cv

        # try:
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("not returned")
        # except:
        #     raise ValueError("Exception occured, mola")

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(rgb_image) # pil_

    def receive_image(self, image):
        self.image = image
        self.image = Image.fromarray(rgb_image)



    def get_depth(self):

        # Prepare image
        inputs = self.feature_extractor(images=self.image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            depth = predicted_depth.squeeze(0).detach().cpu().numpy()  # [H, W], float32/64

            depth_min = depth.min()
            depth_max = depth.max()
            self.depth_vis = (depth - depth_min) / (depth_max - depth_min) # normalized

        return self.depth_vis


    def get_image(self):
        return self.image


    """ FUNCTION: segment image & return haptic data """
    def get_haptic(self):
        depth_map = self.get_depth()
        width, height = depth_map.shape() # 256, 256

        # haptic signal loop

    # def get_depth_bbox(self, bbox_list):
    #     # crop image with bounding box
    #     for bbox in bbox_list:
    #         for bbox[0]

