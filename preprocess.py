# preprocess.py
# Purpose: Exact image preprocessing pipeline required by competition
# - Crop black borders
# - Remove specular highlights (white glare inpainting)
# - Resize to 384x384 (optimal for BLIP-2 Flan-T5)
# - ImageNet normalization (handled by processor, but return ready PIL)

import cv2
import numpy as np
from PIL import Image

def crop_black_borders(image: Image.Image) -> Image.Image:
    """Crop black borders (common in endoscopy frames)"""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]
    return Image.fromarray(img)

def remove_specular_highlights(image: Image.Image) -> Image.Image:
    """Remove white glare using threshold + inpainting"""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(img)

def preprocess_image(image: Image.Image) -> Image.Image:
    """Full pipeline: crop -> remove highlights -> resize 384x384"""
    image = crop_black_borders(image)
    image = remove_specular_highlights(image)
    image = image.resize((384, 384), Image.BILINEAR)
    return image