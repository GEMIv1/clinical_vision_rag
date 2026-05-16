import cv2
import numpy as np
from app.preprocessing.transforms import resize, denoise, to_gray_scale, threshold, sharpen
from app.preprocessing.deskew import deskew

def preprocess(img_bytes: bytes) -> bytes:
    
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    
    img = denoise(img)
    img = deskew(img)
    img = resize(img, max_dim=640)
    gray = to_gray_scale(img)
    gray = threshold(gray)
    gray = sharpen(gray)
    
    _, buffer = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buffer.tobytes()