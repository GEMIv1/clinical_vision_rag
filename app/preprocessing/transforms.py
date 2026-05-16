import cv2
import numpy as np

def resize(img: np.ndarray, max_dim: int) -> np.ndarray:
    h,w = img.shape[:2]
    if max(h,w) <= max_dim:
        return img
    scale = max_dim/max(h,w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
              [-1, 5, -1],
              [-1, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

def threshold(img: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

def to_gray_scale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    