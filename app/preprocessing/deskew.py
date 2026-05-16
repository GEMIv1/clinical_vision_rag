import cv2
import numpy as np

def deskew(img: np.ndarray) -> np.ndarray:
    
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 100, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return img
    
    angles = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return img
    
    median_angle = np.median(angles)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated