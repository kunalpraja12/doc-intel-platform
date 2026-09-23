"""Computer vision preprocessing utilities for OCR.

Provides small helper functions that perform common preprocessing steps:
- load_image(path)
- to_grayscale(image)
- denoise(image)
- threshold(image)
- deskew(image)
- preprocess_image(path_or_image) — convenience wrapper

These are lightweight and intended for local development. Adjust or extend
for production (configurable pipelines, quality checks, heuristics, etc.).
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Union


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray) -> np.ndarray:
    # Fast denoise using bilateral filter which preserves edges
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def threshold(image: np.ndarray) -> np.ndarray:
    # Use Otsu thresholding after Gaussian blur for binarization
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def deskew(image: np.ndarray) -> np.ndarray:
    """Attempt to deskew using image moments / minAreaRect on edges.

    Returns a rotated image with the detected angle corrected.
    """
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.size == 0:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # minAreaRect angle handling
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_image(path_or_image: Union[str, np.ndarray], deskew_flag: bool = True) -> np.ndarray:
    if isinstance(path_or_image, str):
        img = load_image(path_or_image)
    else:
        img = path_or_image

    if deskew_flag:
        img = deskew(img)

    gray = to_grayscale(img)
    den = denoise(gray)
    th = threshold(den)
    # Return a 3-channel BGR image expected by OCR extractor
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
