
from __future__ import annotations

import cv2
import numpy as np

def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def entropy(image: np.ndarray) -> float:
    if image.ndim == 2:
        hist = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
        p = hist / hist.sum()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    vals = []
    for ch in range(image.shape[2]):
        hist = np.bincount(image[:, :, ch].reshape(-1), minlength=256).astype(np.float64)
        p = hist / hist.sum()
        p = p[p > 0]
        vals.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(vals))

def correlation_coefficients(image: np.ndarray) -> tuple[float, float, float]:
    gray = to_gray(image).astype(np.float64)
    h, w = gray.shape
    if h < 2 or w < 2:
        return 0.0, 0.0, 0.0

    x_h = gray[:, :-1].flatten()
    y_h = gray[:, 1:].flatten()
    x_v = gray[:-1, :].flatten()
    y_v = gray[1:, :].flatten()
    x_d = gray[:-1, :-1].flatten()
    y_d = gray[1:, 1:].flatten()

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        a_mean = a.mean()
        b_mean = b.mean()
        num = np.mean((a - a_mean) * (b - b_mean))
        den = np.std(a) * np.std(b)
        return float(num / den) if den != 0 else 0.0

    return corr(x_h, y_h), corr(x_v, y_v), corr(x_d, y_d)

def npcr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Images must have the same shape for NPCR.")
    return float((a != b).mean() * 100.0)

def uaci(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Images must have the same shape for UACI.")
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)) / 255.0) * 100.0)
