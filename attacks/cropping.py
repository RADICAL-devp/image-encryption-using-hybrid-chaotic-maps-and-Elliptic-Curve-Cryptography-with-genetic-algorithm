import numpy as np

def apply_cropping(image: np.ndarray, fraction: float) -> np.ndarray:
    """Zero out the top-left (fraction × fraction) region of the image."""
    h, w = image.shape[:2]
    out  = image.copy()
    out[:int(h * fraction), :int(w * fraction)] = 0
    return out
