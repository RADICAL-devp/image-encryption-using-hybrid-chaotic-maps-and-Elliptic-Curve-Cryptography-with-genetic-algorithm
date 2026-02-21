import numpy as np

def salt_pepper(image: np.ndarray, density: float) -> np.ndarray:
    """Add salt-and-pepper noise to image (density = fraction of pixels corrupted)."""
    out  = image.copy()
    num  = int(density * image.size)
    rows = np.random.randint(0, image.shape[0], num)
    cols = np.random.randint(0, image.shape[1], num)
    out[rows, cols] = np.random.choice([0, 255], num)
    return out
