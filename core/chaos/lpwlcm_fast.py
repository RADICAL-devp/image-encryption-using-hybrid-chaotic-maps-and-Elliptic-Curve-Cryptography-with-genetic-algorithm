"""
Hybrid LPWLCM (Logistic-Piecewise Linear Chaotic Map) — Diffusion layer.

      ⎧  r·x·(1−x)       if x < 0.5
f =   ⎨  x / p           if 0.5 ≤ x < p        (NOTE: p must be > 0.5 for this branch to be reachable)
      ⎩  (1−x) / (1−p)   if x ≥ p

This combines:
  • Logistic map nonlinearity  (x < 0.5)
  • PWLCM diffusion strength   (x ≥ 0.5)

Parameters:
  r = 3.99  (logistic, chaotic regime: 3.57 ≤ r ≤ 4)
  p = 0.7   (must be > 0.5 so the second branch is reachable)
"""
import numpy as np

try:
    from numba import njit

    @njit(fastmath=True, cache=True)
    def lpwlcm_sequence(n, x0, r=3.99, p=0.7, warmup=500):
        out = np.empty(n, dtype=np.float64)
        x = x0
        for _ in range(warmup):
            if x < 0.5:
                x = r * x * (1.0 - x)
            elif x < p:
                x = x / p
            else:
                x = (1.0 - x) / (1.0 - p)
        for i in range(n):
            if x < 0.5:
                x = r * x * (1.0 - x)
            elif x < p:
                x = x / p
            else:
                x = (1.0 - x) / (1.0 - p)
            out[i] = x
        return out

except ImportError:
    def lpwlcm_sequence(n, x0, r=3.99, p=0.7, warmup=500):
        out = np.empty(n, dtype=np.float64)
        x = float(x0)
        for _ in range(warmup):
            if x < 0.5:
                x = r * x * (1.0 - x)
            elif x < p:
                x = x / p
            else:
                x = (1.0 - x) / (1.0 - p)
        for i in range(n):
            if x < 0.5:
                x = r * x * (1.0 - x)
            elif x < p:
                x = x / p
            else:
                x = (1.0 - x) / (1.0 - p)
            out[i] = x
        return out


def xor_mask_from_lpwlcm(shape, x0, r=3.99, p=0.7):
    """
    Generate a uint8 XOR mask using the hybrid LPWLCM.
    Each value in [0,1] is scaled to [0,255].
    """
    n = shape[0] * shape[1]
    seq = lpwlcm_sequence(n, x0, r, p)
    return (seq * 255).astype(np.uint8).reshape(shape)
