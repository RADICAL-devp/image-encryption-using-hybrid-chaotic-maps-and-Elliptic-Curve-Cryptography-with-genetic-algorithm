"""
Security analysis metrics.

  H(s)       — Shannon entropy (ideal ≈ 8.0 bits)
  r_{τ,ν}    — Correlation coefficient (ideal ≈ 0)
  NPCR       — Number of Pixels Change Rate  (ideal ≈ 99.61%)
  UACI       — Unified Average Changing Intensity (ideal ≈ 33.46%)
  PSNR       — Peak Signal-to-Noise Ratio  (∞ dB = perfect reconstruction)
"""

import numpy as np


# ── Entropy ───────────────────────────────────────────────────────────────────

def shannon_entropy(image: np.ndarray) -> float:
    """H(s) = Σ P(sj) · log2(1 / P(sj))"""
    hist = np.bincount(image.flatten(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


# ── Correlation ───────────────────────────────────────────────────────────────

def _corr1d(a, b):
    cov = np.cov(a.astype(np.float64), b.astype(np.float64))
    denom = np.sqrt(cov[0, 0] * cov[1, 1])
    return float(cov[0, 1] / denom) if denom > 0 else 0.0


def correlation_coefficients(image: np.ndarray):
    """
    r_{τ,ν} = |cov(τ,ν)| / √(D(τ)·D(ν))
    Returns (H, V, D) — horizontal, vertical, diagonal.
    """
    H = _corr1d(image[:, :-1].ravel(), image[:, 1:].ravel())
    V = _corr1d(image[:-1, :].ravel(), image[1:, :].ravel())
    D = _corr1d(image[:-1, :-1].ravel(), image[1:, 1:].ravel())
    return H, V, D


# ── NPCR & UACI ───────────────────────────────────────────────────────────────

def npcr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    NPCR = (1/ST) · Σ E(i,j)   where E(i,j) = 0 if C1==C2, else 1
    Ideal ≈ 0.9961
    """
    return float(np.sum(img1 != img2) / img1.size)


def uaci(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    UACI = (1/ST) · Σ |C1 - C2| / 255
    Ideal ≈ 0.3346
    Cast to int16 first to avoid uint8 underflow.
    """
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    return float(diff.mean() / 255.0)


# ── PSNR ──────────────────────────────────────────────────────────────────────

def psnr(original: np.ndarray, decrypted: np.ndarray) -> float:
    """
    PSNR = 10 · log10(255² / MSE)
    Returns inf if images are identical (perfect reconstruction).
    """
    mse = float(np.mean(
        (original.astype(np.float64) - decrypted.astype(np.float64)) ** 2
    ))
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
