"""
ECC ElGamal per-pixel encryption.

For each pixel M (0–255):
  1. Choose random r ∈ [1, n-1]
  2. C1 = r·G
  3. S  = r·PublicKey
  4. C2 = (M + S.x) mod 256

Ciphertext per pixel: (C1, C2)

Decryption:
  1. S  = PrivateKey · C1   →  S = k·r·G = r·(k·G) = r·PublicKey ✓
  2. M  = (C2 - S.x) mod 256

Security: breaking this requires solving the ECDLP (find k from k·G).
"""

import os
import random
import numpy as np
import multiprocessing as mp

from core.crypto.ecc_ops import (
    G, n, generate_keys,
    scalar_mult, _precompute_table
)

# ── Worker (must be at module level for multiprocessing pickle) ───────────────

_G_TABLE = None   # cached in each worker process

def _worker_init(g_table_data):
    """Called once per worker process to cache the G table."""
    global _G_TABLE
    _G_TABLE = g_table_data


def _encrypt_chunk(args):
    """
    Encrypt a flat chunk of pixel values.
    args = (pixels_flat, pub_key_affine, pub_table_data)
    Returns (enc_visual_flat, c1x_flat, c1y_flat, c2_flat)
    """
    pixels, pub_key, pub_table = args
    rng = random.SystemRandom()

    enc_vis = np.empty(len(pixels), dtype=np.uint8)
    c1x_arr = np.empty(len(pixels), dtype=object)
    c1y_arr = np.empty(len(pixels), dtype=object)
    c2_arr  = np.empty(len(pixels), dtype=object)

    for i, M in enumerate(pixels):
        r   = rng.randrange(1, n - 1)
        C1  = scalar_mult(r, G, _G_TABLE)
        S   = scalar_mult(r, pub_key, pub_table)
        C2  = (int(M) + S[0]) % n
        c1x_arr[i] = C1[0]
        c1y_arr[i] = C1[1]
        c2_arr[i]  = C2
        enc_vis[i] = C2 % 256

    return enc_vis, c1x_arr, c1y_arr, c2_arr


def _decrypt_chunk(args):
    """
    Decrypt a flat chunk.
    args = (c1x_flat, c1y_flat, c2_flat, private_key)
    Returns recovered pixel values as uint8 flat array.
    """
    c1x_arr, c1y_arr, c2_arr, priv = args
    out = np.empty(len(c1x_arr), dtype=np.uint8)

    for i in range(len(c1x_arr)):
        C1 = (int(c1x_arr[i]), int(c1y_arr[i]))
        C2 = int(c2_arr[i])
        S  = scalar_mult(priv, C1)
        M  = (C2 - S[0]) % n
        out[i] = int(M) & 0xFF

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def ecc_encrypt_image(image: np.ndarray, pub_key, pub_table,
                      n_workers: int = None):
    """
    ECC ElGamal encrypt every pixel in parallel.

    Returns:
      enc_visual  — uint8 image (C2 % 256) for saving / analysis
      ecc_store   — dict with c1x, c1y, c2 arrays needed for decryption
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count())

    flat   = image.flatten().tolist()
    total  = len(flat)
    chunk  = max(1, total // n_workers)
    chunks = [flat[i:i+chunk] for i in range(0, total, chunk)]

    # Precompute G table once and share (serialised via pickle)
    g_table = _precompute_table(G)
    args    = [(c, pub_key, pub_table) for c in chunks]

    with mp.Pool(n_workers,
                 initializer=_worker_init,
                 initargs=(g_table,)) as pool:
        results = pool.map(_encrypt_chunk, args)

    # Reassemble
    enc_vis_parts = [r[0] for r in results]
    c1x_parts     = [r[1] for r in results]
    c1y_parts     = [r[2] for r in results]
    c2_parts      = [r[3] for r in results]

    enc_visual = np.concatenate(enc_vis_parts).reshape(image.shape)
    c1x = np.concatenate(c1x_parts)
    c1y = np.concatenate(c1y_parts)
    c2  = np.concatenate(c2_parts)

    ecc_store = {'c1x': c1x, 'c1y': c1y, 'c2': c2, 'shape': image.shape}
    return enc_visual, ecc_store


def ecc_decrypt_image(ecc_store: dict, private_key: int,
                      n_workers: int = None) -> np.ndarray:
    """
    ECC ElGamal decrypt using the private key and stored ciphertext arrays.
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count())

    c1x   = ecc_store['c1x']
    c1y   = ecc_store['c1y']
    c2    = ecc_store['c2']
    shape = ecc_store['shape']
    total = len(c1x)
    chunk = max(1, total // n_workers)

    args = [
        (c1x[i:i+chunk], c1y[i:i+chunk], c2[i:i+chunk], private_key)
        for i in range(0, total, chunk)
    ]

    with mp.Pool(n_workers) as pool:
        results = pool.map(_decrypt_chunk, args)

    flat = np.concatenate(results)
    return flat.reshape(shape)
