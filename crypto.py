
from __future__ import annotations

import hashlib
import numpy as np

SECRET_KEY = "RADICAL-devp-image-encryption-v5"

def _derive_seed(shape: tuple[int, ...], channel_index: int, secret_key: str = SECRET_KEY) -> float:
    h, w = shape[:2]
    payload = f"{secret_key}|{h}|{w}|{channel_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed_int = int.from_bytes(digest[:8], "big", signed=False)
    seed = (seed_int % 10**12) / 10**12
    return seed if seed > 0.0 else 0.618033988749

def _chaotic_iterate(seed: float, rounds: int = 1200) -> float:
    x = seed % 1.0
    if x <= 0.0:
        x = 0.618033988749
    for _ in range(rounds):
        x = (3.999 * x * (1.0 - x) + 0.71 * np.sin(np.pi * x)) % 1.0
    return float(x)

def _make_rng(channel_shape: tuple[int, int], channel_index: int, secret_key: str = SECRET_KEY) -> np.random.Generator:
    seed0 = _derive_seed(channel_shape, channel_index, secret_key)
    seed1 = _chaotic_iterate(seed0, rounds=1200)
    seed_int = int(seed1 * (2**63 - 1)) ^ ((channel_index + 1) * 0x9E3779B97F4A7C15 & ((1 << 63) - 1))
    return np.random.default_rng(seed_int)

def _encrypt_channel(channel: np.ndarray, channel_index: int, secret_key: str = SECRET_KEY) -> np.ndarray:
    flat = channel.reshape(-1).astype(np.uint8)
    n = flat.size
    rng = _make_rng(channel.shape, channel_index, secret_key)

    perm = rng.permutation(n)
    permuted = flat[perm]
    keystream = rng.integers(0, 256, size=n, dtype=np.uint8)

    temp = np.bitwise_xor(permuted, keystream)
    cipher = np.empty_like(temp)
    cipher[0] = temp[0]
    for i in range(1, n):
        cipher[i] = (int(temp[i]) + int(cipher[i - 1])) & 0xFF
    return cipher.reshape(channel.shape)

def _decrypt_channel(channel: np.ndarray, channel_index: int, secret_key: str = SECRET_KEY) -> np.ndarray:
    flat = channel.reshape(-1).astype(np.uint8)
    n = flat.size
    rng = _make_rng(channel.shape, channel_index, secret_key)

    perm = rng.permutation(n)
    keystream = rng.integers(0, 256, size=n, dtype=np.uint8)

    temp = np.empty_like(flat)
    temp[0] = flat[0]
    for i in range(1, n):
        temp[i] = (int(flat[i]) - int(flat[i - 1])) & 0xFF

    permuted = np.bitwise_xor(temp, keystream)

    original = np.empty_like(permuted)
    original[perm] = permuted
    return original.reshape(channel.shape)

def encrypt_image(image: np.ndarray, secret_key: str = SECRET_KEY) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim == 2:
        return _encrypt_channel(image, 0, secret_key)
    if image.ndim == 3 and image.shape[2] == 3:
        out = np.empty_like(image)
        for ch in range(3):
            out[:, :, ch] = _encrypt_channel(image[:, :, ch], ch, secret_key)
        return out
    raise ValueError(f"Unsupported image shape: {image.shape}")

def decrypt_image(image: np.ndarray, secret_key: str = SECRET_KEY) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim == 2:
        return _decrypt_channel(image, 0, secret_key)
    if image.ndim == 3 and image.shape[2] == 3:
        out = np.empty_like(image)
        for ch in range(3):
            out[:, :, ch] = _decrypt_channel(image[:, :, ch], ch, secret_key)
        return out
    raise ValueError(f"Unsupported image shape: {image.shape}")
