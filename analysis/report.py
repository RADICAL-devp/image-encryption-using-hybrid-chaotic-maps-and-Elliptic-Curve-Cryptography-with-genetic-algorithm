"""
Security report generator.
Saves histograms, metrics TXT, and metrics CSV into result_dir.
"""

import os
import csv
import datetime
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')   # non-interactive backend
import matplotlib.pyplot as plt

from analysis.metrics import (
    shannon_entropy, correlation_coefficients,
    npcr, uaci, psnr
)
from attacks.cropping import apply_cropping
from attacks.noise import salt_pepper


# ── Histogram ─────────────────────────────────────────────────────────────────

def _plot_histogram(image, title, path):
    plt.figure(figsize=(6, 3))
    plt.hist(image.flatten(), bins=256, range=(0, 255),
             color='steelblue', edgecolor='none')
    plt.title(title, fontsize=11)
    plt.xlabel('Pixel Intensity'); plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ── Main report function ──────────────────────────────────────────────────────

def generate_report(original_path: str,
                    encrypted_path: str,
                    decrypted_path: str,
                    result_dir: str,
                    timing: dict = None):
    """
    Run full security analysis and write:
      - original_hist.png, encrypted_hist.png, decrypted_hist.png
      - cropped_encrypted.png, noisy_encrypted.png
      - results.txt (human-readable)
      - results.csv (machine-readable)
    """
    original  = cv2.imread(original_path,  cv2.IMREAD_GRAYSCALE)
    encrypted = cv2.imread(encrypted_path, cv2.IMREAD_GRAYSCALE)
    decrypted = cv2.imread(decrypted_path, cv2.IMREAD_GRAYSCALE)

    if any(x is None for x in [original, encrypted, decrypted]):
        raise FileNotFoundError("Could not load one or more images for analysis.")

    os.makedirs(result_dir, exist_ok=True)

    # ── Histograms ────────────────────────────────────────────────────────────
    _plot_histogram(original,  "Original Histogram",
                    os.path.join(result_dir, "original_hist.png"))
    _plot_histogram(encrypted, "Encrypted Histogram",
                    os.path.join(result_dir, "encrypted_hist.png"))
    _plot_histogram(decrypted, "Decrypted Histogram",
                    os.path.join(result_dir, "decrypted_hist.png"))

    # ── Entropy ───────────────────────────────────────────────────────────────
    ent_plain = shannon_entropy(original)
    ent_enc   = shannon_entropy(encrypted)
    ent_dec   = shannon_entropy(decrypted)

    # ── Correlation ───────────────────────────────────────────────────────────
    H_p, V_p, D_p = correlation_coefficients(original)
    H_c, V_c, D_c = correlation_coefficients(encrypted)

    # ── NPCR & UACI ───────────────────────────────────────────────────────────
    npcr_val = npcr(original, encrypted)
    uaci_val = uaci(original, encrypted)

    # ── PSNR ──────────────────────────────────────────────────────────────────
    psnr_val = psnr(original, decrypted)
    psnr_str = "∞ (perfect)" if psnr_val == float('inf') else f"{psnr_val:.4f} dB"

    # ── Attack tests ──────────────────────────────────────────────────────────
    cropped  = apply_cropping(encrypted, 0.25)
    noisy    = salt_pepper(encrypted, 0.05)
    cv2.imwrite(os.path.join(result_dir, "cropped_encrypted.png"), cropped)
    cv2.imwrite(os.path.join(result_dir, "noisy_encrypted.png"),   noisy)
    npcr_crop  = npcr(encrypted, cropped);  uaci_crop  = uaci(encrypted, cropped)
    npcr_noise = npcr(encrypted, noisy);    uaci_noise = uaci(encrypted, noisy)

    # ── TXT report ───────────────────────────────────────────────────────────
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txt = os.path.join(result_dir, "results.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Image Encryption Security Report\n")
        f.write(f"Generated : {ts}\n")
        f.write("=" * 52 + "\n\n")

        f.write("Shannon Entropy  [ideal cipher ≈ 8.0 bits]\n")
        f.write(f"  Plain     : {ent_plain:.6f} bits\n")
        f.write(f"  Encrypted : {ent_enc:.6f} bits\n")
        f.write(f"  Decrypted : {ent_dec:.6f} bits\n\n")

        f.write("Correlation Coefficients  [ideal cipher ≈ 0]\n")
        f.write(f"  Plain     H={H_p:+.6f}  V={V_p:+.6f}  D={D_p:+.6f}\n")
        f.write(f"  Encrypted H={H_c:+.6f}  V={V_c:+.6f}  D={D_c:+.6f}\n\n")

        f.write("NPCR & UACI  (original vs encrypted)\n")
        f.write(f"  NPCR : {npcr_val*100:.4f}%   [ideal ≈ 99.6094%]\n")
        f.write(f"  UACI : {uaci_val*100:.4f}%   [ideal ≈ 33.4635%]\n\n")

        f.write("PSNR  (original vs decrypted)\n")
        f.write(f"  PSNR : {psnr_str}\n\n")

        f.write("Attack Robustness — Cropping 25%\n")
        f.write(f"  NPCR : {npcr_crop*100:.4f}%   UACI : {uaci_crop*100:.4f}%\n\n")

        f.write("Attack Robustness — Salt & Pepper Noise 5%\n")
        f.write(f"  NPCR : {npcr_noise*100:.4f}%   UACI : {uaci_noise*100:.4f}%\n\n")

        if timing:
            f.write("Execution Time\n")
            for k_, v_ in timing.items():
                f.write(f"  {k_:<22}: {v_:.4f}s\n")

    # ── CSV report ────────────────────────────────────────────────────────────
    csv_path = os.path.join(result_dir, "results.csv")
    rows = [
        ["Metric", "Value", "Ideal"],
        ["Entropy (Plain)",            f"{ent_plain:.6f}",        "—"],
        ["Entropy (Encrypted)",        f"{ent_enc:.6f}",          "~8.0"],
        ["Entropy (Decrypted)",        f"{ent_dec:.6f}",          "—"],
        ["Correlation H (Plain)",      f"{H_p:+.6f}",             "—"],
        ["Correlation V (Plain)",      f"{V_p:+.6f}",             "—"],
        ["Correlation D (Plain)",      f"{D_p:+.6f}",             "—"],
        ["Correlation H (Encrypted)",  f"{H_c:+.6f}",             "~0"],
        ["Correlation V (Encrypted)",  f"{V_c:+.6f}",             "~0"],
        ["Correlation D (Encrypted)",  f"{D_c:+.6f}",             "~0"],
        ["NPCR (%)",                   f"{npcr_val*100:.4f}",     "~99.6094"],
        ["UACI (%)",                   f"{uaci_val*100:.4f}",     "~33.4635"],
        ["PSNR (dB)",                  psnr_str,                  "inf (perfect)"],
        ["NPCR Cropping (%)",          f"{npcr_crop*100:.4f}",    "—"],
        ["UACI Cropping (%)",          f"{uaci_crop*100:.4f}",    "—"],
        ["NPCR Noise (%)",             f"{npcr_noise*100:.4f}",   "—"],
        ["UACI Noise (%)",             f"{uaci_noise*100:.4f}",   "—"],
    ]
    if timing:
        rows.append(["", "", ""])
        rows.append(["Timing", "Seconds", ""])
        for k_, v_ in timing.items():
            rows.append([k_, f"{v_:.4f}", ""])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"  Results saved → {result_dir}/results.txt  |  results.csv")
    return {
        "entropy_plain": ent_plain, "entropy_enc": ent_enc,
        "corr_plain": (H_p, V_p, D_p), "corr_enc": (H_c, V_c, D_c),
        "npcr": npcr_val, "uaci": uaci_val, "psnr": psnr_val,
    }
