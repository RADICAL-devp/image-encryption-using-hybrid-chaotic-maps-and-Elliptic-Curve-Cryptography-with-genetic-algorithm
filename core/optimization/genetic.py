"""
Genetic Algorithm — post-encryption optimization.

Operates on the ECC-encrypted cipher image.
Minimizes the mean absolute inter-pixel correlation coefficient.

Pipeline (Section 3.3 of paper):
  Step A: Generate initial population of 64 images by applying random
          XOR masks to the cipher image (fast — no re-encryption).
  Step B: Fitness = mean(|corr_H|, |corr_V|, |corr_D|)  (lower = better)
  Step C: Select top 10%
  Step D: Crossover — uniform pixel crossover between two elites
  Step E: Mutation  — flip random pixels
  Repeat 100 generations → return best individual I_best = argmin(correlation)
"""

import numpy as np


# ── Correlation helper ────────────────────────────────────────────────────────

def _corr(a, b):
    """Pearson correlation between two 1-D arrays (float)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    da = a - a.mean(); db = b - b.mean()
    denom = np.sqrt((da**2).sum() * (db**2).sum())
    if denom == 0:
        return 0.0
    return float(np.dot(da, db) / denom)


def _fitness(img):
    """
    Mean absolute correlation in H, V, D directions.
    Lower → better randomness → better encryption.
    """
    H = abs(_corr(img[:, :-1].ravel(), img[:, 1:].ravel()))
    V = abs(_corr(img[:-1, :].ravel(), img[1:, :].ravel()))
    D = abs(_corr(img[:-1, :-1].ravel(), img[1:, 1:].ravel()))
    return (H + V + D) / 3.0


# ── GA ────────────────────────────────────────────────────────────────────────

def genetic_optimize(cipher: np.ndarray,
                     population_size: int = 64,
                     generations: int = 100,
                     elite_frac: float = 0.10,
                     mutation_rate: float = 0.01) -> np.ndarray:
    """
    Optimize the cipher image via a Genetic Algorithm.

    Parameters
    ----------
    cipher          : ECC-encrypted uint8 image (phase 2 output)
    population_size : number of individuals (default 64)
    generations     : number of GA iterations (default 100)
    elite_frac      : fraction selected as elite parents (default 10%)
    mutation_rate   : fraction of pixels mutated per child (default 1%)

    Returns
    -------
    Best individual found (uint8 ndarray, same shape as cipher).
    """
    rng       = np.random.default_rng()
    n_elite   = max(2, int(population_size * elite_frac))
    h, w      = cipher.shape
    n_pixels  = h * w

    # ── Step A: Initial population ────────────────────────────────────────────
    # Each individual = cipher XOR random mask (fast, no re-encryption)
    pop = [cipher.copy()]  # individual 0 is the unmodified cipher
    for _ in range(population_size - 1):
        mask = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        pop.append(np.bitwise_xor(cipher, mask).astype(np.uint8))

    best_img     = pop[0].copy()
    best_fitness = _fitness(best_img)

    for gen in range(generations):
        # ── Step B: Evaluate fitness ──────────────────────────────────────────
        fitness = np.array([_fitness(ind) for ind in pop])

        # Track global best
        gi = int(np.argmin(fitness))
        if fitness[gi] < best_fitness:
            best_fitness = float(fitness[gi])
            best_img     = pop[gi].copy()

        # ── Step C: Selection — top elite_frac ───────────────────────────────
        sorted_idx = np.argsort(fitness)
        elite      = [pop[i] for i in sorted_idx[:n_elite]]

        # ── Steps D & E: Crossover + Mutation ────────────────────────────────
        new_pop = [e.copy() for e in elite]   # elites carry over unchanged

        while len(new_pop) < population_size:
            i, j = rng.choice(n_elite, size=2, replace=False)
            # Uniform crossover
            mask_cross = rng.integers(0, 2, size=(h, w), dtype=np.uint8)
            child = np.where(mask_cross, elite[i], elite[j]).astype(np.uint8)
            # Mutation — flip random pixels
            n_mut = max(1, int(mutation_rate * n_pixels))
            rows  = rng.integers(0, h, n_mut)
            cols  = rng.integers(0, w, n_mut)
            child[rows, cols] = rng.integers(0, 256, n_mut, dtype=np.uint8)
            new_pop.append(child)

        pop = new_pop

    return best_img
