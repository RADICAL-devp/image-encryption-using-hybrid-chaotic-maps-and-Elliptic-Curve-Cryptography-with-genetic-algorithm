"""
Lorenz Chaotic Map — Confusion layer.

  dx/dt = sigma*(y - x)
  dy/dt = x*(rho - z) - y
  dz/dt = x*y - beta*z

Standard parameters: sigma=10, rho=28, beta=2.66
"""
import numpy as np

try:
    from numba import njit

    @njit(fastmath=True, cache=True)
    def lorenz_sequence(n, x0, y0, z0,
                        sigma=10.0, rho=28.0, beta=2.66,
                        dt=0.01, warmup=1000):
        out = np.empty(n, dtype=np.float64)
        x, y, z = x0, y0, z0
        for _ in range(warmup):
            dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
            x += dx*dt; y += dy*dt; z += dz*dt
        for i in range(n):
            dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
            x += dx*dt; y += dy*dt; z += dz*dt
            out[i] = x
        return out

except ImportError:
    def lorenz_sequence(n, x0, y0, z0,
                        sigma=10.0, rho=28.0, beta=2.66,
                        dt=0.01, warmup=1000):
        out = np.empty(n, dtype=np.float64)
        x, y, z = float(x0), float(y0), float(z0)
        for _ in range(warmup):
            dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
            x += dx*dt; y += dy*dt; z += dz*dt
        for i in range(n):
            dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
            x += dx*dt; y += dy*dt; z += dz*dt
            out[i] = x
        return out


def permutation_from_lorenz(n, x0=0.1, y0=0.2, z0=0.3):
    """
    Generate a pixel permutation of length n using the Lorenz attractor.
    Steps:
      1. Generate Lorenz sequence of length n
      2. argsort → unique permutation p such that seq[p] is sorted
      3. A'[i] = A[p[i]]  shuffles pixel positions
    """
    seq = lorenz_sequence(n, x0, y0, z0)
    return np.argsort(seq)   # shape (n,), dtype int64
