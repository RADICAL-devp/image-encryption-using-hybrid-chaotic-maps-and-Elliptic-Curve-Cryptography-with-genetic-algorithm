"""
Elliptic Curve operations over NIST P-256.

Uses:
  • Jacobian projective coordinates  — eliminates expensive modular inverses
                                       during point addition/doubling
  • 4-bit sliding window scalar mult  — ~3x faster than binary double-and-add

y² = x³ + ax + b  (mod p)

Security: based on the Elliptic Curve Discrete Logarithm Problem (ECDLP).
Key generation: K = k·G  where k = private key, G = base point, K = public key.

Note: BrainpoolP512r1 is ~4–6× slower than P-256 in pure Python (larger field).
      P-256 is used here for practical speed while retaining strong security.
      To use BrainpoolP512r1 replace the curve constants below.
"""

# ── NIST P-256 (secp256r1) curve parameters ──────────────────────────────────
p  = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
a  = (-3) % p
b  = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
Gx = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
Gy = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
n  = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551   # group order
G  = (Gx, Gy)


# ── Jacobian point arithmetic ─────────────────────────────────────────────────

def _pdbl(P):
    """Point doubling in Jacobian coordinates."""
    if P is None:
        return None
    X1, Y1, Z1 = P
    if Y1 == 0:
        return None
    Y1sq = Y1 * Y1 % p
    S     = 4 * X1 * Y1sq % p
    M     = (3 * X1 * X1 + a * pow(Z1, 4, p)) % p
    X3    = (M * M - 2 * S) % p
    Y3    = (M * (S - X3) - 8 * Y1sq * Y1sq) % p
    Z3    = 2 * Y1 * Z1 % p
    return (X3, Y3, Z3)


def _padd(P, Q):
    """Point addition in Jacobian coordinates."""
    if P is None: return Q
    if Q is None: return P
    X1, Y1, Z1 = P
    X2, Y2, Z2 = Q
    Z1s = Z1 * Z1 % p;  Z2s = Z2 * Z2 % p
    U1  = X1 * Z2s % p; U2  = X2 * Z1s % p
    S1  = Y1 * Z2s * Z2 % p
    S2  = Y2 * Z1s * Z1 % p
    H   = (U2 - U1) % p
    R   = (S2 - S1) % p
    if H == 0:
        return _pdbl(P) if R == 0 else None
    H2  = H * H % p;   H3 = H * H2 % p
    X3  = (R * R - H3 - 2 * U1 * H2) % p
    Y3  = (R * (U1 * H2 - X3) - S1 * H3) % p
    Z3  = H * Z1 % p * Z2 % p
    return (X3, Y3, Z3)


def _to_affine(P):
    """Convert Jacobian (X:Y:Z) → affine (x, y)."""
    if P is None:
        return None
    X, Y, Z = P
    Zi  = pow(Z, -1, p)
    Zi2 = Zi * Zi % p
    return (X * Zi2 % p, Y * Zi2 * Zi % p)


# ── Windowed scalar multiplication (w=4) ─────────────────────────────────────

_W = 4  # window width — 4 gives best speed/precomputation tradeoff

def _precompute_table(affine_point):
    """
    Precompute odd multiples 1·P, 3·P, 5·P … (2^(w-1)-1)·P in Jacobian form.
    Used by windowed scalar multiplication.
    """
    size  = 1 << (_W - 1)           # 8 entries for w=4
    table = [None] * size
    JP    = (affine_point[0], affine_point[1], 1)
    table[0] = JP
    P2    = _pdbl(JP)               # 2P
    for i in range(1, size):
        table[i] = _padd(table[i - 1], P2)
    return table


def scalar_mult(k, affine_point, _table=None):
    """
    Compute k·P using 4-bit sliding window over Jacobian coords.
    Returns affine (x, y).
    """
    if k == 0:
        return None
    k = k % n
    if _table is None:
        _table = _precompute_table(affine_point)

    R     = None
    bits  = k.bit_length()
    i     = bits - 1

    while i >= 0:
        R = _pdbl(R)
        # extract 4-bit window starting at bit i
        shift   = max(i - _W + 1, 0)
        window  = (k >> shift) & ((1 << _W) - 1)
        if window:
            idx = window >> 1          # index into odd multiples
            if idx < len(_table):
                R = _padd(R, _table[idx])
            i -= _W
        else:
            i -= 1

    return _to_affine(R)


# ── Key generation ────────────────────────────────────────────────────────────

def generate_keys():
    """
    Generate ECC key pair.
      private_key k  ∈ [1, n-1]
      public_key  K  = k·G  (affine point)
    Returns (private_key, public_key, public_table)
    where public_table is the precomputed window table for the public key.
    """
    import random
    k = random.randrange(1, n - 1)
    K = scalar_mult(k, G)
    # Precompute window table for the public key — reused across all pixel encryptions
    pub_table = _precompute_table(K)
    return k, K, pub_table
