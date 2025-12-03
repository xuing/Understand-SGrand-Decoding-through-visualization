import numpy as np
from numpy._typing import NDArray


class LinearBlockCode:
    def __init__(self, n: int, k: int, name: str = "Generic Code"):
        self.n = n  # Codeword length
        self.k = k  # Information length
        self.name = name
        self.field_order = 2
        self.G = None
        self.H = None
        self.d_min = None
        self.t = None  # Error correction capability

        self.queries = 0  # just for statistics

    def encode(self, u: np.ndarray) -> np.ndarray:
        return (u @ self.G) % self.field_order

    def extract_info(self, c: np.ndarray) -> np.ndarray:
        return c[:self.k]

    def syndrome(self, c: np.ndarray) -> np.ndarray:
        return (c @ self.H.T) % self.field_order

    def is_codeword(self, c: np.ndarray) -> bool:
        self.queries += 1
        return not np.any(self.syndrome(c))

    def generate_message(self) -> NDArray:
        return np.random.default_rng().integers(
            0, self.field_order, size=(self.k,), dtype=np.int64
        )

    def generate_messages(self, num_blocks: int) -> NDArray:
        return np.random.default_rng().integers(
            0, self.field_order, size=(num_blocks, self.k), dtype=np.int64
        )


class ExtendedHamming84(LinearBlockCode):
    def __init__(self):
        super().__init__(8, 4, "Hamming(8,4)")
        self.G = np.array([
            [1, 0, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0],
        ], dtype=int)
        self.H = np.array([
            [1, 1, 0, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1],
        ], dtype=int)
        self.d_min = 4
        self.t = 1


def gf2_rref(mat):
    """
    Compute the Reduced Row Echelon Form (RREF) of a binary matrix over GF(2).
    Args:
        mat (np.ndarray): Input binary matrix (2D numpy array with elements 0 or 1).
    Returns:
        np.ndarray: RREF of the input matrix over GF(2).
    """
    M = mat.copy()
    rows, cols = M.shape
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows: break
        pivot = -1
        for r in range(pivot_row, rows):
            if M[r, col] == 1:
                pivot = r
                break
        if pivot == -1: continue
        if pivot != pivot_row:
            M[[pivot_row, pivot]] = M[[pivot, pivot_row]]
        for r in range(rows):
            if r != pivot_row and M[r, col] == 1:
                M[r] = M[r] ^ M[pivot_row]
        pivot_row += 1
    return M


class Golay24(LinearBlockCode):
    def __init__(self):
        super().__init__(24, 12, "Golay(24,12)")

        g_poly = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=int)
        G23 = np.zeros((12, 23), dtype=int)
        pad = np.zeros(23, dtype=int)
        pad[:12] = g_poly
        for i in range(12):
            G23[i] = np.roll(pad, i)

        G24_raw = np.hstack((G23, (np.sum(G23, axis=1) % 2).reshape(-1, 1)))
        G_sys = gf2_rref(G24_raw)

        self.G = G_sys
        P = G_sys[:, 12:]
        self.H = np.hstack((P.T, np.eye(12, dtype=int)))

        self.d_min = 8
        self.t = 3


class Golay23(LinearBlockCode):
    def __init__(self):
        super().__init__(23, 12, "Golay(23,12)")

        self.g_poly = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=int)
        self.h_poly = np.array([1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=int)

        self.d_min = 7
        self.t = 3

        self.G = np.zeros((12, 23), dtype=int)
        pad_g = np.zeros(23, dtype=int)
        pad_g[:12] = self.g_poly
        for i in range(12):
            self.G[i] = np.roll(pad_g, i)

        self.H = np.zeros((11, 23), dtype=int)
        h_rev = self.h_poly[::-1]
        pad_h = np.zeros(23, dtype=int)
        pad_h[-13:] = h_rev
        for i in range(11):
            self.H[i] = np.roll(pad_h, -i)

    def extract_info(self, c: np.ndarray) -> np.ndarray:
        u = np.zeros(self.k, dtype=int)
        for i in range(self.k):
            curr = c[i]
            for j in range(i):
                if self.G[j, i] == 1:
                    curr ^= u[j]
            u[i] = curr
        return u
