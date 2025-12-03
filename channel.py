import numpy as np
from abc import ABC, abstractmethod


class Channel(ABC):
    """Abstract base class for memoryless channels."""
    rng = np.random.default_rng()

    @abstractmethod
    def transmit(self, c: np.ndarray) -> np.ndarray:
        """Transmit a codeword c through the channel and return the received vector."""
        pass


class BSCChannel(Channel):
    """Binary Symmetric Channel (BSC) with hard-decision output."""

    def __init__(self, p: float, force_errors: list[int] | None = None):
        """
        Parameters
        ----------
        p : float
            Crossover probability.
        force_errors : list[int] | None
            Optional list of bit positions that are forced to be flipped (debug mode).
        """
        self.p = float(p)
        self.force_errors = force_errors

    def transmit(self, c: np.ndarray) -> np.ndarray:
        y = c.copy().astype(int)
        if self.force_errors is not None:
            # Debug mode: force specific positions to be in error
            y[self.force_errors] ^= 1
        else:
            # Random error pattern according to BSC(p)
            noise = (self.rng.random(size=c.shape) < self.p).astype(int)
            y ^= noise
        return y


class BECChannel(Channel):
    """Binary Erasure Channel (BEC) with output in {0, 1, erasure_symbol}."""

    def __init__(self, p: float, erasure_symbol: int = -1):
        """
        Parameters
        ----------
        p : float
            Erasure probability.
        erasure_symbol : int
            Symbol used to represent an erasure.
        """
        self.p = float(p)
        self.erasure_symbol = erasure_symbol

    def transmit(self, c: np.ndarray) -> np.ndarray:
        y = c.copy().astype(int)
        erase_mask = self.rng.random(size=c.shape) < self.p
        y[erase_mask] = self.erasure_symbol
        return y


class BIAWGNChannel(Channel):
    """Binary-input AWGN channel with BPSK modulation."""

    def __init__(self, snr_db: float):
        """
        Parameters
        ----------
        snr_db : float
            Eb/N0 in dB.
        """
        self.ebn0_db = float(snr_db)
        ebn0_linear = 10.0 ** (self.ebn0_db / 10.0)
        self.sigma2 = 1.0 / (2.0 * ebn0_linear)
        self.sigma = np.sqrt(self.sigma2)

    def transmit(self, c: np.ndarray) -> np.ndarray:
        """
        Map bits c ∈ {0,1}^n to BPSK {+1, -1}, add AWGN, and return y.
        """
        c = c.astype(int)
        x = 1 - 2 * c  # BPSK: 0 → +1, 1 → -1
        noise = self.rng.normal(0.0, self.sigma, size=c.shape)
        y = x + noise
        return y
