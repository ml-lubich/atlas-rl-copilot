"""Spectral diagnostics on scalar reward series (episode returns)."""

from __future__ import annotations

import math

import numpy as np


def instability_index(returns: np.ndarray) -> float:
    """
    Map high-frequency energy in the (centered) return sequence to [0, 1].

    Uses normalized high-frequency ratio: sum of |FFT[k]| for k > n/4 divided by total energy.
    Noisy, oscillating learning curves score higher than smooth monotone improvement.
    """
    if returns.size < 8:
        return 0.0
    x = returns.astype(np.float64, copy=False)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    if np.sum(spec) <= 1e-12:
        return 0.0
    n = spec.shape[0]
    cutoff = max(1, n // 4)
    high = float(np.sum(spec[cutoff:]))
    total = float(np.sum(spec))
    ratio = high / total
    # squash to (0,1)
    return float(1.0 - math.exp(-3.0 * ratio))
