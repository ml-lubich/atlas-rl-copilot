import numpy as np

from atlas_rl_copilot.spectral import instability_index


def test_instability_increases_with_alternating_signs() -> None:
    smooth = np.linspace(10, 20, 64)
    noisy = np.array([10 + (i % 2) * 5 for i in range(64)], dtype=np.float64)
    assert instability_index(noisy) >= instability_index(smooth)


def test_short_series_returns_zero() -> None:
    assert instability_index(np.array([1.0, 2.0])) == 0.0
