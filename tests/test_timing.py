"""Sweep-time estimate on hand-computable frequency sets (no data needed)."""

import numpy as np
import pytest

from frugal_eis.timing import sweep_time


def test_sum_of_periods():
    # 3 periods at 1 Hz (3 s) + 3 periods at 0.1 Hz (30 s) + 3 periods at 10 Hz (0.3 s)
    assert sweep_time([1.0, 0.1, 10.0]) == pytest.approx(33.3)


def test_n_periods_and_settle():
    assert sweep_time([1.0, 0.5], n_periods=1) == pytest.approx(3.0)
    assert sweep_time([1.0, 0.5], n_periods=1, settle_s=2) == pytest.approx(7.0)


def test_scalar_and_empty():
    assert sweep_time(0.04) == pytest.approx(75.0)
    assert sweep_time([]) == 0.0


def test_low_frequencies_dominate():
    freqs = np.geomspace(20000, 0.02, 60)
    low = freqs[freqs < 1]
    assert sweep_time(low) / sweep_time(freqs) > 0.98


def test_invalid_inputs():
    with pytest.raises(ValueError):
        sweep_time([1.0, 0.0])
    with pytest.raises(ValueError):
        sweep_time([1.0], n_periods=0)
