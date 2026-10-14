"""Estimated acquisition time of an impedance measurement at a set of frequencies."""

from __future__ import annotations

import numpy as np


def sweep_time(freqs_selected, n_periods: float = 3, settle_s: float = 0) -> float:
    """Estimated time in seconds to measure the given frequencies: Σ (n_periods / f + settle_s).

    This is a **lower-bound estimate, not a measurement**. Assumptions:
    - each frequency is measured for ``n_periods`` full periods of the excitation signal, one frequency
      after another (no multisine excitation, which measures several frequencies at once);
    - ``settle_s`` per frequency is the only overhead, and it is 0 by default. Settling of the cell
      after the excitation changes, averaging of repeated readings and instrument overhead (range
      switching, data transfer) are all ignored;
    - the time of the 15 min rest before state V is not included: it is the same for every budget.

    Real instruments typically measure more than one period at low frequencies and add per-point
    overhead, so actual sweeps take longer. Use the estimate to compare frequency sets, not as an
    absolute time.
    """
    f = np.atleast_1d(np.asarray(freqs_selected, dtype=float))
    if f.size == 0:
        return 0.0
    if np.any(f <= 0):
        raise ValueError("frequencies must be positive")
    if n_periods <= 0 or settle_s < 0:
        raise ValueError("n_periods must be > 0 and settle_s >= 0")
    return float(np.sum(n_periods / f) + settle_s * f.size)
