"""Feature construction from impedance spectra.

All functions take ``X_re`` and ``X_im`` (= -Im(Z)) of shape (n_spectra, n_freq), frequencies descending,
as returned by :func:`frugal_eis.data.load_dataset`.
"""

from __future__ import annotations

import numpy as np


def full_spectrum(X_re: np.ndarray, X_im: np.ndarray) -> np.ndarray:
    """All 2 * n_freq features: [Re(f_1..f_n), -Im(f_1..f_n)] (Zhang et al. 2020 input)."""
    return np.hstack([X_re, X_im])


def full_spectrum_names(freqs: np.ndarray) -> list[str]:
    """Column names matching :func:`full_spectrum`, e.g. 'Re@17.8Hz', 'mIm@17.8Hz'."""
    return [f"Re@{f:.3g}Hz" for f in freqs] + [f"mIm@{f:.3g}Hz" for f in freqs]


def multiplicative_noise(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Relative measurement noise: X * (1 + sigma * N(0, 1)), drawn independently for every entry.

    On a full-spectrum matrix [Re, -Im] this perturbs Re(Z) and Im(Z) of every frequency independently,
    i.e. Z_noisy = Re(Z)(1 + σ ε1) + j Im(Z)(1 + σ ε2). Returns a new array; sigma = 0 returns a copy.
    """
    X = np.asarray(X, dtype=float)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return X.copy()
    return X * (1.0 + sigma * rng.standard_normal(X.shape))


def ohmic_feature(X_re: np.ndarray, X_im: np.ndarray) -> np.ndarray:
    """High-frequency real-axis intercept, an estimate of the ohmic resistance R0. Shape (n_spectra, 1).

    Walking from the highest frequency downwards, -Im(Z) starts negative (inductive part) and turns
    positive; R0 is Re(Z) where the spectrum crosses -Im = 0, found by linear interpolation between the
    two points around the first sign change. If the first point is already >= 0 (no inductive part),
    Re at the highest frequency is used.
    """
    n = len(X_re)
    r0 = np.empty(n)
    for i in range(n):
        re, im = X_re[i], X_im[i]
        k = int(np.argmax(im >= 0))           # first index with -Im >= 0
        if im[k] < 0:
            raise ValueError(f"spectrum {i}: -Im(Z) never crosses zero")
        if k == 0:
            r0[i] = re[0]
        else:
            t = -im[k - 1] / (im[k] - im[k - 1])   # fraction of the way from point k-1 to k
            r0[i] = re[k - 1] + t * (re[k] - re[k - 1])
    return r0[:, None]
