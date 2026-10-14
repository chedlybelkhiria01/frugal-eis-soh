"""Feature functions on small synthetic spectra with known answers (no data needed)."""

import numpy as np
import pytest

from frugal_eis.features import full_spectrum, full_spectrum_names, multiplicative_noise, ohmic_feature


def test_full_spectrum_order():
    re = np.array([[1.0, 2.0, 3.0]])
    im = np.array([[10.0, 20.0, 30.0]])
    np.testing.assert_array_equal(full_spectrum(re, im), [[1, 2, 3, 10, 20, 30]])
    assert full_spectrum_names(np.array([100.0, 17.8, 2.16])) == [
        "Re@100Hz", "Re@17.8Hz", "Re@2.16Hz", "mIm@100Hz", "mIm@17.8Hz", "mIm@2.16Hz"]


def test_ohmic_interpolates_zero_crossing():
    # -Im goes -0.02 -> +0.02 between Re 0.40 and 0.42: crossing exactly halfway, at 0.41
    re = np.array([[0.38, 0.40, 0.42, 0.50]])
    im = np.array([[-0.05, -0.02, 0.02, 0.10]])
    assert ohmic_feature(re, im)[0, 0] == pytest.approx(0.41)


def test_ohmic_without_inductive_part():
    re = np.array([[0.30, 0.35]])
    im = np.array([[0.01, 0.05]])
    assert ohmic_feature(re, im)[0, 0] == pytest.approx(0.30)


def test_ohmic_shape_and_rows_independent():
    re = np.array([[0.38, 0.40, 0.42], [0.50, 0.60, 0.70]])
    im = np.array([[-0.02, 0.02, 0.05], [-0.01, -0.01, 0.01]])
    r0 = ohmic_feature(re, im)
    assert r0.shape == (2, 1)
    np.testing.assert_allclose(r0[:, 0], [0.39, 0.65])


def test_multiplicative_noise_is_relative_and_independent():
    X = np.full((20000, 2), [0.1, 10.0])
    noisy = multiplicative_noise(X, 0.02, np.random.default_rng(0))
    rel = noisy / X - 1
    np.testing.assert_allclose(rel.std(axis=0), 0.02, rtol=0.03)      # same relative size at 0.1 Ω and 10 Ω
    assert abs(np.corrcoef(rel[:, 0], rel[:, 1])[0, 1]) < 0.03          # Re and Im perturbed independently
    np.testing.assert_array_equal(multiplicative_noise(X, 0, np.random.default_rng(0)), X)
    with pytest.raises(ValueError):
        multiplicative_noise(X, -0.1, np.random.default_rng(0))


def test_ohmic_never_crossing_raises():
    with pytest.raises(ValueError):
        ohmic_feature(np.array([[0.4, 0.5]]), np.array([[-0.1, -0.05]]))
