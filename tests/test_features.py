"""Feature functions on small synthetic spectra with known answers (no data needed)."""

import numpy as np
import pytest

from frugal_eis.features import full_spectrum, full_spectrum_names, ohmic_feature


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


def test_ohmic_never_crossing_raises():
    with pytest.raises(ValueError):
        ohmic_feature(np.array([[0.4, 0.5]]), np.array([[-0.1, -0.05]]))
