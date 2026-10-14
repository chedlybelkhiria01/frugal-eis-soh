"""Model factories on small synthetic data (no dataset needed)."""

import numpy as np
import pytest

from frugal_eis.models import ARDGaussianProcess, gpr_ard, mlp, random_forest


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = np.sin(X[:, 0]) + 0.01 * rng.normal(size=n)   # only feature 0 matters
    return X, y


@pytest.mark.parametrize("factory", [random_forest, mlp, gpr_ard])
def test_factories_fit_and_predict(factory):
    X, y = _toy()
    m = factory().fit(X, y)
    pred = m.predict(X)
    assert pred.shape == y.shape
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.3   # learns the easy toy function


def test_factories_return_fresh_unfitted_models():
    for factory in (random_forest, mlp, gpr_ard):
        assert factory() is not factory()


def test_gpr_ard_ranks_the_relevant_feature_first():
    X, y = _toy()
    m = ARDGaussianProcess().fit(X, y)
    assert m.length_scales_.shape == (4,)
    assert int(np.argmin(m.length_scales_)) == 0          # ARD: shortest length scale = relevant feature
    assert m.length_scales_[0] < m.length_scales_[1:].min()


def test_gpr_stride_subsamples_training_rows():
    X, y = _toy(n=100)
    m = ARDGaussianProcess(stride=2).fit(X, y)
    assert m.pipeline_.named_steps["gpr"].X_train_.shape[0] == 50


def test_gpr_returns_uncertainty():
    X, y = _toy()
    mean, std = ARDGaussianProcess().fit(X, y).predict(X[:5], return_std=True)
    assert mean.shape == std.shape == (5,)
    assert (std > 0).all()
