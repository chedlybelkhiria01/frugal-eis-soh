"""Leakage tests: no cell may appear on both sides of a split."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from conftest import requires_data
from frugal_eis.evaluate import loco_evaluate, loco_splits, save_results, summarize
from frugal_eis.models import GroupRidgeCV


def _check_loco(groups):
    groups = np.asarray(groups)
    held_out = []
    for train_idx, test_idx, cell in loco_splits(groups):
        train_cells, test_cells = set(groups[train_idx]), set(groups[test_idx])
        assert train_cells.isdisjoint(test_cells)
        assert test_cells == {cell}
        assert len(train_idx) + len(test_idx) == len(groups)
        held_out.append(cell)
    assert sorted(held_out) == sorted(set(groups))     # every cell held out exactly once


def test_loco_disjoint_synthetic():
    groups = np.array(["a", "a", "b", "c", "c", "c", "b", "a"])   # interleaved on purpose
    _check_loco(groups)
    assert len(list(loco_splits(groups))) == 3


@requires_data
def test_loco_disjoint_real(ds_all):
    _check_loco(ds_all.cell_id)


# --- loco_evaluate ------------------------------------------------------------------------------

def _toy(n_per_cell=20, cells=("a", "b", "c", "d"), seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(cells, n_per_cell)
    X = rng.normal(size=(len(groups), 3))
    y = 1.0 + X @ np.array([0.5, -0.2, 0.1])          # exactly linear
    return X, y, groups


def test_loco_evaluate_perfect_linear_model():
    X, y, groups = _toy()
    res = loco_evaluate(LinearRegression, X, y, groups, model_name="lin")
    assert list(res.per_cell["cell"]) == ["a", "b", "c", "d"]
    np.testing.assert_allclose(res.per_cell["rmse"], 0, atol=1e-10)
    np.testing.assert_allclose(res.per_cell["r2"], 1)
    assert len(res.predictions) == len(y)
    assert set(res.predictions.columns) >= {"model", "cell", "y_true", "y_pred"}


def test_loco_evaluate_metrics_match_sklearn():
    X, y, groups = _toy()
    res = loco_evaluate(DummyRegressor(), X, y, groups)   # instance -> cloned per fold
    for _, row in res.per_cell.iterrows():
        p = res.predictions[res.predictions.cell == row.cell]
        assert row.rmse == pytest.approx(np.sqrt(mean_squared_error(p.y_true, p.y_pred)))
        assert row.mae == pytest.approx(mean_absolute_error(p.y_true, p.y_pred))
        assert row.r2 == pytest.approx(r2_score(p.y_true, p.y_pred))
        # the dummy predicts the training mean, i.e. the mean of the OTHER cells
        assert p.y_pred.iloc[0] == pytest.approx(y[groups != row.cell].mean())


class _SpyModel(BaseEstimator, RegressorMixin):
    """Records which group ids (stored in column 0) it was trained on."""
    seen: list = []

    def fit(self, X, y):
        _SpyModel.seen.append(set(X[:, 0]))
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_loco_evaluate_never_trains_on_held_out_cell():
    groups = np.array([0, 0, 1, 1, 1, 2, 2, 0], dtype=float)
    X = np.c_[groups, np.zeros(len(groups))]
    _SpyModel.seen = []
    res = loco_evaluate(_SpyModel, X, np.arange(len(groups), dtype=float), groups.astype(int).astype(str))
    for trained_on, held_out in zip(_SpyModel.seen, res.per_cell["cell"]):
        assert float(held_out) not in trained_on


class _GroupSpy(BaseEstimator, RegressorMixin):
    """Accepts groups in fit; records them."""
    seen: list = []

    def fit(self, X, y, groups):
        _GroupSpy.seen.append(set(groups))
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_loco_evaluate_passes_training_groups_only():
    X, y, groups = _toy()
    _GroupSpy.seen = []
    res = loco_evaluate(_GroupSpy, X, y, groups)
    for seen, held_out in zip(_GroupSpy.seen, res.per_cell["cell"]):
        assert seen == set(groups) - {held_out}


def test_group_ridge_cv_selects_from_grid():
    X, y, groups = _toy()
    m = GroupRidgeCV(alphas=np.array([1e-3, 1.0, 1e3])).fit(X, y, groups)
    assert m.alpha_ in (1e-3, 1.0, 1e3)
    assert m.alpha_ != 1e3                     # exactly linear data: heavy shrinkage must lose
    assert m.predict(X).shape == y.shape


def test_loco_evaluate_saves_csv(tmp_path):
    X, y, groups = _toy()
    cycle = np.tile(np.arange(20), 4)
    r1 = loco_evaluate(LinearRegression, X, y, groups, model_name="lin", cycle=cycle,
                       experiment="toy", results_dir=tmp_path)
    assert (tmp_path / "toy.csv").exists() and (tmp_path / "toy_predictions.csv").exists()
    r2 = loco_evaluate(DummyRegressor, X, y, groups, model_name="dummy", cycle=cycle)
    save_results([r1, r2], "toy2", tmp_path)
    per_cell = pd.read_csv(tmp_path / "toy2.csv")
    preds = pd.read_csv(tmp_path / "toy2_predictions.csv")
    assert len(per_cell) == 8 and set(per_cell.model) == {"lin", "dummy"}
    assert len(preds) == 2 * len(y) and "cycle" in preds.columns
    s = summarize([r1, r2])
    assert s.loc["lin", "rmse_mean"] < s.loc["dummy", "rmse_mean"]


def test_loco_evaluate_length_mismatch_raises():
    with pytest.raises(ValueError):
        loco_evaluate(LinearRegression, np.zeros((5, 2)), np.zeros(4), np.zeros(5))
