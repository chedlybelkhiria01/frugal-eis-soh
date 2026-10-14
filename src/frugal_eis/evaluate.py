"""Cell-wise evaluation.

Spectra from consecutive cycles of the same cell are nearly identical, so a random row split lets a model
recognise the cell instead of learning ageing. Every split here holds out whole cells.

Typical use::

    res = loco_evaluate(lambda: make_pipeline(StandardScaler(), RidgeCV()), X, d.soh, d.cell_id,
                        model_name="ridge", cycle=d.cycle)
    res.summary()                        # mean ± std over held-out cells
    save_results([res, other], "e1_full_spectrum")   # -> results/e1_full_spectrum.csv (+ _predictions.csv)
"""

from __future__ import annotations

import inspect as inspect_mod
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from frugal_eis.data import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"


def loco_splits(groups: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Leave-one-cell-out: yields (train_idx, test_idx, held_out_cell), one fold per cell."""
    groups = np.asarray(groups)
    for train_idx, test_idx in LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups):
        yield train_idx, test_idx, str(groups[test_idx[0]])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # R² is undefined (nan) for < 2 samples
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2),
    }


@dataclass
class LOCOResult:
    """Outcome of one model under leave-one-cell-out."""

    model_name: str
    per_cell: pd.DataFrame      # one row per held-out cell: rmse, mae, r2, n_train, n_test, fit_s
    predictions: pd.DataFrame   # one row per spectrum: cell, (cycle), y_true, y_pred

    def summary(self) -> pd.Series:
        """Mean and std over held-out cells, plus metrics pooled over all predictions."""
        out = {"model": self.model_name, "n_cells": len(self.per_cell)}
        for m in ("rmse", "mae", "r2"):
            out[f"{m}_mean"] = self.per_cell[m].mean()
            out[f"{m}_std"] = self.per_cell[m].std(ddof=1)
        pooled = _metrics(self.predictions["y_true"].to_numpy(), self.predictions["y_pred"].to_numpy())
        out.update({f"{m}_pooled": v for m, v in pooled.items()})
        return pd.Series(out)


def loco_evaluate(
    model_factory: Callable[[], BaseEstimator] | BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model_name: str = "model",
    cycle: np.ndarray | None = None,
    inspect: Callable[[BaseEstimator], dict] | None = None,
    experiment: str | None = None,
    results_dir: Path | str | None = None,
) -> LOCOResult:
    """Fit on all cells but one, predict the held-out cell; repeat for every cell.

    ``model_factory`` returns a fresh, unfitted estimator (an estimator instance is cloned instead).
    Anything fitted inside it (scaler, feature selector) therefore only ever sees training cells.
    If the model's ``fit`` accepts ``groups``, it receives the training rows' cell ids (for inner
    cell-wise tuning); the held-out cell is never among them.
    ``inspect(fitted_model)`` may return extra per-fold columns (e.g. the chosen hyperparameter).
    If ``experiment`` is given, results are written to ``results/<experiment>.csv`` and
    ``results/<experiment>_predictions.csv``.
    """
    X, y, groups = np.asarray(X), np.asarray(y, dtype=float), np.asarray(groups)
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y and groups must have the same length")
    make = (lambda: clone(model_factory)) if isinstance(model_factory, BaseEstimator) else model_factory

    rows, preds = [], []
    for train_idx, test_idx, cell in loco_splits(groups):
        model = make()
        fit_kw = {"groups": groups[train_idx]} if "groups" in inspect_mod.signature(model.fit).parameters else {}
        t0 = time.perf_counter()
        model.fit(X[train_idx], y[train_idx], **fit_kw)
        fit_s = time.perf_counter() - t0
        y_pred = np.asarray(model.predict(X[test_idx]), dtype=float).ravel()

        rows.append({"model": model_name, "cell": cell, **_metrics(y[test_idx], y_pred),
                     "n_train": len(train_idx), "n_test": len(test_idx), "fit_s": fit_s,
                     **(inspect(model) if inspect is not None else {})})
        p = pd.DataFrame({"model": model_name, "cell": cell, "y_true": y[test_idx], "y_pred": y_pred})
        if cycle is not None:
            p.insert(2, "cycle", np.asarray(cycle)[test_idx])
        preds.append(p)

    result = LOCOResult(model_name, pd.DataFrame(rows), pd.concat(preds, ignore_index=True))
    if experiment is not None:
        save_results([result], experiment, results_dir)
    return result


def save_results(results: list[LOCOResult], experiment: str, results_dir: Path | str | None = None) -> Path:
    """Write per-cell metrics of one or more models to ``<experiment>.csv`` and predictions alongside."""
    out = Path(results_dir) if results_dir is not None else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{experiment}.csv"
    pd.concat([r.per_cell for r in results], ignore_index=True).to_csv(path, index=False)
    pd.concat([r.predictions for r in results], ignore_index=True).to_csv(
        out / f"{experiment}_predictions.csv", index=False)
    return path


def summarize(results: list[LOCOResult]) -> pd.DataFrame:
    """One summary row per model (see :meth:`LOCOResult.summary`)."""
    return pd.DataFrame([r.summary() for r in results]).set_index("model")
