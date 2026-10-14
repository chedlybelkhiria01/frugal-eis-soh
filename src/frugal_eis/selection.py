"""Frequency selection: which k of the 60 measured frequencies to keep.

The unit of selection is a **frequency**: keeping frequency i keeps both of its columns, Re(Z) and -Im(Z).
Inputs ``X`` are full spectra laid out as by :func:`frugal_eis.features.full_spectrum`:
columns [Re(f_0..f_{n-1}), -Im(f_0..f_{n-1})], frequencies descending.

Data-driven methods produce a full **ranking** of the frequencies; ``select_*(X, y, k)`` returns its first k.
For greedy mRMR and step-1 RFE the top-k of the ranking equals what the method would pick when asked for
k directly, and computing the ranking once lets E2 reuse it for every budget k.

All selectors must be fitted on training cells only. :class:`FrequencySelector` wraps them as a
scikit-learn transformer so that, inside a Pipeline evaluated with ``loco_evaluate``, this is automatic.
"""

from __future__ import annotations

from inspect import signature

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from frugal_eis.models import ARDGaussianProcess

N_FREQ = 60


# --- helpers ------------------------------------------------------------------------------------

def columns_for(freq_idx, n_freq: int = N_FREQ) -> np.ndarray:
    """Column indices (Re and -Im) of the given frequency indices, in a full-spectrum matrix."""
    freq_idx = np.asarray(freq_idx, dtype=int)
    return np.concatenate([freq_idx, freq_idx + n_freq])


def _columns_to_frequency_order(column_order, n_freq: int) -> np.ndarray:
    """Frequency ranking from a column ranking: a frequency enters when its first column does."""
    seen, order = set(), []
    for c in column_order:
        f = int(c) % n_freq
        if f not in seen:
            seen.add(f)
            order.append(f)
    return np.array(order)


def _abs_corr_with(X, v):
    Xc, vc = X - X.mean(0), v - v.mean()
    denom = np.sqrt((Xc ** 2).sum(0) * (vc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(Xc.T @ vc) / denom
    return np.nan_to_num(r)          # constant column -> 0


# --- rankings -----------------------------------------------------------------------------------

def rank_mrmr(X, y, n_freq: int = N_FREQ) -> np.ndarray:
    """Greedy mRMR on columns: score = |corr(x, y)| - mean |corr(x, already chosen)| (difference form)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    relevance = _abs_corr_with(X, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
    chosen = [int(np.argmax(relevance))]
    remaining = set(range(X.shape[1])) - set(chosen)
    redundancy_sum = cc[:, chosen[0]].copy()
    while remaining:
        cand = np.array(sorted(remaining))
        score = relevance[cand] - redundancy_sum[cand] / len(chosen)
        best = int(cand[np.argmax(score)])
        chosen.append(best)
        remaining.remove(best)
        redundancy_sum += cc[:, best]
    return _columns_to_frequency_order(chosen, n_freq)


def rank_rfe(X, y, n_freq: int = N_FREQ, alpha: float = 1.0) -> np.ndarray:
    """Recursive feature elimination with ridge on standardised columns (step 1, down to one column)."""
    Xs = StandardScaler().fit_transform(np.asarray(X, float))
    rfe = RFE(Ridge(alpha=alpha), n_features_to_select=1, step=1).fit(Xs, y)
    return _columns_to_frequency_order(np.argsort(rfe.ranking_, kind="stable"), n_freq)


def rank_ard(X, y, n_freq: int = N_FREQ, stride: int = 4) -> np.ndarray:
    """GPR with ARD on the -Im columns (60 features, every ``stride``-th row); short length scale = relevant.

    Frequencies whose length scales are tied (typically all pushed to the upper bound = "switched off")
    are ordered by |corr(-Im, y)|.
    """
    X_im = np.asarray(X, float)[:, n_freq:2 * n_freq]
    ls = ARDGaussianProcess(stride=stride).fit(X_im, y).length_scales_
    return np.lexsort((-_abs_corr_with(X_im, np.asarray(y, float)), np.round(np.log10(ls), 6)))


# --- selectors (the names used in the roadmap) ----------------------------------------------------

def select_mrmr(X, y, k: int, n_freq: int = N_FREQ) -> np.ndarray:
    return rank_mrmr(X, y, n_freq)[:k]


def select_rfe(X, y, k: int, n_freq: int = N_FREQ) -> np.ndarray:
    return rank_rfe(X, y, n_freq)[:k]


def select_ard(X, y, k: int, n_freq: int = N_FREQ) -> np.ndarray:
    return rank_ard(X, y, n_freq)[:k]


def select_logspaced(freqs, k: int) -> np.ndarray:
    """k frequencies evenly spaced on a log scale (control). The grid itself is log-uniform, so this is
    evenly spaced indices; k = 1 gives the geometric middle of the band."""
    n = len(freqs)
    if not 1 <= k <= n:
        raise ValueError(f"k must be in 1..{n}")
    if k == 1:
        return np.array([int(np.argmin(np.abs(np.log(freqs) - np.log(freqs).mean())))])
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))


def select_random(k: int, seed: int, n_freq: int = N_FREQ) -> np.ndarray:
    """k distinct random frequencies (control). E2 averages the result over 20 seeds."""
    return np.sort(np.random.default_rng(seed).choice(n_freq, size=k, replace=False))


# --- scikit-learn wrapper ----------------------------------------------------------------------

_RANKERS = {"mrmr": rank_mrmr, "rfe": rank_rfe, "ard": rank_ard}
_rank_cache: dict = {}


class FrequencySelector(TransformerMixin, BaseEstimator):
    """Keeps the Re and -Im columns of k selected frequencies; selection is learned in ``fit``.

    method: "mrmr" | "rfe" | "ard" (data-driven, fitted on the training rows) or
            "logspaced" | "random" | "fixed" (controls; ``fixed`` uses ``freq_idx``).
    Rankings are cached per training set, so evaluating many k on the same fold ranks only once.
    """

    def __init__(self, method: str = "mrmr", k: int = 5, freqs=None, seed: int = 0, freq_idx=None,
                 n_freq: int = N_FREQ):
        self.method = method
        self.k = k
        self.freqs = freqs
        self.seed = seed
        self.freq_idx = freq_idx
        self.n_freq = n_freq

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        if X.shape[1] != 2 * self.n_freq:
            raise ValueError(f"expected {2 * self.n_freq} columns [Re, -Im], got {X.shape[1]}")
        if self.method in _RANKERS:
            key = (self.method, X.shape, hash(X.tobytes()), hash(np.asarray(y, float).tobytes()))
            if key not in _rank_cache:
                _rank_cache[key] = _RANKERS[self.method](X, y, self.n_freq)
            self.selected_ = _rank_cache[key][: self.k]
        elif self.method == "logspaced":
            freqs = self.freqs if self.freqs is not None else np.arange(self.n_freq, 0, -1)
            self.selected_ = select_logspaced(freqs, self.k)
        elif self.method == "random":
            self.selected_ = select_random(self.k, self.seed, self.n_freq)
        elif self.method == "fixed":
            self.selected_ = np.asarray(self.freq_idx, dtype=int)
        else:
            raise ValueError(f"unknown method {self.method!r}")
        self.columns_ = columns_for(self.selected_, self.n_freq)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.columns_]


def clear_rank_cache() -> None:
    _rank_cache.clear()


class SelectedModel(RegressorMixin, BaseEstimator):
    """Frequency selection followed by a model, fitted together on the training rows.

    Unlike a Pipeline, ``fit`` accepts ``groups`` and forwards them to the model when it takes them
    (e.g. GroupRidgeCV's inner leave-one-cell-out), so loco_evaluate's cell ids reach the tuning step.
    """

    def __init__(self, selector: FrequencySelector, model):
        self.selector = selector
        self.model = model

    def fit(self, X, y, groups=None):
        self.selector_ = clone(self.selector).fit(X, y)
        self.model_ = clone(self.model)
        kw = {"groups": groups} if groups is not None and "groups" in signature(self.model_.fit).parameters else {}
        self.model_.fit(self.selector_.transform(X), y, **kw)
        return self

    def predict(self, X):
        return self.model_.predict(self.selector_.transform(X))
