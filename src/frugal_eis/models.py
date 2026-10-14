"""Model factories. Each returns a fresh, unfitted estimator (pass the function itself to loco_evaluate)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Roadmap grid was logspace(-4, 3, 30); widened because cell-wise tuning chose the upper edge (1e3) in 7/8 folds.
RIDGE_ALPHAS = np.logspace(-4, 6, 41)


def mean_predictor() -> DummyRegressor:
    """Reference floor: always predicts the mean SoH of the training cells."""
    return DummyRegressor(strategy="mean")


def ohmic_linear() -> LinearRegression:
    """Physics baseline: SoH as a linear function of the ohmic resistance R0 (one feature)."""
    return LinearRegression()


def ridge_full() -> Pipeline:
    """Standardised full spectrum -> ridge regression, alpha chosen by RidgeCV's built-in LOO over rows."""
    return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS))])


class GroupRidgeCV(RegressorMixin, BaseEstimator):
    """Scaler + ridge whose alpha is chosen by leave-one-*cell*-out inside the training cells.

    RidgeCV's default leaves out single rows; neighbouring spectra of the same cell are almost identical,
    so that inner validation is leaky and favours tiny alphas. Here the inner validation mirrors the outer
    one: each training cell is held out in turn, and the alpha with the lowest mean per-cell RMSE wins.
    ``fit`` needs ``groups``; loco_evaluate passes them automatically.
    """

    def __init__(self, alphas=RIDGE_ALPHAS):
        self.alphas = alphas

    @staticmethod
    def _pipe(alpha):
        return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])

    def fit(self, X, y, groups):
        groups = np.asarray(groups)
        cv_rmse = np.zeros(len(self.alphas))
        for tr, va in LeaveOneGroupOut().split(X, y, groups):
            for j, a in enumerate(self.alphas):
                pred = self._pipe(a).fit(X[tr], y[tr]).predict(X[va])
                cv_rmse[j] += np.sqrt(np.mean((pred - y[va]) ** 2))
        self.cv_rmse_ = cv_rmse / len(np.unique(groups))
        self.alpha_ = float(self.alphas[int(np.argmin(self.cv_rmse_))])
        self.pipeline_ = self._pipe(self.alpha_).fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)


def ridge_full_groupcv() -> GroupRidgeCV:
    """Standardised full spectrum -> ridge, alpha chosen by inner leave-one-cell-out."""
    return GroupRidgeCV()


def random_forest() -> RandomForestRegressor:
    """Random forest; trees are scale-invariant, so no scaler. n_jobs=-1 only affects speed."""
    return RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)


class ARDGaussianProcess(RegressorMixin, BaseEstimator):
    """Scaler + GPR with an ARD kernel: ConstantKernel * RBF(one length scale per feature) + WhiteKernel.

    The kernel needs the number of features, which is only known at fit time, hence this wrapper.
    Hyperparameters are fitted by maximising the marginal likelihood on the training rows (as in
    Zhang et al. 2020). After fitting, ``length_scales_`` gives the ARD relevance: short = important.
    ``stride=2`` keeps every 2nd training row (rows are ordered by cell, then cycle -> every 2nd cycle).
    """

    def __init__(self, stride: int = 1, n_restarts_optimizer: int = 0, random_state: int = 0):
        self.stride = stride
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X)[:: self.stride], np.asarray(y)[:: self.stride]
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e5))
                  + WhiteKernel(1e-2, (1e-8, 1e0)))
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                       n_restarts_optimizer=self.n_restarts_optimizer,
                                       random_state=self.random_state)
        self.pipeline_ = Pipeline([("scale", StandardScaler()), ("gpr", gpr)]).fit(X, y)
        k = self.pipeline_.named_steps["gpr"].kernel_
        self.kernel_ = k
        self.length_scales_ = np.atleast_1d(k.k1.k2.length_scale)
        self.noise_level_ = float(k.k2.noise_level)
        return self

    def predict(self, X, return_std: bool = False):
        return self.pipeline_.predict(X, return_std=return_std)


def gpr_ard(stride: int = 1) -> ARDGaussianProcess:
    """GPR with ARD on the standardised input (Zhang et al. 2020 style)."""
    return ARDGaussianProcess(stride=stride)


def mlp() -> Pipeline:
    """Small neural network behind a scaler.

    Caution: early_stopping holds out a random 10 % of *rows* for validation, i.e. a row-wise split, the
    same pattern that made RidgeCV's tuning leak (NOTES, Day 2 Block 4). It cannot reach the held-out test
    cell, but it may stop training at a misleading point.
    """
    return Pipeline([("scale", StandardScaler()),
                     ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                                          max_iter=2000, random_state=0))])
