"""E6: model footprint. File size and single-sample inference latency on CPU.

Run from the project root (after E2):  python -m experiments.e6_footprint
Writes results/e6_footprint.csv, one row per model:
  model, variant, n_freq, n_inputs, size_bytes, n_numbers, latency_median_us, latency_p95_us, fit_s
  (n_numbers: parameters an embedded port would store, e.g. scaler + coefficients for ridge, 3 per tree node)

* Every model is fitted once on all 25 °C cells (as deployed, not per fold), the same models as E4/E5.
* Size: bytes of joblib.dump of the whole fitted object (selector + model), compressed=0. This is the pickled
  Python object, an upper bound for an embedded port (a ridge model is n_inputs + 1 numbers after export).
* Latency: 1000 calls of predict() on one spectrum (1 x 120 array, full spectrum in, selection included),
  timed one by one with perf_counter_ns after 50 warm-up calls. Median and 95th percentile reported.
  The random forest predicts with n_jobs=1 (it is trained with n_jobs=-1; with a thread pool, one-sample
  calls took 100-190 ms in a first run, almost all of it pool overhead).
  This measures scikit-learn's per-call Python overhead on a laptop CPU, not the arithmetic a microcontroller
  would do; it is only comparable between models on this machine.
* ONNX export (optional in the plan) is skipped: onnxruntime has no wheel for Python 3.14 (NOTES, Day 6).
"""

import io
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from experiments.e1_full_spectrum import GPR_STRIDE
from experiments.e4_noise import BASE, best_small_k, neg_im_only
from frugal_eis.data import load_dataset
from frugal_eis.evaluate import RESULTS_DIR
from frugal_eis.features import full_spectrum
from frugal_eis.models import gpr_ard
from frugal_eis.selection import N_FREQ, FrequencySelector, SelectedModel

OUT = RESULTS_DIR / "e6_footprint.csv"
N_CALLS, N_WARMUP = 1000, 50


def variants(freqs):
    out = []
    for model, make in BASE.items():
        out.append((f"{model}_full", "full", N_FREQ,
                    lambda make=make: SelectedModel(FrequencySelector("fixed", freq_idx=list(range(N_FREQ))),
                                                    make())))
        sel, k = best_small_k(model)
        out.append((f"{model}_{sel}_k{k}", "small", k,
                    lambda make=make, sel=sel, k=k: SelectedModel(FrequencySelector(sel, k=k, freqs=freqs),
                                                                  make())))
    out.append(("gpr_full", "full", N_FREQ,
                lambda: make_pipeline(FunctionTransformer(neg_im_only), gpr_ard(GPR_STRIDE))))
    return out


def size_bytes(model):
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=0)
    return buf.getbuffer().nbytes


def n_numbers(model):
    """Numbers an embedded port would have to store (no Python object overhead)."""
    if isinstance(model, SelectedModel):
        inner = model.model_
        if hasattr(inner, "pipeline_"):                       # GroupRidgeCV: StandardScaler + Ridge
            scaler, ridge = inner.pipeline_[0], inner.pipeline_[-1]
            return scaler.mean_.size + scaler.scale_.size + ridge.coef_.size + 1
        if hasattr(inner, "estimators_"):                     # random forest: per node feature, threshold, value
            return sum(3 * t.tree_.node_count for t in inner.estimators_)
    ard = model[-1]                                           # GPR: scaler, training inputs, weights, length scales
    scaler, gpr = ard.pipeline_["scale"], ard.pipeline_["gpr"]
    return int(scaler.mean_.size + scaler.scale_.size + gpr.X_train_.size + gpr.alpha_.size
               + ard.length_scales_.size)


def latency_us(model, X, rng):
    rows = rng.integers(0, len(X), N_WARMUP + N_CALLS)
    times = []
    for j, r in enumerate(rows):
        x = X[r:r + 1]
        t0 = time.perf_counter_ns()
        model.predict(x)
        if j >= N_WARMUP:
            times.append((time.perf_counter_ns() - t0) / 1e3)
    return float(np.median(times)), float(np.percentile(times, 95))


def main():
    warnings.simplefilter("ignore")
    d = load_dataset(state="V", temps=("25",))
    X, y = full_spectrum(d.X_re, d.X_im), d.soh
    rows = []
    for name, variant, n_freq, make in variants(d.freqs):
        m = make()
        t0 = time.perf_counter()
        if isinstance(m, SelectedModel):
            m.fit(X, y, groups=d.cell_id)
        else:
            m.fit(X, y)
        fit_s = time.perf_counter() - t0
        inner = getattr(m, "model_", None)
        if hasattr(inner, "n_jobs"):
            inner.n_jobs = 1        # single-threaded inference: a thread pool per 1-sample call costs ~100 ms
        med, p95 = latency_us(m, X, np.random.default_rng(0))
        n_inputs = N_FREQ if name == "gpr_full" else 2 * n_freq
        rows.append({"model": name, "variant": variant, "n_freq": n_freq, "n_inputs": n_inputs,
                     "size_bytes": size_bytes(m), "n_numbers": n_numbers(m), "latency_median_us": med, "latency_p95_us": p95,
                     "fit_s": fit_s})
        print(f"{name:18s} size {rows[-1]['size_bytes'] / 1024:9.1f} KiB  numbers {rows[-1]['n_numbers']:9d}  median {med:8.1f} us  p95 {p95:8.1f} us",
              flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    print(f"\nCPU: {platform.processor()} | Python {platform.python_version()}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
