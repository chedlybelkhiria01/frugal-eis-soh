"""E4: robustness to measurement noise. Leave-one-cell-out, 25 °C, state V.

Run from the project root (after E2):  python -m experiments.e4_noise
Writes results/e4_noise.csv, one row per (model, held-out cell, sigma, seed):
  model, variant, selector, k, cell, sigma, seed, rmse, mae

Design (see NOTES, Day 5 Block 2):
* Models are fitted once per fold on **clean** training spectra; noise is added to the **test spectra only**:
  Z_noisy = Z * (1 + sigma * N(0, 1)), on Re and Im independently (features.multiplicative_noise).
* For each (fold, sigma, seed) one noisy copy of the test spectra is drawn and given to every model,
  so the models are compared on identical noise.
* Compared: full spectrum vs the best small-k subset from E2 (lowest mean RMSE with k <= 5, excluding the
  random control) for ridge and the random forest; plus the full-spectrum GPR of E1 (-Im only, every 4th
  training spectrum), as the plan asks. Frequency selection is refitted per fold on the clean training cells.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from experiments.e1_full_spectrum import GPR_STRIDE
from frugal_eis.data import load_dataset
from frugal_eis.evaluate import RESULTS_DIR, loco_splits
from frugal_eis.features import full_spectrum, multiplicative_noise
from frugal_eis.models import GroupRidgeCV, gpr_ard, random_forest
from frugal_eis.selection import N_FREQ, FrequencySelector, SelectedModel

SIGMAS = [0, 0.005, 0.01, 0.02, 0.05]
N_SEEDS = 10
MAX_SMALL_K = 5
OUT = RESULTS_DIR / "e4_noise.csv"
BASE = {"ridge": GroupRidgeCV, "rf": random_forest}


def neg_im_only(X):
    return X[:, N_FREQ:]


def best_small_k(model):
    """(selector, k) with the lowest mean RMSE in E2 among k <= MAX_SMALL_K, random control excluded."""
    s = pd.read_csv(RESULTS_DIR / "e2_budget_summary.csv")
    s = s[(s.model == model) & (s.k <= MAX_SMALL_K) & ~s.selector.isin(["random", "all"])]
    best = s.loc[s.rmse_mean.idxmin()]
    return best.selector, int(best.k)


def build_variants(freqs):
    variants = []
    for model, make in BASE.items():
        variants.append((f"{model}_full", "full", "all", N_FREQ,
                         lambda make=make: SelectedModel(FrequencySelector("fixed", freq_idx=list(range(N_FREQ))),
                                                         make())))
        sel, k = best_small_k(model)
        variants.append((f"{model}_{sel}_k{k}", "small", sel, k,
                         lambda make=make, sel=sel, k=k: SelectedModel(FrequencySelector(sel, k=k, freqs=freqs),
                                                                       make())))
    variants.append(("gpr_full", "full", "all (-Im only)", N_FREQ,
                     lambda: make_pipeline(FunctionTransformer(neg_im_only), gpr_ard(GPR_STRIDE))))
    return variants


def main():
    warnings.simplefilter("ignore")
    d = load_dataset(state="V", temps=("25",))
    X, y = full_spectrum(d.X_re, d.X_im), d.soh
    variants = build_variants(d.freqs)
    for name, *_ in variants:
        print("variant:", name)

    rows, t_start = [], time.perf_counter()
    for fold, (tr, te, cell) in enumerate(loco_splits(d.cell_id)):
        fitted = []
        for name, variant, selector, k, make in variants:
            m = make()
            if isinstance(m, SelectedModel):
                m.fit(X[tr], y[tr], groups=d.cell_id[tr])
            else:
                m.fit(X[tr], y[tr])
            fitted.append((name, variant, selector, k, m))
        for sigma in SIGMAS:
            for seed in range(N_SEEDS):
                X_te = multiplicative_noise(X[te], sigma, np.random.default_rng([seed, fold]))
                for name, variant, selector, k, m in fitted:
                    err = m.predict(X_te) - y[te]
                    rows.append({"model": name, "variant": variant, "selector": selector, "k": k, "cell": cell,
                                 "sigma": sigma, "seed": seed, "rmse": float(np.sqrt(np.mean(err ** 2))),
                                 "mae": float(np.mean(np.abs(err)))})
        print(f"fold {fold + 1}/8 ({cell}) done, total {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    table = res.groupby(["model", "sigma"]).rmse.mean().unstack("sigma").round(4)
    with pd.option_context("display.width", 200):
        print("\nMean per-cell RMSE (over cells and seeds)\n", table)
    print(f"\nsaved {OUT}  ({len(res)} rows)")


if __name__ == "__main__":
    main()
