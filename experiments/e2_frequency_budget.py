"""E2: SoH error vs frequency budget k. Leave-one-cell-out, 25 °C, state V.

Run from the project root:  python -m experiments.e2_frequency_budget
Writes results/e2_budget.csv, one row per (model, selector, k, seed, held-out cell):
  model, selector, k, seed, cell, rmse, mae, n_train, n_test, alpha (ridge only), selected_freqs (Hz, ';'-separated)

Design (see NOTES, Day 4 Block 2):
* Selection is fitted inside each fold on the training cells only (SelectedModel under loco_evaluate).
* Primary model: ridge with alpha tuned by inner leave-one-cell-out. GPR was worse than the mean
  predictor in E1, so it is not used here. Robustness check: random forest (not for the random control).
* k = "all" (60 frequencies) is identical for every selector, so it is computed once (selector "all").
* Random control: 20 seeds per k (ridge only).
"""

import time
import warnings

import pandas as pd

from frugal_eis.data import load_dataset
from frugal_eis.evaluate import RESULTS_DIR, loco_evaluate
from frugal_eis.features import full_spectrum
from frugal_eis.models import GroupRidgeCV, random_forest
from frugal_eis.selection import N_FREQ, FrequencySelector, SelectedModel

BUDGETS = [1, 2, 3, 5, 8, 12, 20]
SELECTORS = ["mrmr", "rfe", "ard", "logspaced"]
N_RANDOM_SEEDS = 20
MODELS = {"ridge": GroupRidgeCV, "rf": random_forest}
OUT = RESULTS_DIR / "e2_budget.csv"


def run_config(d, X, model_name, selector, k, seed=None):
    sel = FrequencySelector("fixed", freq_idx=list(range(N_FREQ))) if selector == "all" else \
        FrequencySelector(selector, k=k, freqs=d.freqs, seed=0 if seed is None else seed)

    def inspect(m):
        out = {"selected_freqs": ";".join(f"{d.freqs[i]:.4g}" for i in sorted(m.selector_.selected_))}
        if hasattr(m.model_, "alpha_"):
            out["alpha"] = m.model_.alpha_
        return out

    res = loco_evaluate(lambda: SelectedModel(sel, MODELS[model_name]()), X, d.soh, d.cell_id,
                        model_name=model_name, inspect=inspect)
    df = res.per_cell.drop(columns=["fit_s", "r2"])
    df.insert(1, "selector", selector)
    df.insert(2, "k", N_FREQ if selector == "all" else k)
    df.insert(3, "seed", seed)
    return df


def main():
    warnings.simplefilter("ignore")              # GPR/ARD bound warnings during ARD ranking
    d = load_dataset(state="V", temps=("25",))
    X = full_spectrum(d.X_re, d.X_im)

    configs = [(m, "all", N_FREQ, None) for m in MODELS]
    configs += [(m, s, k, None) for m in MODELS for s in SELECTORS for k in BUDGETS]
    configs += [("ridge", "random", k, seed) for k in BUDGETS for seed in range(N_RANDOM_SEEDS)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    t_start = time.perf_counter()
    for i, (model_name, selector, k, seed) in enumerate(configs, 1):
        t0 = time.perf_counter()
        df = run_config(d, X, model_name, selector, k, seed)
        df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"[{i:3d}/{len(configs)}] {model_name:5s} {selector:9s} k={k:2d} seed={seed}  "
              f"RMSE {df.rmse.mean():.4f}  ({time.perf_counter() - t0:5.1f} s, "
              f"total {(time.perf_counter() - t_start) / 60:5.1f} min)", flush=True)

    res = pd.read_csv(OUT)
    summary = (res.groupby(["model", "selector", "k"]).rmse.agg(["mean", "std"]).round(4)
               .unstack("k"))
    with pd.option_context("display.width", 200):
        print("\nMean per-cell RMSE (random: averaged over seeds and cells)\n", summary["mean"])
    print(f"\nsaved {OUT}  ({len(res)} rows)")


if __name__ == "__main__":
    main()
