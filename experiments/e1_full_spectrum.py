"""E1: five models on the full spectrum, leave-one-cell-out, 25 °C, state V (plus the mean-predictor floor).

Run from the project root:  python -m experiments.e1_full_spectrum
Writes
  results/e1_full_spectrum.csv               one row per model: mean ± std over held-out cells (+ pooled)
  results/e1_full_spectrum_per_cell.csv      one row per model x held-out cell
  results/e1_full_spectrum_predictions.csv   one row per spectrum and model
All seeds are fixed inside the model factories (random_state=0).
"""

import time

import numpy as np
import pandas as pd

from frugal_eis.data import load_dataset
from frugal_eis.evaluate import RESULTS_DIR, loco_evaluate, summarize
from frugal_eis.features import full_spectrum, ohmic_feature
from frugal_eis.models import gpr_ard, mean_predictor, mlp, ohmic_linear, random_forest, ridge_full_groupcv

# GPR on all 120 ARD features did not finish one fold in 68 min (NOTES, Day 3 Block 1); see GPR_SETUP there.
GPR_STRIDE = 4          # keep every 4th training spectrum
GPR_INPUT = "neg_im"    # "neg_im" (60 features) or "full" (120)


def main():
    d = load_dataset(state="V", temps=("25",))
    X_full = full_spectrum(d.X_re, d.X_im)
    X_r0 = ohmic_feature(d.X_re, d.X_im)
    X_gpr = d.X_im if GPR_INPUT == "neg_im" else X_full
    common = dict(y=d.soh, groups=d.cell_id, cycle=d.cycle)

    runs = [
        ("mean_predictor", mean_predictor, X_full, None),
        ("ohmic_linear", ohmic_linear, X_r0, None),
        ("ridge_groupcv", ridge_full_groupcv, X_full, lambda m: {"alpha": m.alpha_}),
        ("random_forest", random_forest, X_full, None),
        (f"gpr_ard_{GPR_INPUT}_s{GPR_STRIDE}", lambda: gpr_ard(GPR_STRIDE), X_gpr,
         lambda m: {"noise": m.noise_level_, "n_ls_at_bound": int(np.sum(m.length_scales_ >= 9.9e4))}),
        ("mlp", mlp, X_full, lambda m: {"n_iter": m.named_steps["mlp"].n_iter_}),
    ]
    results = []
    for name, factory, X, inspect in runs:
        t0 = time.perf_counter()
        results.append(loco_evaluate(factory, X, model_name=name, inspect=inspect, **common))
        print(f"{name:24s} done in {time.perf_counter() - t0:7.1f} s", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    summary.to_csv(RESULTS_DIR / "e1_full_spectrum.csv")
    pd.concat([r.per_cell for r in results], ignore_index=True).to_csv(
        RESULTS_DIR / "e1_full_spectrum_per_cell.csv", index=False)
    pd.concat([r.predictions for r in results], ignore_index=True).to_csv(
        RESULTS_DIR / "e1_full_spectrum_predictions.csv", index=False)

    with pd.option_context("display.width", 170, "display.precision", 4):
        print()
        print(summary[["rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std", "r2_pooled"]])
        print()
        print(pd.concat([r.per_cell for r in results]).pivot(index="cell", columns="model", values="rmse"))


if __name__ == "__main__":
    main()
