"""E1 (partial): physics baseline and ridge on the full spectrum, leave-one-cell-out, 25 °C, state V.

Run from the project root:  python -m experiments.e1_partial
Writes results/e1_partial.csv (per held-out cell) and results/e1_partial_predictions.csv.
"""

import numpy as np
import pandas as pd

from frugal_eis.data import load_dataset
from frugal_eis.evaluate import loco_evaluate, save_results, summarize
from frugal_eis.features import full_spectrum, ohmic_feature
from frugal_eis.models import RIDGE_ALPHAS, mean_predictor, ohmic_linear, ridge_full, ridge_full_groupcv


def main():
    d = load_dataset(state="V", temps=("25",))
    X_full = full_spectrum(d.X_re, d.X_im)
    X_r0 = ohmic_feature(d.X_re, d.X_im)
    common = dict(y=d.soh, groups=d.cell_id, cycle=d.cycle)

    results = [
        loco_evaluate(mean_predictor, X_full, model_name="mean_predictor", **common),
        loco_evaluate(ohmic_linear, X_r0, model_name="ohmic_linear", **common,
                      inspect=lambda m: {"slope": m.coef_[0], "intercept": m.intercept_}),
        loco_evaluate(ridge_full, X_full, model_name="ridge_full", **common,
                      inspect=lambda m: {"alpha": m.named_steps["ridge"].alpha_}),
        # Added after the run above showed tiny alphas: tune alpha by inner leave-one-cell-out instead
        loco_evaluate(ridge_full_groupcv, X_full, model_name="ridge_full_groupcv", **common,
                      inspect=lambda m: {"alpha": m.alpha_}),
    ]
    path = save_results(results, "e1_partial")

    with pd.option_context("display.width", 160, "display.precision", 4):
        print(f"{len(d)} spectra, {len(np.unique(d.cell_id))} cells, {X_full.shape[1]} spectral features\n")
        print(summarize(results)[["rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_pooled"]], "\n")
        print(pd.concat([r.per_cell for r in results]).pivot(index="cell", columns="model", values="rmse"), "\n")
        print(f"alpha grid {RIDGE_ALPHAS[0]:g}..{RIDGE_ALPHAS[-1]:g}")
        for r in results[2:]:
            print(f"{r.model_name} alpha per fold:", r.per_cell["alpha"].round(5).tolist())
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
