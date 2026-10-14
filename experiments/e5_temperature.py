"""E5: transfer across temperature. Train on all 8 cells at 25 °C, test on each 35 °C and 45 °C cell (state V).

Run from the project root (after E2):  python -m experiments.e5_temperature
Writes results/e5_temperature.csv, one row per (model, test cell):
  model, variant, selector, k, test_cell, test_temp, n_test, rmse, mae, bias, selected_freqs, alpha

* No temperature information is given to the models and nothing is adapted: a pure out-of-distribution test.
  (Zhang et al. trained with one 35 °C and one 45 °C cell; this setup does not.)
* Full spectrum vs the best small-k subset from E2 (same rule as E4: lowest mean E2 RMSE with k <= 5, random
  excluded), for ridge (alpha by leave-one-cell-out over the 25 °C cells) and the random forest. The selection
  is fitted on all 25 °C cells. Mean predictor (25 °C training mean) as the floor.
* bias = mean(prediction - true): positive means the model thinks the cell is healthier than it is.
"""

import warnings

import numpy as np
import pandas as pd

from experiments.e4_noise import BASE, best_small_k
from frugal_eis.data import load_dataset
from frugal_eis.evaluate import RESULTS_DIR
from frugal_eis.features import full_spectrum
from frugal_eis.models import mean_predictor
from frugal_eis.selection import N_FREQ, FrequencySelector, SelectedModel

OUT = RESULTS_DIR / "e5_temperature.csv"


def variants(freqs):
    out = [("mean_predictor", "floor", "-", 0, mean_predictor)]
    for model, make in BASE.items():
        out.append((f"{model}_full", "full", "all", N_FREQ,
                    lambda make=make: SelectedModel(FrequencySelector("fixed", freq_idx=list(range(N_FREQ))),
                                                    make())))
        sel, k = best_small_k(model)
        out.append((f"{model}_{sel}_k{k}", "small", sel, k,
                    lambda make=make, sel=sel, k=k: SelectedModel(FrequencySelector(sel, k=k, freqs=freqs),
                                                                  make())))
    return out


def main():
    warnings.simplefilter("ignore")
    d = load_dataset(state="V", temps=("25", "35", "45"))
    X, y = full_spectrum(d.X_re, d.X_im), d.soh
    train = d.temp == 25
    rows = []
    for name, variant, selector, k, make in variants(d.freqs):
        m = make()
        if isinstance(m, SelectedModel):
            m.fit(X[train], y[train], groups=d.cell_id[train])
            freqs = ";".join(f"{d.freqs[i]:.4g}" for i in sorted(m.selector_.selected_))
            alpha = getattr(m.model_, "alpha_", np.nan)
        else:
            m.fit(X[train], y[train])
            freqs, alpha = "", np.nan
        for cell in sorted(set(d.cell_id[~train])):
            te = d.cell_id == cell
            err = m.predict(X[te]) - y[te]
            rows.append({"model": name, "variant": variant, "selector": selector, "k": k, "test_cell": cell,
                         "test_temp": int(d.temp[te][0]), "n_test": int(te.sum()),
                         "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
                         "bias": float(np.mean(err)), "selected_freqs": freqs, "alpha": alpha})
        print(f"{name:20s} done", flush=True)

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    with pd.option_context("display.width", 200):
        print(res.pivot(index="model", columns="test_cell", values="rmse").round(3))
        print(res.pivot(index="model", columns="test_cell", values="bias").round(3))
    print(f"\nsaved {OUT}  ({len(res)} rows)")


if __name__ == "__main__":
    main()
