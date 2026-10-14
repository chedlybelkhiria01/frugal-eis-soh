"""Figures for E1. Run after experiments.e1_full_spectrum:  python -m experiments.e1_figures

fig1_pred_vs_true.png      predicted vs true SoH of the best model (lowest mean per-cell RMSE), one colour per held-out cell
fig1b_gpr_uncertainty.png  SoH vs cycle for one held-out cell: truth, GPR mean and ±2σ band
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.e1_full_spectrum import GPR_INPUT, GPR_STRIDE
from frugal_eis.data import load_dataset
from frugal_eis.evaluate import RESULTS_DIR
from frugal_eis.features import full_spectrum
from frugal_eis.models import gpr_ard

FIG_DIR = RESULTS_DIR / "figures"
UNCERTAINTY_CELL = "25C01"   # longest-lived 25 °C cell: 261 spectra from SoH 1.0 to 0.37


def fig_pred_vs_true():
    summary = pd.read_csv(RESULTS_DIR / "e1_full_spectrum.csv", index_col="model")
    best = summary["rmse_mean"].idxmin()
    preds = pd.read_csv(RESULTS_DIR / "e1_full_spectrum_predictions.csv")
    p = preds[preds.model == best]

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    cmap = plt.get_cmap("tab10")
    for k, (cell, g) in enumerate(p.groupby("cell")):
        rmse = np.sqrt(np.mean((g.y_pred - g.y_true) ** 2))
        ax.scatter(g.y_true, g.y_pred, s=7, color=cmap(k), alpha=.8, label=f"{cell} (RMSE {rmse:.3f})")
    lo = min(p.y_true.min(), p.y_pred.min()) - 0.03
    hi = max(p.y_true.max(), p.y_pred.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="true SoH", ylabel="predicted SoH",
           title=f"{best}: leave-one-cell-out predictions\nmean per-cell RMSE {summary.loc[best, 'rmse_mean']:.3f}"
                 f" ± {summary.loc[best, 'rmse_std']:.3f}")
    ax.set_aspect("equal"); ax.grid(alpha=.25)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", title="held-out cell", title_fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_pred_vs_true.png", dpi=180); plt.close(fig)
    return best


def fig_gpr_uncertainty():
    d = load_dataset("V", ("25",))
    X = d.X_im if GPR_INPUT == "neg_im" else full_spectrum(d.X_re, d.X_im)
    tr, te = d.cell_id != UNCERTAINTY_CELL, d.cell_id == UNCERTAINTY_CELL
    model = gpr_ard(GPR_STRIDE).fit(X[tr], d.soh[tr])          # same setup as the E1 fold for this cell
    mean, std = model.predict(X[te], return_std=True)
    cyc, truth = d.cycle[te], d.soh[te]
    inside = np.mean(np.abs(truth - mean) <= 2 * std)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.fill_between(cyc, mean - 2 * std, mean + 2 * std, color="#2457a6", alpha=.18, label="GPR ±2σ")
    ax.plot(cyc, mean, color="#2457a6", lw=1.6, label="GPR mean")
    ax.plot(cyc, truth, "k.", ms=3, label="measured SoH")
    ax.set(xlabel="cycle", ylabel="SoH",
           title=f"Held-out cell {UNCERTAINTY_CELL}: GPR prediction with uncertainty "
                 f"({inside:.0%} of points inside ±2σ)")
    ax.grid(alpha=.25); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig1b_gpr_uncertainty.png", dpi=180); plt.close(fig)
    return inside, float(np.sqrt(np.mean((mean - truth) ** 2)))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    best = fig_pred_vs_true()
    inside, rmse = fig_gpr_uncertainty()
    print(f"fig1: best model = {best}")
    print(f"fig1b: {UNCERTAINTY_CELL} GPR RMSE {rmse:.4f}, {inside:.1%} of true values inside ±2σ")


if __name__ == "__main__":
    main()
