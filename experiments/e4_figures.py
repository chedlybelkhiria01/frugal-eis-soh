"""Figure for E4. Run after experiments.e4_noise:  python -m experiments.e4_figures

fig4_noise_robustness.png  RMSE vs noise level sigma, one line per model (solid = full spectrum,
                           dashed = best small-k subset from E2). Line: mean over the 8 held-out cells and
                           10 noise seeds. Band: min-max over seeds of the cell-averaged RMSE.
e4_noise_summary.csv       mean, std over cells (seed-averaged per cell), and seed min/max, per model and sigma
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from frugal_eis.evaluate import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
COLOURS = {"ridge": "#2457a6", "rf": "#27864a", "gpr": "#8e44ad"}


def summarize(res):
    per_cell = res.groupby(["model", "sigma", "cell"]).rmse.mean()                  # average over seeds
    per_seed = res.groupby(["model", "sigma", "seed"]).rmse.mean()                  # average over cells
    out = pd.DataFrame({
        "rmse_mean": per_cell.groupby(["model", "sigma"]).mean(),
        "rmse_std_cells": per_cell.groupby(["model", "sigma"]).std(),
        "rmse_seed_min": per_seed.groupby(["model", "sigma"]).min(),
        "rmse_seed_max": per_seed.groupby(["model", "sigma"]).max(),
    }).reset_index()
    meta = res.drop_duplicates("model").set_index("model")[["variant", "selector", "k"]]
    return out.join(meta, on="model")


def fig_noise(s):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sigmas = sorted(s.sigma.unique())
    x_of = {sg: i for i, sg in enumerate(sigmas)}            # evenly spaced categories (0 cannot go on a log axis)
    for model, g in s.groupby("model", sort=False):
        g = g.sort_values("sigma")
        base = model.split("_")[0]
        small = g.variant.iloc[0] == "small"
        x = g.sigma.map(x_of)
        label = f"{model} ({g.k.iloc[0]} freq.)"
        ax.plot(x, g.rmse_mean, color=COLOURS[base], ls="--" if small else "-", marker="s" if small else "o",
                ms=4, lw=1.6, label=label)
        ax.fill_between(x, g.rmse_seed_min, g.rmse_seed_max, color=COLOURS[base], alpha=.12, lw=0)
    ax.set_xticks(range(len(sigmas)), [f"{sg:g}\n({sg * 100:g} %)" for sg in sigmas])
    ax.set(xlabel="relative noise σ on test spectra (Re and Im independently)",
           ylabel="RMSE of SoH on held-out cell\n(mean over 8 cells, 10 seeds)",
           title="E4: robustness to measurement noise (models trained on clean spectra)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_noise_robustness.png", dpi=180)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(RESULTS_DIR / "e4_noise.csv")
    s = summarize(res)
    s.to_csv(RESULTS_DIR / "e4_noise_summary.csv", index=False)
    fig_noise(s)
    with pd.option_context("display.width", 200):
        print(s.pivot(index="model", columns="sigma", values="rmse_mean").round(4))
    print(f"saved {FIG_DIR / 'fig4_noise_robustness.png'} and {RESULTS_DIR / 'e4_noise_summary.csv'}")


if __name__ == "__main__":
    main()
