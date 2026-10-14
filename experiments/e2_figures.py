"""Figure and summary for E2. Run after experiments.e2_frequency_budget:  python -m experiments.e2_figures

fig2_rmse_vs_budget.png  RMSE vs frequency budget k (log axis), one panel per model.
                         Selectors: mean ± std of the per-cell RMSE across the 8 held-out cells.
                         Random control (ridge only): line = median over 20 seeds of the mean per-cell RMSE,
                         band = 10th-90th percentile over the seeds (how much a random choice of k frequencies varies).
                         Dashed line: all 60 frequencies.
e2_budget_summary.csv    mean and std of the per-cell RMSE and MAE per (model, selector, k); random averaged over seeds.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frugal_eis.evaluate import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
SELECTORS = {"mrmr": ("mRMR", "#2457a6"), "rfe": ("RFE", "#c0392b"), "ard": ("ARD", "#8e44ad"),
             "logspaced": ("log-spaced", "#27864a")}
MODEL_TITLES = {"ridge": "Ridge (alpha by inner leave-one-cell-out)", "rf": "Random forest"}


def summarize(res):
    # random: average each cell's RMSE over seeds first, so std is across cells as for the other selectors
    per_cell = res.groupby(["model", "selector", "k", "cell"])[["rmse", "mae"]].mean()
    out = per_cell.groupby(["model", "selector", "k"]).agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def fig_rmse_vs_budget(res):
    models = [m for m in MODEL_TITLES if m in set(res.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.6), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        r = res[res.model == model]
        ks = sorted(r[r.selector != "all"].k.unique())

        rnd = r[r.selector == "random"]
        if len(rnd):
            per_seed = rnd.groupby(["k", "seed"]).rmse.mean().groupby("k")
            q = per_seed.quantile([0.1, 0.5, 0.9]).unstack()
            ax.fill_between(q.index, q[0.1], q[0.9], color="0.6", alpha=.3, lw=0,
                            label="random, 10–90 % over 20 seeds")
            ax.plot(q.index, q[0.5], color="0.45", lw=1.2, ls=":", label="random, median")

        for j, (sel, (name, colour)) in enumerate(SELECTORS.items()):
            g = r[r.selector == sel].groupby("k").rmse.agg(["mean", "std"]).reindex(ks)
            x = np.array(ks) * 1.04 ** (j - 1.5)          # small horizontal offsets so error bars don't overlap
            ax.errorbar(x, g["mean"], yerr=g["std"], color=colour, marker="o", ms=4, lw=1.5,
                        capsize=2.5, elinewidth=.8, label=name)

        full = r[r.selector == "all"].rmse
        ax.axhline(full.mean(), color="k", ls="--", lw=1, label=f"all 60 frequencies ({full.mean():.3f})")

        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set(xlabel="frequency budget k (number of frequencies)", title=MODEL_TITLES[model])
        ax.grid(alpha=.25)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    axes[0][0].set_ylabel("RMSE of SoH on held-out cell\n(mean ± std over 8 cells)")
    fig.suptitle("E2: SoH error vs frequency budget, leave-one-cell-out, 25 °C", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_rmse_vs_budget.png", dpi=180)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(RESULTS_DIR / "e2_budget.csv")
    summary = summarize(res)
    summary.to_csv(RESULTS_DIR / "e2_budget_summary.csv", index=False)
    fig_rmse_vs_budget(res)
    with pd.option_context("display.width", 200):
        print(summary.pivot_table(index=["model", "selector"], columns="k", values="rmse_mean").round(4))
    print(f"saved {FIG_DIR / 'fig2_rmse_vs_budget.png'} and {RESULTS_DIR / 'e2_budget_summary.csv'}")


if __name__ == "__main__":
    main()
