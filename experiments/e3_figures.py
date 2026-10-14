"""E3: SoH error vs estimated acquisition time. Run after experiments.e2_frequency_budget:
    python -m experiments.e3_figures

No new models: re-uses E2's per-fold results. Each fold's selected frequencies (column selected_freqs,
written to 4 significant digits) are converted to an estimated sweep time with frugal_eis.timing.sweep_time
(3 periods per frequency, no overhead: a lower bound, not a measurement).

e3_time.csv              per (model, selector, k, seed): mean and std over the 8 folds of the sweep time,
                         and of the per-cell RMSE
fig3_rmse_vs_time.png    RMSE vs estimated sweep time (log x), one panel per model, one line per selector
                         through its budgets k (labelled), random control as grey dots (one per seed and k),
                         full sweep marked with a star
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.e2_figures import MODEL_TITLES, SELECTORS
from frugal_eis.evaluate import RESULTS_DIR
from frugal_eis.timing import sweep_time

FIG_DIR = RESULTS_DIR / "figures"
N_PERIODS = 3


def add_sweep_time(res):
    res = res.copy()
    res["time_s"] = [sweep_time([float(f) for f in s.split(";")], n_periods=N_PERIODS)
                     for s in res.selected_freqs]
    return res


def summarize(res):
    keys = ["model", "selector", "k", "seed"]
    out = (res.fillna({"seed": -1}).groupby(keys)[["time_s", "rmse"]].agg(["mean", "std"]))
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out = out.reset_index()
    out["seed"] = out.seed.replace(-1, np.nan)
    return out


def fig_rmse_vs_time(t):
    models = [m for m in MODEL_TITLES if m in set(t.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.8), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        r = t[t.model == model]

        rnd = r[r.selector == "random"]
        if len(rnd):
            ax.scatter(rnd.time_s_mean, rnd.rmse_mean, s=9, color="0.6", alpha=.55, lw=0,
                       label="random (one dot per seed and k)")

        for sel, (name, colour) in SELECTORS.items():
            g = r[r.selector == sel].sort_values("k")
            ax.plot(g.time_s_mean, g.rmse_mean, color=colour, marker="o", ms=4, lw=1.4, label=name)
            for _, row in g.iterrows():
                ax.annotate(str(row.k), (row.time_s_mean, row.rmse_mean), xytext=(3, 3),
                            textcoords="offset points", fontsize=6.5, color=colour)

        full = r[r.selector == "all"].iloc[0]
        ax.scatter(full.time_s_mean, full.rmse_mean, marker="*", s=180, color="k", zorder=5,
                   label=f"full sweep, 60 freq. ({full.time_s_mean:.0f} s, RMSE {full.rmse_mean:.3f})")
        ax.axvline(full.time_s_mean, color="k", ls=":", lw=.8)

        ax.set_xscale("log")
        ax.set(xlabel=f"estimated sweep time [s]  ({N_PERIODS} periods per frequency, no overhead)",
               title=MODEL_TITLES[model])
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    axes[0][0].set_ylabel("RMSE of SoH on held-out cell\n(mean over 8 cells)")
    fig.suptitle("E3: SoH error vs estimated acquisition time (numbers = budget k), leave-one-cell-out, 25 °C",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_rmse_vs_time.png", dpi=180)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = add_sweep_time(pd.read_csv(RESULTS_DIR / "e2_budget.csv"))
    t = summarize(res)
    t.to_csv(RESULTS_DIR / "e3_time.csv", index=False)
    fig_rmse_vs_time(t)
    det = t[t.selector != "random"]
    with pd.option_context("display.width", 200):
        print("mean sweep time [s]\n", det.pivot_table(index=["model", "selector"], columns="k",
                                                        values="time_s_mean").round(1))
    print(f"saved {FIG_DIR / 'fig3_rmse_vs_time.png'} and {RESULTS_DIR / 'e3_time.csv'}")


if __name__ == "__main__":
    main()
