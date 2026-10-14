"""Where on the spectrum do the selectors pick? Run after experiments.e2_frequency_budget:
    python -m experiments.e5_selected_on_nyquist

For each selector, the k = 5 frequencies are taken from E2's per-fold selections (results/e2_budget.csv, ridge
rows; selection does not depend on the model). Selections differ between folds, so we show the 5 frequencies
chosen in the most folds (ties broken in favour of the higher frequency) and report "chosen in n/8 folds".

Regions are read off the typical spectrum itself (no circuit fit):
  inductive / ohmic       f >= the frequency where -Im(Z) crosses zero (real-axis intercept ≈ R0)
  SEI + charge transfer   between that crossing and the local minimum of -Im(Z) between arc and tail
  diffusion               below that minimum (the low-frequency, roughly 45° Warburg tail)
The SEI and charge-transfer arcs overlap in these spectra, so they are not separated.

Writes results/figures/fig5_selected_on_nyquist.png and results/e5_selected_k5.csv
(selector, freq_hz, n_folds, region).
"""

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.e2_figures import SELECTORS
from frugal_eis.data import load_dataset
from frugal_eis.evaluate import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
K = 5
TYPICAL_CELL, TYPICAL_SOH = "25C01", 0.90         # mid-life spectrum of the longest-lived 25 °C cell
MARKERS = {"mrmr": ("o", 150), "rfe": ("s", 105), "ard": ("D", 65), "logspaced": ("^", 35)}
REGION_COLOURS = {"inductive / ohmic": "#f4d03f", "SEI + charge transfer": "#85c1e9", "diffusion": "#f1948a"}


def typical_spectrum(d):
    m = np.where(d.cell_id == TYPICAL_CELL)[0]
    i = m[np.argmin(np.abs(d.soh[m] - TYPICAL_SOH))]
    return i, d.X_re[i], d.X_im[i]


def region_bounds(freqs, im):
    """(f_cross, f_min): frequency of the -Im zero crossing and of the arc/tail -Im minimum (descending freqs)."""
    k_cross = int(np.argmax(im >= 0))
    falling = np.where(np.diff(im[k_cross:]) < 0)[0]
    peak = k_cross + int(falling[0])                               # top of the arc: first local maximum
    k_min = peak + int(np.argmin(im[peak:]))                       # valley between arc and tail
    # (the tail ends higher than the arc in aged spectra, so the global maximum would be the tail's end)
    return freqs[k_cross], freqs[k_min]


def region_of(f, f_cross, f_min):
    if f >= f_cross:
        return "inductive / ohmic"
    return "SEI + charge transfer" if f > f_min else "diffusion"


def top_k_per_selector(freqs):
    r = pd.read_csv(RESULTS_DIR / "e2_budget.csv")
    r = r[(r.model == "ridge") & (r.k == K) & r.selector.isin(list(SELECTORS))]
    out = {}
    for sel, g in r.groupby("selector"):
        counts = Counter(float(f) for s in g.selected_freqs for f in s.split(";"))
        # map the 4-significant-digit strings back to grid indices
        idx_counts = Counter({int(np.argmin(np.abs(np.log(freqs) - np.log(f)))): n for f, n in counts.items()})
        top = sorted(idx_counts, key=lambda i: (-idx_counts[i], i))[:K]
        out[sel] = [(i, idx_counts[i], len(g)) for i in sorted(top)]
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = load_dataset("V", ("25",))
    i, re, im = typical_spectrum(d)
    f_cross, f_min = region_bounds(d.freqs, im)
    chosen = top_k_per_selector(d.freqs)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 6.6), gridspec_kw={"width_ratios": [1.15, 1]})
    regions = np.array([region_of(f, f_cross, f_min) for f in d.freqs])
    for name, colour in REGION_COLOURS.items():
        m = regions == name
        ax.scatter(re[m], im[m], s=14, color=colour, edgecolor="0.4", lw=.4, zorder=2, label=f"{name}")
        bx.axvspan(d.freqs[m].min() / 1.13, d.freqs[m].max() * 1.13, color=colour, alpha=.35, lw=0)
    ax.plot(re, im, color="0.55", lw=.8, zorder=1)
    bx.plot(d.freqs, im, color="0.4", lw=1, marker=".", ms=3)

    rows = []
    for sel, picks in chosen.items():
        name, colour = SELECTORS[sel]
        marker, size = MARKERS[sel]
        idx = [p[0] for p in picks]
        label = f"{name}: " + ", ".join(f"{d.freqs[j]:.3g}" for j in idx) + " Hz"
        ax.scatter(re[idx], im[idx], s=size, marker=marker, facecolor="none", edgecolor=colour, lw=1.6, zorder=3,
                   label=label)
        bx.scatter(d.freqs[idx], im[idx], s=size, marker=marker, facecolor="none", edgecolor=colour, lw=1.6,
                   zorder=3)
        for j, n, n_folds in picks:
            rows.append({"selector": sel, "freq_hz": float(d.freqs[j]), "n_folds": n, "of_folds": n_folds,
                         "region": regions[j]})

    for f, off in ((2e4, (10, -4)), (1e3, (8, -14)), (36, (0, 10)), (1.07, (0, -16)), (0.02, (-10, 4))):
        j = int(np.argmin(np.abs(np.log(d.freqs) - np.log(f))))        # a few frequency labels on the curve
        ax.annotate(f"{d.freqs[j]:.3g} Hz", (re[j], im[j]), xytext=off, textcoords="offset points",
                    fontsize=7, color="0.3", ha="left" if off[0] > 0 else ("center" if off[0] == 0 else "right"))

    ax.set_aspect("equal")
    ax.set(xlabel="Re(Z) [Ω]", ylabel="−Im(Z) [Ω]",
           title=f"Nyquist, cell {TYPICAL_CELL}, cycle {d.cycle[i]} (SoH {d.soh[i]:.2f})")
    ax.axhline(0, color="k", lw=.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              title=f"k = {K}: the 5 frequencies each selector chose most often over the 8 folds",
              title_fontsize=8)
    bx.set_xscale("log")
    bx.invert_xaxis()                                                 # high frequency on the left, as in Nyquist
    bx.axhline(0, color="k", lw=.5)
    bx.set(xlabel="frequency [Hz]", ylabel="−Im(Z) [Ω]", title="Same spectrum vs frequency, regions shaded")
    bx.grid(alpha=.25)
    for name in REGION_COLOURS:
        fs = d.freqs[regions == name]
        bx.text(np.sqrt(fs.min() * fs.max()), bx.get_ylim()[1] * .93, name, ha="center", fontsize=7.5)
    fig.suptitle("Which frequencies do the selectors pick? (25 °C, state V)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_selected_on_nyquist.png", dpi=180)
    plt.close(fig)

    t = pd.DataFrame(rows)
    t.to_csv(RESULTS_DIR / "e5_selected_k5.csv", index=False)
    print(f"regions: inductive/ohmic >= {f_cross:.4g} Hz > SEI+CT > {f_min:.4g} Hz >= diffusion")
    print(t.to_string(index=False))
    print(t.groupby(["selector", "region"]).size().unstack(fill_value=0))
    # every fold's own k = 5 set: how many picks per region, summed over the 8 folds (40 picks per selector)
    r = pd.read_csv(RESULTS_DIR / "e2_budget.csv")
    r = r[(r.model == "ridge") & (r.k == K) & r.selector.isin(list(SELECTORS))]
    per_fold = [(sel, region_of(float(f), f_cross, f_min)) for sel, s in zip(r.selector, r.selected_freqs)
                for f in s.split(";")]
    print("\nall per-fold picks (8 folds x 5):\n",
          pd.DataFrame(per_fold, columns=["selector", "region"]).value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
