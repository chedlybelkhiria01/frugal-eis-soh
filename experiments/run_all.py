"""Regenerate every CSV and figure in results/ from the raw data, in dependency order.

Run from the project root:  python -m experiments.run_all            (all steps, about 1 hour on a laptop CPU)
                            python -m experiments.run_all e3 e4      (only steps whose name starts with these)

Needs the raw Zenodo data in data/raw/ (see data/README.md). Every random choice has a fixed seed
(random_state=0 in the model factories, explicit seeds for the random control and the noise), so re-running
reproduces the committed CSVs exactly, except columns that are wall-clock timings (fit_s, latency_*).
"""

import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [                                 # (name, module or notebook, approximate minutes)
    ("e1_partial", "experiments.e1_partial", 1),
    ("e1_full_spectrum", "experiments.e1_full_spectrum", 5),
    ("e1_figures", "experiments.e1_figures", 1),
    ("eda_notebook", "notebooks/01_eda.ipynb", 1),
    ("e2_frequency_budget", "experiments.e2_frequency_budget", 40),
    ("e2_figures", "experiments.e2_figures", 0),
    ("e3_figures", "experiments.e3_figures", 0),
    ("e4_noise", "experiments.e4_noise", 7),
    ("e4_figures", "experiments.e4_figures", 0),
    ("e5_selected_on_nyquist", "experiments.e5_selected_on_nyquist", 0),
    ("e5_temperature", "experiments.e5_temperature", 1),
    ("e6_footprint", "experiments.e6_footprint", 2),
]


def run_notebook(rel_path):
    """Execute the notebook in its own folder (it uses ../data and ../results). The .ipynb is not modified."""
    import nbformat
    from nbclient import NotebookClient

    path = ROOT / rel_path
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()


def main(prefixes):
    steps = [s for s in STEPS if not prefixes or any(s[0].startswith(p) for p in prefixes)]
    if not steps:
        sys.exit(f"no step matches {prefixes}; steps: {', '.join(s[0] for s in STEPS)}")
    print(f"{len(steps)} steps, about {sum(s[2] for s in steps)} min", flush=True)
    t_all = time.perf_counter()
    for i, (name, target, _) in enumerate(steps, 1):
        print(f"\n=== [{i}/{len(steps)}] {name} ===", flush=True)
        t0 = time.perf_counter()
        if target.endswith(".ipynb"):
            run_notebook(target)
        else:
            importlib.import_module(target).main()
        print(f"=== {name} done in {time.perf_counter() - t0:.0f} s ===", flush=True)
    print(f"\nall done in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
