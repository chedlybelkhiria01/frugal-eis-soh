"""Loading the Zhang et al. (2020) EIS + capacity dataset.

Data: Zhang et al., Nat. Commun. 11:1706 (2020), https://zenodo.org/records/3633835 (CC BY 4.0).
Unzip both archives anywhere under ``data/raw/``; the loader finds the files itself.

Conventions
-----------
* One sample = one impedance spectrum (60 frequencies, 20 kHz -> 0.02 Hz) of one cell at one cycle.
* ``X_im`` holds **-Im(Z)**, exactly as stored in the raw files (positive above the real axis on a
  Nyquist plot; negative at the highest frequencies, where the cell behaves inductively).
* Label pairing: EIS cycle ``n`` <-> discharge capacity of capacity-file cycle ``n - 1``.
  This reproduces the preprocessed tutorial CSV exactly (see notebooks/01_eda.ipynb).
* SoH = capacity / capacity of the first cycle *of the same cell*. A relative definition: cells
  start at slightly different capacities (34-42 mAh), so this measures each cell's own fade and
  makes the first spectrum of every cell SoH = 1.

Raw-file quirks handled here (documented in NOTES.md)
-----------------------------------------------------
1. 35 °C and 45 °C EIS files have no header row; 25 °C files do.
2. Capacity headers use spaces instead of a tab between two column names.
3. Capacity files have 4 or 6 columns; capacity is always the last one.
4. The last capacity cycle can lack a discharge value (NaN) -> dropped.
5. ``25C42`` (a single state-VI file) is not one of the paper's 12 cells -> never loaded.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CELLS = [f"25C0{i}" for i in range(1, 9)] + ["35C01", "35C02", "45C01", "45C02"]
STATES = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
N_FREQ = 60

EIS_COLS = ["time_s", "cycle", "freq_hz", "re_ohm", "negim_ohm", "absz_ohm", "phase_deg"]


@dataclass
class EISDataset:
    """Impedance spectra with SoH labels. Row i of every array describes the same spectrum."""

    X_re: np.ndarray      # (n_spectra, n_freq)  Re(Z) [Ohm]
    X_im: np.ndarray      # (n_spectra, n_freq)  -Im(Z) [Ohm]
    freqs: np.ndarray     # (n_freq,)            [Hz], descending
    soh: np.ndarray       # (n_spectra,)         capacity / first-cycle capacity of that cell
    capacity: np.ndarray  # (n_spectra,)         discharge capacity [mAh]
    cell_id: np.ndarray   # (n_spectra,)         e.g. "25C01"  -> use as the group for splitting
    cycle: np.ndarray     # (n_spectra,)         EIS cycle number
    temp: np.ndarray      # (n_spectra,)         cycling temperature [°C]

    def __len__(self) -> int:
        return len(self.soh)

    def subset(self, mask: np.ndarray) -> "EISDataset":
        """Rows where ``mask`` is True (boolean array or index array)."""
        return EISDataset(self.X_re[mask], self.X_im[mask], self.freqs, self.soh[mask],
                          self.capacity[mask], self.cell_id[mask], self.cycle[mask], self.temp[mask])


def _find_dir(raw_dir: Path, marker: str) -> Path:
    """Folder containing ``marker``; the zips extract into nested folders and add __MACOSX junk."""
    for p in raw_dir.rglob(marker):
        if "__MACOSX" not in p.parts:
            return p.parent
    raise FileNotFoundError(f"{marker} not found under {raw_dir}. See data/README.md for download steps.")


def read_eis(path: Path) -> pd.DataFrame:
    """One raw EIS file (all spectra of one cell at one state), with or without a header row."""
    first_token = Path(path).read_text().split("\n", 1)[0].split("\t")[0].strip()
    try:
        float(first_token)
        has_header = False
    except ValueError:
        has_header = True
    df = pd.read_csv(path, sep="\t", header=None, skiprows=1 if has_header else 0, usecols=range(7))
    df.columns = EIS_COLS
    return df


def read_capacity(path: Path) -> pd.Series:
    """Discharge capacity [mAh] per capacity-file cycle, indexed by cycle number."""
    df = pd.read_csv(path, sep="\t", skiprows=1, header=None).dropna(axis=1, how="all")
    df = pd.DataFrame({"cycle": df.iloc[:, 1], "ox_red": df.iloc[:, 2], "capacity": df.iloc[:, -1]})
    per_cycle = df.groupby(["cycle", "ox_red"])["capacity"].max().unstack()
    return per_cycle[0].dropna().rename("discharge_mAh")  # ox_red == 0 -> discharge


def cells_for(temps=("25",)) -> list[str]:
    """Cell IDs cycled at the given temperatures, e.g. ("25",) or ("25", "35", "45")."""
    temps = {str(t) for t in temps}
    return [c for c in CELLS if c[:2] in temps]


def load_dataset(state: str = "V", temps=("25",), raw_dir: Path | str | None = None) -> EISDataset:
    """Load all labelled spectra of one state for the cells at the given temperatures.

    Spectra without a matching capacity value (cycle n - 1) are dropped. Cells that have no file for
    the requested state (states VI-VIII exist only for some 25 °C cells) are skipped with a warning.
    """
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    eis_dir = _find_dir(raw_dir, "EIS_state_V_25C01.txt")
    cap_dir = _find_dir(raw_dir, "Data_Capacity_25C01.txt")

    parts, freqs = [], None
    for cell in cells_for(temps):
        path = eis_dir / f"EIS_state_{state}_{cell}.txt"
        if not path.exists():
            warnings.warn(f"no state-{state} file for cell {cell}; skipped")
            continue
        eis = read_eis(path)
        cap = read_capacity(cap_dir / f"Data_Capacity_{cell}.txt")

        counts = eis.groupby("cycle").size()
        if not (counts == N_FREQ).all():
            raise ValueError(f"{path.name}: expected {N_FREQ} frequencies per spectrum, got {counts.unique()}")
        cycles = counts.index.to_numpy()
        f = eis["freq_hz"].to_numpy().reshape(len(cycles), N_FREQ)
        if freqs is None:
            freqs = f[0]
        if not np.allclose(f, freqs, rtol=1e-6):
            raise ValueError(f"{path.name}: frequency grid differs from the other files")

        cap_cycle = cycles - 1
        labelled = np.isin(cap_cycle, cap.index)
        re = eis["re_ohm"].to_numpy().reshape(len(cycles), N_FREQ)[labelled]
        im = eis["negim_ohm"].to_numpy().reshape(len(cycles), N_FREQ)[labelled]
        capacity = cap.loc[cap_cycle[labelled]].to_numpy()

        parts.append(dict(
            X_re=re, X_im=im, capacity=capacity, soh=capacity / cap.iloc[0],
            cell_id=np.full(len(capacity), cell), cycle=cycles[labelled].astype(int),
            temp=np.full(len(capacity), int(cell[:2])),
        ))

    if not parts:
        raise ValueError(f"no data for state {state!r} at temperatures {temps}")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return EISDataset(freqs=freqs, **cat)
