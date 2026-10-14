"""Checks on the loaded dataset. Skipped automatically if the raw data is not downloaded."""

import numpy as np
import pandas as pd
import pytest

from conftest import requires_data
from frugal_eis.data import CELLS, N_FREQ, RAW_DIR, load_dataset

pytestmark = requires_data


@pytest.mark.parametrize("name", ["ds25", "ds_all"])
def test_shapes_consistent(name, request):
    d = request.getfixturevalue(name)
    n = len(d)
    assert n > 0
    assert d.X_re.shape == d.X_im.shape == (n, N_FREQ)
    assert d.freqs.shape == (N_FREQ,)
    for arr in (d.soh, d.capacity, d.cell_id, d.cycle, d.temp):
        assert arr.shape == (n,)


def test_no_nans(ds_all):
    for arr in (ds_all.X_re, ds_all.X_im, ds_all.freqs, ds_all.soh, ds_all.capacity):
        assert not np.isnan(arr).any()


def test_soh_range(ds_all):
    # Not (0.3, 1.1): cell 25C05 genuinely collapses to SoH 0.18 (see NOTES.md, Day 2 Block 2).
    assert (ds_all.soh > 0).all()
    assert (ds_all.soh <= 1 + 1e-12).all()
    first = pd.Series(ds_all.soh).groupby(ds_all.cell_id).first()
    np.testing.assert_allclose(first.to_numpy(), 1.0)


def test_each_cell_has_one_temperature(ds_all):
    temps_per_cell = pd.Series(ds_all.temp).groupby(ds_all.cell_id).nunique()
    assert (temps_per_cell == 1).all()
    for cell, t in zip(ds_all.cell_id, ds_all.temp):
        assert int(cell[:2]) == t


def test_cells_and_counts(ds25, ds_all):
    assert sorted(np.unique(ds25.cell_id)) == [c for c in CELLS if c.startswith("25")]
    assert sorted(np.unique(ds_all.cell_id)) == sorted(CELLS)   # 25C42 never loaded
    assert len(ds25) == 1343


def test_frequencies_descending(ds25):
    assert (np.diff(ds25.freqs) < 0).all()
    assert ds25.freqs[0] == pytest.approx(20004.453)
    assert ds25.freqs[-1] == pytest.approx(0.01999)


def test_cycles_increase_within_each_cell(ds_all):
    for cell in np.unique(ds_all.cell_id):
        assert (np.diff(ds_all.cycle[ds_all.cell_id == cell]) > 0).all()


def test_matches_tutorial_csv(ds25):
    path = RAW_DIR / "data_eis_zhang2020.csv"
    if not path.exists():
        pytest.skip("tutorial CSV not present")
    tut = pd.read_csv(path)
    assert (np.array([f"25C0{i}" for i in tut.seriesIdx]) == ds25.cell_id).all()
    assert (tut.cycle.to_numpy() == ds25.cycle).all()
    np.testing.assert_allclose(tut.filter(like="ZReal_").to_numpy(), ds25.X_re)
    np.testing.assert_allclose(tut.filter(like="ZImag_").to_numpy(), ds25.X_im)
    np.testing.assert_allclose(tut.Q.to_numpy(), ds25.capacity)


def test_missing_state_files_warn():
    with pytest.warns(UserWarning, match="no state-VII file"):
        d = load_dataset("VII", ("25",))
    assert sorted(np.unique(d.cell_id)) == ["25C04", "25C06", "25C07", "25C08"]


def test_invalid_state_raises():
    with pytest.raises(ValueError):
        load_dataset("X")
