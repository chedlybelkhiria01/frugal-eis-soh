import pytest

from frugal_eis.data import RAW_DIR, load_dataset


def _have_raw_data() -> bool:
    return any(p for p in RAW_DIR.rglob("EIS_state_V_25C01.txt") if "__MACOSX" not in p.parts)


requires_data = pytest.mark.skipif(not _have_raw_data(), reason="raw data not downloaded (see data/README.md)")


@pytest.fixture(scope="session")
def ds25():
    """State V, 25 °C cells (the main experimental set)."""
    return load_dataset("V", ("25",))


@pytest.fixture(scope="session")
def ds_all():
    """State V, all 12 cells."""
    return load_dataset("V", ("25", "35", "45"))
