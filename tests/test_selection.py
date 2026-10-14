"""Frequency selectors on synthetic spectra with known answers (no dataset needed)."""

import numpy as np
import pytest

from conftest import requires_data
from frugal_eis.selection import (FrequencySelector, SelectedModel, clear_rank_cache, columns_for, rank_ard, rank_mrmr,
                                  rank_rfe, select_logspaced, select_mrmr, select_random)

NF = 10   # small toy grid: 10 frequencies -> 20 columns [Re, -Im]


def _toy(n=300, seed=0):
    """y depends on -Im at freq 2 (strongly) and Re at freq 7 (weaker, independent).
    -Im at freq 3 is a near-copy of -Im at freq 2: relevant but redundant."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2 * NF))
    a, b = rng.normal(size=n), rng.normal(size=n)
    X[:, NF + 2] = a
    X[:, NF + 3] = a + 0.05 * rng.normal(size=n)
    X[:, 7] = b
    y = 2.0 * a + 1.0 * b + 0.1 * rng.normal(size=n)
    return X, y


def test_columns_for_maps_frequency_to_re_and_im():
    np.testing.assert_array_equal(columns_for([0, 3], n_freq=10), [0, 3, 10, 13])


def test_mrmr_prefers_independent_over_redundant():
    X, y = _toy()
    order = rank_mrmr(X, y, NF)
    assert order[0] in (2, 3)              # the strongest signal (2 and 3 are near-identical)
    assert order[1] == 7                   # independent signal beats the redundant copy
    assert sorted(order) == list(range(NF))


def test_relevance_alone_would_pick_the_copy():
    # sanity check of the toy: by plain |corr| the duplicate (freq 3) ranks above freq 7
    X, y = _toy()
    rel = [abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(2 * NF)]
    assert rel[NF + 3] > rel[7]


def test_selectors_are_prefixes_of_their_ranking():
    X, y = _toy()
    np.testing.assert_array_equal(select_mrmr(X, y, 3, NF), select_mrmr(X, y, 6, NF)[:3])


def test_rfe_ranks_a_true_signal_first():
    X, y = _toy()
    order = rank_rfe(X, y, NF)
    assert order[0] in (2, 3, 7)
    assert set(order[:3]) >= {7}
    assert sorted(order) == list(range(NF))


def test_ard_ranks_the_relevant_im_frequency_first():
    X, y = _toy()
    order = rank_ard(X, y, NF, stride=2)   # ARD sees only -Im columns: freq 2/3 carry the signal
    assert order[0] in (2, 3)
    assert sorted(order) == list(range(NF))


def test_logspaced():
    freqs = np.geomspace(20000, 0.02, 60)
    for k in range(1, 61):
        idx = select_logspaced(freqs, k)
        assert len(idx) == k and len(set(idx)) == k
    np.testing.assert_array_equal(select_logspaced(freqs, 2), [0, 59])          # both band edges
    assert select_logspaced(freqs, 1)[0] in (29, 30)                             # geometric middle
    with pytest.raises(ValueError):
        select_logspaced(freqs, 0)


def test_random_is_reproducible_and_varies_with_seed():
    a, b, c = select_random(5, 0), select_random(5, 0), select_random(5, 1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(set(a)) == 5


def test_frequency_selector_keeps_both_columns():
    X, y = _toy()
    sel = FrequencySelector("fixed", freq_idx=[2, 7], n_freq=NF).fit(X, y)
    np.testing.assert_array_equal(sel.transform(X), X[:, [2, 7, NF + 2, NF + 7]])
    with pytest.raises(ValueError):
        FrequencySelector("mrmr", n_freq=NF).fit(X[:, :5], y)


def test_frequency_selector_cache_gives_same_ranking():
    X, y = _toy()
    clear_rank_cache()
    s3 = FrequencySelector("mrmr", k=3, n_freq=NF).fit(X, y).selected_
    s5 = FrequencySelector("mrmr", k=5, n_freq=NF).fit(X, y).selected_
    np.testing.assert_array_equal(s3, s5[:3])


def test_selected_model_forwards_groups_to_group_tuned_model():
    from frugal_eis.models import GroupRidgeCV
    X, y = _toy()
    groups = np.repeat(["a", "b", "c"], 100)
    m = SelectedModel(FrequencySelector("fixed", freq_idx=[2, 7], n_freq=NF), GroupRidgeCV()).fit(X, y, groups)
    assert hasattr(m.model_, "alpha_")                      # GroupRidgeCV ran its inner cell-wise tuning
    np.testing.assert_array_equal(m.selector_.selected_, [2, 7])
    assert m.predict(X).shape == y.shape


@requires_data
def test_selection_inside_loco_never_sees_held_out_cell(ds25):
    """Pipeline(selector, model) under loco_evaluate: each fold's selection is fitted on its training rows."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    from frugal_eis.evaluate import loco_evaluate
    from frugal_eis.features import full_spectrum

    X = full_spectrum(ds25.X_re, ds25.X_im)
    seen = []

    class Spy(FrequencySelector):
        def fit(self, X, y=None):
            seen.append(len(X))
            return super().fit(X, y)

    res = loco_evaluate(lambda: make_pipeline(Spy("mrmr", k=3), LinearRegression()), X, ds25.soh,
                        ds25.cell_id)
    for n_fit, (_, row) in zip(seen, res.per_cell.iterrows()):
        assert n_fit == row.n_train                    # fitted on exactly the training rows of that fold
