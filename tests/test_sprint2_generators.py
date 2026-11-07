import numpy as np
import pytest

from src.sprint2_generators import inverse_cdf, alias_sample, make_alias_table


VALUES = np.array([2, 3, 4, 5, 7])
P = np.array([0.1, 0.4, 0.25, 0.2, 0.05])


@pytest.mark.parametrize("method", [inverse_cdf, alias_sample])
def test_only_allowed_values(method):
    result = method(VALUES, P, 2000, seed=7)
    assert set(result).issubset(set(VALUES))
    assert len(result) == 2000


@pytest.mark.parametrize("method", [inverse_cdf, alias_sample])
def test_seed_repeats_result(method):
    first = method(VALUES, P, 1000, seed=13)
    second = method(VALUES, P, 1000, seed=13)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("method", [inverse_cdf, alias_sample])
def test_frequencies_are_close(method):
    result = method(VALUES, P, 40000, seed=26)
    frequencies = np.array([(result == value).mean() for value in VALUES])
    assert np.max(np.abs(frequencies - P)) < 0.015


@pytest.mark.parametrize("method", [inverse_cdf, alias_sample])
def test_invalid_probabilities(method):
    with pytest.raises(ValueError):
        method(VALUES, [0.2, 0.2, 0.2, 0.2, 0.3], 100)
    with pytest.raises(ValueError):
        method(VALUES, [-0.1, 0.4, 0.25, 0.4, 0.05], 100)


@pytest.mark.parametrize("method", [inverse_cdf, alias_sample])
def test_single_possible_value(method):
    result = method(np.array([4]), np.array([1.0]), 100, seed=0)
    assert np.all(result == 4)


def test_alias_table_bounds():
    cut, alias = make_alias_table(P)
    assert np.all((cut >= 0) & (cut <= 1))
    assert np.all((alias >= 0) & (alias < len(P)))
