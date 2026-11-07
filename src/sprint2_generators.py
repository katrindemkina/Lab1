import numpy as np


def _check(values, probabilities):
    values = np.asarray(values)
    p = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or p.ndim != 1 or len(values) == 0 or len(values) != len(p):
        raise ValueError("Values and probabilities must have the same length.")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Probabilities must be nonnegative and sum to one.")
    return values, p / p.sum()


def inverse_cdf(values, probabilities, size, seed=42):
    values, p = _check(values, probabilities)
    rng = np.random.default_rng(seed)
    borders = np.cumsum(p)
    borders[-1] = 1.0
    indexes = np.searchsorted(borders, rng.random(size), side="right")
    return values[indexes]


def make_alias_table(probabilities):
    p = np.asarray(probabilities, dtype=float)
    n = len(p)
    q = p.copy() * n
    cut = np.ones(n)
    alias = np.arange(n)
    small = [i for i in range(n) if q[i] < 1]
    large = [i for i in range(n) if q[i] >= 1]

    while small and large:
        s = small.pop()
        l = large.pop()
        cut[s] = q[s]
        alias[s] = l
        q[l] -= 1 - q[s]
        (small if q[l] < 1 else large).append(l)

    return cut, alias


def alias_sample(values, probabilities, size, seed=42):
    values, p = _check(values, probabilities)
    cut, alias = make_alias_table(p)
    rng = np.random.default_rng(seed)
    columns = rng.integers(0, len(values), size=size)
    coins = rng.random(size)
    indexes = np.where(coins < cut[columns], columns, alias[columns])
    return values[indexes]
