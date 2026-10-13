"""Shared WDM channel-selection schemes (extremes-first etc.)."""

import numpy as np


def select_channels(n_full, count, scheme="alternating"):
    """
    Return sorted indices into the full channel grid for the requested count.

    Schemes
    -------
    alternating : extremes inward (max, min, max-1, min+1, ...) - nested.
    extremes    : half from the low end + half from the high end.
    uniform     : evenly spaced across the band.
    first_n     : the first `count` channels.
    """
    count = int(count)
    if count >= n_full:
        return list(range(n_full))
    if scheme == "alternating":
        selected = []
        left, right = 0, n_full - 1
        while len(selected) < count and left <= right:
            selected.append(right)
            if len(selected) < count and left < right:
                selected.append(left)
            right -= 1
            left += 1
        return sorted(selected[:count])
    if scheme == "extremes":
        half = count // 2
        return sorted(list(range(half))
                      + list(range(n_full - (count - half), n_full)))
    if scheme == "uniform":
        step = max(1, n_full // count)
        return sorted(list(range(0, n_full, step))[:count])
    if scheme == "first_n":
        return list(range(count))
    raise ValueError("Unknown channel_selection: %s" % scheme)


def union_of_selections(n_full, counts, scheme="alternating"):
    """Sorted union of the indices selected for every count in `counts`."""
    s = set()
    for c in counts:
        s.update(select_channels(n_full, c, scheme))
    return sorted(s)


def local_indices(selected_full_indices, union_indices):
    """Map full-grid indices to positions inside the stored union array."""
    pos = {idx: k for k, idx in enumerate(union_indices)}
    return np.array([pos[i] for i in selected_full_indices], dtype=int)
