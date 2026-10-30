"""Small rank statistics for the pilots (no scipy dependency)."""

import math
import statistics


def ranks(values: list[float]) -> list[float]:
    """Average ranks (1-based), ties share their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None  # a constant series has no correlation
    return sxy / math.sqrt(sxx * syy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(ranks(x), ranks(y)) if len(x) >= 3 else None


def partial(r_xy: float | None, r_xz: float | None, r_yz: float | None) -> float | None:
    """Correlation of x and y controlling for z (first-order partial correlation)."""
    if None in (r_xy, r_xz, r_yz):
        return None
    denom = math.sqrt(max((1 - r_xz**2) * (1 - r_yz**2), 0.0))
    return None if denom < 1e-12 else (r_xy - r_xz * r_yz) / denom


def top_set(values: list[float], fraction: float) -> set[int]:
    k = max(1, math.ceil(fraction * len(values)))
    return set(sorted(range(len(values)), key=lambda i: -values[i])[:k])


def jaccard_top(x: list[float], y: list[float], fraction: float = 0.1) -> float:
    a, b = top_set(x, fraction), top_set(y, fraction)
    return len(a & b) / len(a | b)


def summarize(values: list[float | None]) -> dict:
    present = [v for v in values if v is not None and math.isfinite(v)]
    if not present:
        return {"n": 0}
    return {"n": len(present), "mean": statistics.fmean(present), "median": statistics.median(present),
            "std": statistics.pstdev(present)}
