import numpy as np
import pytest


def _coverage_at_target(mid, half_width, y_test, target: float) -> float:
    lo, hi = 0.5, 5.0
    for _ in range(50):
        scale = (lo + hi) / 2
        lower = mid - scale * half_width
        upper = mid + scale * half_width
        cov = float(np.mean((lower <= y_test) & (y_test <= upper)))
        if cov < target:
            lo = scale
        else:
            hi = scale
    return round(cov, 4)


def test_coverage_levels_differ():
    rng = np.random.default_rng(42)
    y_test = rng.normal(0.1, 0.05, 100)
    # Use deliberately narrow/shifted intervals so coverage is not trivially 100%
    # mid is shifted away from y_test so binary search scale actually matters
    y_p10 = np.full(100, np.percentile(y_test, 10))
    y_p90 = np.full(100, np.percentile(y_test, 90))
    half_width = (y_p90 - y_p10) / 2
    mid = (y_p10 + y_p90) / 2
    c80 = _coverage_at_target(mid, half_width, y_test, 0.80)
    c85 = _coverage_at_target(mid, half_width, y_test, 0.85)
    c90 = _coverage_at_target(mid, half_width, y_test, 0.90)
    assert c85 >= c80, f"85% coverage {c85} must be >= 80% {c80}"
    assert c90 >= c85, f"90% coverage {c90} must be >= 85% {c85}"
    assert c90 != c80, "coverage levels must differ"


def test_coverage_at_target_converges():
    rng = np.random.default_rng(0)
    y_test = rng.normal(0, 1, 500)
    mid = np.zeros(500)
    half_width = np.ones(500)
    c80 = _coverage_at_target(mid, half_width, y_test, 0.80)
    assert 0.75 <= c80 <= 0.95, f"Coverage {c80} out of reasonable range"
