from __future__ import annotations

def max_abs_error(actual: list[float], expected: list[float]) -> float:
    if len(actual) != len(expected):
        raise ValueError(
            f"Length mismatch: actual has {len(actual)} values,"
            f"expected has {len(expected)} values"
        )
    
    if not actual:
        return 0.0
    
    return max(abs(a - e) for a, e in zip(actual, expected))