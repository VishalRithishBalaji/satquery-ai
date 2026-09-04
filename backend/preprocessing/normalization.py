import numpy as np

def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx <= mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)
