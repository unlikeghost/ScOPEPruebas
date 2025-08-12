import numpy as np


def matching(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute matching (intersection) between two arrays"""
    return np.sum(
            np.minimum(x1, x2), 
            dtype=np.float32
        ).item()

def jaccard(x1: np.ndarray, x2: np.ndarray) -> float:
    """Jaccard distance: 1 - (intersection / union)"""
    intersection = matching(x1, x2)
    union = np.sum(np.maximum(x1, x2), dtype=np.float32)
    
    if union < 1e-12:
        return 0.0
    
    return 1.0 - (intersection / union)

def dice(x1: np.ndarray, x2: np.ndarray) -> float:
    """Dice distance: 1 - (2 * intersection / (sum1 + sum2))"""
    intersection = matching(x1, x2)  # Corregido: era self._matching
    sum1 = np.sum(x1, dtype=np.float32)
    sum2 = np.sum(x2, dtype=np.float32)
    denominator = sum1 + sum2
    
    if denominator < 1e-12:
        return 0.0
    
    return 1.0 - (2 * intersection / denominator)

def overlap(x1: np.ndarray, x2: np.ndarray) -> float:
    """Overlap distance: 1 - (intersection / min(sum1, sum2))"""
    intersection = matching(x1, x2)  # Corregido: era self._matching
    sum1 = np.sum(x1, dtype=np.float32)
    sum2 = np.sum(x2, dtype=np.float32)
    denominator = min(sum1, sum2)
    
    if denominator < 1e-12:
        return 0.0
    
    return 1.0 - (intersection / denominator)
