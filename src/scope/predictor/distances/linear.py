import numpy as np


def cosine(x1: np.ndarray, x2: np.ndarray) -> float:
    """Robust cosine distance with zero-vector handling"""
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    x2 =  np.expand_dims(x2, axis=0)
    
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 1.0  # Maximum distance for zero vectors
    
    norm1 = np.linalg.norm(x1, axis=-1)
    norm2 = np.linalg.norm(x2, axis=-1)

    cosine_sim = np.sum(x1 * x2, axis=-1) / (norm1 * norm2)

    mask_zero = (norm1 < 1e-12) | (norm2 < 1e-12)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    distances = np.where(mask_zero, 1.0, 1.0 - cosine_sim)      

    return np.sum(distances).item()


def euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
    x1, x2 = np.asarray(x1), np.asarray(x2)
    return float(
        np.linalg.norm(x1 - x2)
    )


def minkowski(x1: np.ndarray, x2: np.ndarray, p: float = 2) -> float:
    x1, x2 = np.asarray(x1), np.asarray(x2)
    return float(np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p))
