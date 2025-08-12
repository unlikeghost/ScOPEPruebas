import numpy as np
from typing import List, Tuple, Dict


def max(all_probas: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], str]:
    
    max_index_flat = np.argmax(all_probas)

    _, col_idx = np.unravel_index(max_index_flat, all_probas.shape)
    
    probas = all_probas[:, col_idx]
    
    probs_dict = dict(zip(classes, probas))

    prediction = classes[np.argmax(probas)]

    return probs_dict, prediction

def median(all_probas: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], str]:
    
    probs_final = np.median(all_probas.T, axis=0)
    
    return dict(zip(classes, probs_final)), classes[np.argmax(probs_final)]

def hard(all_probas: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], str]:
            
    votes = np.argmax(all_probas.T, axis=1)
    counts = np.bincount(votes, minlength=len(classes))
    probs_final = counts / len(votes)
    return dict(zip(classes, probs_final)), classes[np.argmax(probs_final)]


def soft(all_probas: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], str]:
    probs_final = np.mean(all_probas.T, axis=0)        
    return dict(zip(classes, probs_final)), classes[np.argmax(probs_final)]

def borda(all_probas: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], str]:
    all_probas_T = all_probas.T  # shape (n_modelos, n_clases)
    n_classes = all_probas_T.shape[1]
    
    ranks = np.argsort(np.argsort(-all_probas_T, axis=1), axis=1)
    
    scores = n_classes - ranks
    
    total_scores = np.sum(scores, axis=0)
    
    probs_final = total_scores / (n_classes * all_probas_T.shape[0])
    
    return dict(zip(classes, probs_final)), classes[np.argmax(total_scores)]