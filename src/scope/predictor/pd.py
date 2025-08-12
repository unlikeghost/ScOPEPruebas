import numpy as np
from scope.predictor.base import _BasePredictor

from .distances import cosine, euclidean, minkowski


class ScOPEPD(_BasePredictor):

    def __init__(self, 
                 distance_metric: str = "cosine", **kwargs) -> None:
        
        super().__init__(
            **kwargs
        )

        self.supported_distance_metrics = {
            "cosine": lambda x1, x2: cosine(x1, x2),
            "euclidean": lambda x1, x2: euclidean(x1, x2),
            "minkowski": lambda x1, x2: minkowski(x1, x2, p=3),
        }
        
        if distance_metric not in self.supported_distance_metrics:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        self.distance_metric = self.supported_distance_metrics[distance_metric]

    
    def __forward__(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:
        """
        Compute distance between sample and cluster prototype.
        
        Args:
            current_cluster: Cluster data matrix
            current_sample: Sample data matrix
            
        Returns:
            Distance score as float
        """

        if self.aggregation_method is not None:
            score = self.distance_metric(current_sample, current_cluster)
        
        else:
            
            scores = []
            for kw_sample in current_cluster:
                score = self.distance_metric(current_sample, kw_sample)
                scores.append(score)
                
            score = np.sum(scores)
            
    
        if hasattr(score, 'item'):
            score = score.item()
        else:
            score = float(score)
        
        # Validate result
        if np.isnan(score) or np.isinf(score):
            raise ValueError(f"Invalid score calculated: {score}. Check input data for NaN or Inf values.")
        
        return score