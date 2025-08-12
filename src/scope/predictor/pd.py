import numpy as np
from scope.predictor.base import _BasePredictor


class ScOPEPD(_BasePredictor):

    def __init__(self, 
                 distance_metric: str = "cosine", **kwargs) -> None:
        
        super().__init__(
            **kwargs
        )

        self.supported_distance_metrics = {
            "cosine": lambda x1, x2: self.__cosine_distance__(x1, x2),
            "euclidean": lambda x1, x2: self.__euclidean_distance__(x1, x2),
            "minkowski": lambda x1, x2: self.__minkowski_distance__(x1, x2, p=3),
        }
        
        if distance_metric not in self.supported_distance_metrics:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        self.distance_metric = self.supported_distance_metrics[distance_metric]

    
    def __cosine_distance__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Robust cosine distance with zero-vector handling"""
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        
        x2 =  np.expand_dims(x2, axis=0)
        
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        
        if norm1 < self.epsilon or norm2 < self.epsilon:
            return 1.0  # Maximum distance for zero vectors
        
        norm1 = np.linalg.norm(x1, axis=-1)
        norm2 = np.linalg.norm(x2, axis=-1)

        cosine_sim = np.sum(x1 * x2, axis=-1) / (norm1 * norm2)

        mask_zero = (norm1 < self.epsilon) | (norm2 < self.epsilon)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        distances = np.where(mask_zero, 1.0, 1.0 - cosine_sim)      
    
        return np.sum(distances).item()
    
    @staticmethod
    def __euclidean_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        x1, x2 = np.asarray(x1), np.asarray(x2)
        return float(
            np.linalg.norm(x1 - x2)
        )
    
    @staticmethod
    def __minkowski_distance__(x1: np.ndarray, x2: np.ndarray, p: float = 2) -> float:
        x1, x2 = np.asarray(x1), np.asarray(x2)
        return float(np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p))
    
    
    def __forward__(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:
        """
        Compute distance between sample and cluster prototype.
        
        Args:
            current_cluster: Cluster data matrix
            current_sample: Sample data matrix
            
        Returns:
            Distance score as float
        """
        
        current_cluster = current_cluster.reshape(-1, current_cluster.shape[-1])
        current_sample = current_sample.reshape(-1, current_sample.shape[-1]) 
        
        if self.use_prototypes:
            prototype = self.__compute_prototype__(current_cluster)
            score = self.distance_metric(current_sample, prototype)
        
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