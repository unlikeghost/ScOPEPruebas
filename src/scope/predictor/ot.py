import ot
import numpy as np
from scope.predictor.base import _BasePredictor

from .distances import jaccard, dice, overlap


class ScOPEOT(_BasePredictor):

    def __init__(self, matching_metric: str = None, **kwargs) -> None:
        
        super().__init__(
            **kwargs
        )
        
        self.supported_matching_metrics = {
            "jaccard": lambda x1, x2: jaccard(x1, x2),
            "dice": lambda x1, x2: dice(x1, x2),
            "overlap": lambda x1, x2: overlap(x1, x2),
        }
        
        if matching_metric and  matching_metric not in self.supported_matching_metrics:
            raise ValueError(f"Unsupported matching metric: {matching_metric}")
                
        if not matching_metric:
            self.use_matching: bool = False
            self.matching_method = None
            
        else:    
            self.use_matching: bool = True
            self.matching_metric = self.supported_matching_metrics[matching_metric]
    
    def __compute_cost_matrix__(self, sample: np.ndarray, kw_samples: np.ndarray) -> np.ndarray:
        
        if self.use_matching:
            n_kw = kw_samples.shape[0]
            n_sample = sample.shape[0]
            cost_matrix = np.zeros((n_kw, n_sample))
            
            for i, kw_sample_point in enumerate(kw_samples):
                for j, sample_point in enumerate(sample):
                    cost_matrix[i, j] = self.matching_metric(
                        x1=kw_sample_point.reshape(1, -1), 
                        x2=sample_point.reshape(1, -1)
                    )
        else:
            cost_matrix = ot.dist(kw_samples, sample, metric='euclidean')

        return cost_matrix

    @staticmethod
    def __wasserstein_distance__(sample_weights: np.ndarray, cluster_weights: np.ndarray, cost_matrix: np.ndarray) -> float:
        return ot.emd2(cluster_weights, sample_weights, cost_matrix)
    
    def __forward__(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:

        if self.aggregation_method is not None:
            cluster_weights = np.array([1.0])  
            
        else:
            
            cluster_weights = np.ones(current_cluster.shape[0]) / current_cluster.shape[0]
            
        sample_weights = np.ones(current_sample.shape[0]) / current_sample.shape[0]
        
        cost_matrix: np.ndarray = self.__compute_cost_matrix__(sample=current_sample, kw_samples=current_cluster)
                
        score: float = self.__wasserstein_distance__(
            cost_matrix=cost_matrix,
            cluster_weights=cluster_weights,
            sample_weights=sample_weights,
        )
        
        return score