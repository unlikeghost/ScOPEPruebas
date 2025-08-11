import ot
import numpy as np
from scope.predictor.base import _BasePredictor



class ScOPEOT(_BasePredictor):

    def __init__(self,
                 use_prototypes: bool = False,
                 matching_metric: str = None, 
                 get_probas: bool = False, 
                 epsilon: float = 1e-12) -> None:
        
        super().__init__(
            get_probas=get_probas, 
            epsilon=epsilon,
            use_prototypes=use_prototypes
        )
        
        self.supported_matching_metrics = {
            "matching": lambda x1, x2: self.__matching__(x1, x2),
            "jaccard": lambda x1, x2: self.__jaccard__(x1, x2),
            "dice": lambda x1, x2: self.__dice__(x1, x2),
            "overlap": lambda x1, x2: self.__overlap__(x1, x2),
        }
        
        if matching_metric and  matching_metric not in self.supported_matching_metrics:
            raise ValueError(f"Unsupported matching metric: {matching_metric}")
                
        if not matching_metric:
            self.use_matching: bool = False
            self.matching_method = None
            
        else:    
            self.use_matching: bool = True
            self.matching_metric = self.supported_matching_metrics[matching_metric]
    
    def __matching__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute matching (intersection) between two arrays"""
        return np.sum(
                np.minimum(x1, x2), 
                dtype=np.float32
            ).item()
    
    def __jaccard__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Jaccard distance: 1 - (intersection / union)"""
        intersection = self.__matching__(x1, x2)
        union = np.sum(np.maximum(x1, x2), dtype=np.float32)
        
        if union < self.epsilon:
            return 0.0
        
        return 1.0 - (intersection / union)
    
    def __dice__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Dice distance: 1 - (2 * intersection / (sum1 + sum2))"""
        intersection = self.__matching__(x1, x2)  # Corregido: era self._matching
        sum1 = np.sum(x1, dtype=np.float32)
        sum2 = np.sum(x2, dtype=np.float32)
        denominator = sum1 + sum2
        
        if denominator < self.epsilon:
            return 0.0
        
        return 1.0 - (2 * intersection / denominator)
    
    def __overlap__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Overlap distance: 1 - (intersection / min(sum1, sum2))"""
        intersection = self.__matching__(x1, x2)  # Corregido: era self._matching
        sum1 = np.sum(x1, dtype=np.float32)
        sum2 = np.sum(x2, dtype=np.float32)
        denominator = min(sum1, sum2)
        
        if denominator < self.epsilon:
            return 0.0
        
        return 1.0 - (intersection / denominator)

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
        
        current_cluster = current_cluster.reshape(-1, current_cluster.shape[-1])
        current_sample = current_sample.reshape(-1, current_sample.shape[-1]) 
        
        if self.use_prototypes:
            cluster_data = self.__compute_prototype__(current_cluster).reshape(1, -1)
            cluster_weights = np.array([1.0])  
            
        else:
            cluster_data = current_cluster
            
            cluster_weights = np.ones(current_cluster.shape[0]) / current_cluster.shape[0]
            
        sample_weights = np.ones(current_sample.shape[0]) / current_sample.shape[0]
        
        cost_matrix: np.ndarray = self.__compute_cost_matrix__(sample=current_sample, kw_samples=cluster_data)
                
        score: float = self.__wasserstein_distance__(
            cost_matrix=cost_matrix,
            cluster_weights=cluster_weights,
            sample_weights=sample_weights,
        )
        
        return score