import numpy as np
from typing import Dict, Union, Any, List, Tuple, Optional
from abc import abstractmethod, ABC


class _BasePredictor(ABC):
    start_key_value_matrix: str = 'ScOPE_KwSamples_'
    start_key_value_sample: str = 'ScOPE_UkSample_'
    
    valid_methods = {
        "mean": lambda data: np.mean(data, axis=0),
        "median": lambda data: np.median(data, axis=0),
        "min": lambda data: np.min(data, axis=0),
        "max": lambda data: np.max(data, axis=0),
        "sum": lambda data: np.sum(data, axis=0)
    }
    
    __compute_gaussian_function__ = lambda x, sigma: np.exp(-0.5 * np.square((x / sigma)))

    def __init__(self,
                 epsilon: float = 1e-12,
                 aggregation_method: Optional[str] = None
                 ):
        
        self.epsilon = epsilon
        self.aggregation_method = aggregation_method
        
        if aggregation_method is not None and aggregation_method not in self.valid_methods:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}. "
                           f"Valid options: {self.valid_methods} or None for no aggregation")
    
    def __compute_aggregated_prototype__(self, data: np.ndarray) -> np.ndarray:
        """Compute prototype using specified aggregation method"""
        return self.valid_methods[self.aggregation_method](
            data
        ).reshape(1, -1)

    @abstractmethod
    def __forward__(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    
    def forward(self, list_of_data: List[Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:

        if not isinstance(list_of_data, list):
            raise ValueError("Input should be a list of dictionaries containing data matrices.")
        
        if not list_of_data:
            return []
        
        output: List[Dict[str, Any]] = []
        
        for data_matrix in list_of_data:
            
            cluster_keys: list = list(
                filter(
                    lambda x: x.startswith(self.start_key_value_matrix),
                    data_matrix.keys()
                )
            )

            sample_keys: list = list(
                filter(
                    lambda x: x.startswith(self.start_key_value_sample),
                    data_matrix.keys()
                )
            )
            
            this_output: Dict[str, Any] = {
                'scores': {
                    cluster_key[len(self.start_key_value_matrix):]: [0.0]
                    for cluster_key in cluster_keys
                },
            }
            
            for cluster_key in cluster_keys:
                real_cluster_name: str = cluster_key[len(self.start_key_value_matrix):]
                current_sample_key: str = list(
                    filter(
                        lambda x: x.endswith(real_cluster_name),
                        sample_keys)
                )[0]
                
                current_cluster: np.ndarray = data_matrix[cluster_key]
                current_sample: np.ndarray = data_matrix[current_sample_key]
                
                n_compressors, n_metrics, _, _ = current_sample.shape
                this_class_scores = []
                
                if data_matrix.get("best_sigma"):
                    current_cluster = self.__compute_gaussian_function__(
                        x=current_cluster,
                        sigma=data_matrix["best_sigma"]
                    )
                    current_sample = self.__compute_gaussian_function__(
                        x=current_sample,
                        sigma=data_matrix["best_sigma"]
                    )
                    
                for compressor in range(n_compressors):
                    for metric in range(n_metrics):
                        
                        cluster_ = current_cluster[compressor, metric, :, :]
                        if self.aggregation_method is not None:
                            cluster_ = self.__compute_aggregated_prototype__(
                                cluster_
                            )
                        
                        sample_ = current_sample[compressor, metric, :, :]
                        
                        cluster_ = cluster_.reshape(-1, cluster_.shape[-1])
                        sample_ = sample_.reshape(-1, sample_.shape[-1]) 
                        
                        current_score = self.__forward__(
                            cluster_,
                            sample_
                        )
                        
                        this_class_scores.append(current_score)
                
                this_output['scores'][real_cluster_name] = this_class_scores
            
            output.append(this_output)
            
        return output

    def __call__(self, list_of_data: Union[List[Dict[str, np.ndarray]], Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:

        if not list_of_data:
            return []

        if isinstance(list_of_data, dict):
            list_of_data = [list_of_data]
            
        return self.forward(list_of_data)