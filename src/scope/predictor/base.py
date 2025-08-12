import numpy as np
from typing import Dict, Union, Any, List, Tuple
from abc import abstractmethod, ABC


class _BasePredictor(ABC):
    start_key_value_matrix: str = 'ScOPE_KwSamples_'
    start_key_value_sample: str = 'ScOPE_UkSample_'
    
    def __init__(self,
                 epsilon: float = 1e-12,
                 use_prototypes: bool = False,
                 ensemble_strategy: str = 'voting'
                 ):
        self.epsilon = epsilon
        self.use_prototypes = use_prototypes

    @staticmethod
    def __compute_prototype__(data: np.ndarray) -> np.ndarray:
        return np.mean(data, axis=0)


    @staticmethod
    def __gaussian_function__(x: np.ndarray, sigma: Union[np.ndarray, float]) -> np.ndarray:
        return np.exp(
            -0.5 * np.square(
                (x / sigma)
            )
        )

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
                
                if np.allclose(current_sample, 0, atol=self.epsilon):
                    current_sample = current_sample + np.random.normal(0, self.epsilon, current_sample.shape)
                
                if np.allclose(current_cluster, 0, atol=self.epsilon):
                    current_cluster = current_cluster + np.random.normal(0, self.epsilon, current_cluster.shape)

                if data_matrix.get("best_sigma"):
                    current_cluster = self.__gaussian_function__(
                        x=current_cluster,
                        sigma=data_matrix["best_sigma"]
                    )
                    current_sample = self.__gaussian_function__(
                        x=current_sample,
                        sigma=data_matrix["best_sigma"]
                    )
                    
                n_compressors, n_metrics, _, _ = current_sample.shape
                
                this_class_scores = []

                for compressor in range(n_compressors):
                    for metric in range(n_metrics):
                        
                        cluster_ = current_cluster[compressor, metric, :, :]
                        sample_ = current_sample[compressor, metric, :, :]
                        
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