import copy
import numpy as np
from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import Union, Tuple, Any, List, Dict


from scope.compression.metrics import get_metric, _BaseMetric, CompressionData
from scope.compression.compressors import get_compressor, _BaseCompressor 


class _BaseMatrixFactory(ABC):
    def __init__(self,
                 compressor_names: Union[str, List[str]],
                 compression_metric_names: Union[str, List[str]],
                 symmetric: bool = True,
                 concat_value: str = " ",
                 compression_level: int = 9,
                 min_size_threshold: int = 0):
        
        if not isinstance(concat_value, str):
            raise ValueError("`concat_value` must be a string.")
        
        self._compressor_names = self._normalize_to_list(compressor_names)
        self._metric_names = self._normalize_to_list(compression_metric_names)
        
        if not self._compressor_names:
            raise ValueError("At least one compressor name must be provided.")
        if not self._metric_names:
            raise ValueError("At least one metric name must be provided.")
        
        self._metrics: List[_BaseMetric] = [
            get_metric(metric_name)
            for metric_name in self._metric_names
        ]
        
        self._compressors: List[_BaseCompressor] = [
            get_compressor(
                name=compressor_name,
                compression_level=compression_level,
                min_size_threshold=min_size_threshold
            )
            for compressor_name in self._compressor_names
        ]
        
        self._symmetric = symmetric
        self._concat_value = concat_value
        self._compression_level = compression_level
        self._min_size_threshold = min_size_threshold
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"compressors={self._compressor_names}, "
            f"metrics={self._metric_names}, "
            f"symmetric={self._symmetric})"
        )
    
    def __str__(self):
        return self.__repr__()
    
    @staticmethod
    def _normalize_to_list(item: Union[str, List[str]]) -> List[str]:
        """Convierte string a lista de un elemento, o devuelve la lista tal como está"""
        if isinstance(item, str):
            return [item]
        elif isinstance(item, list):
            return item
        else:
            raise TypeError("Expected str or List[str]")
    
    
    def get_best_sigma(self, sample: str, *kw_samples: str) -> float:
        """
        Computes the average sigma for the provided data.

        Parameters:
        sample: str
            Base sample for comparison.
        *kw_samples: str
            Additional samples.

        Returns:
        float
            The average sigma.
        """
        all_data = np.array([sample] + list(kw_samples), dtype=object)
        
        def compute_sigma(sequence: str) -> float:
            """Computes the sigma for a specific sequence."""
            simgas : list = []
            for c_idx, _ in enumerate(self._compressors):
                for m_idx, metric in enumerate(self._metrics):
                    compute = metric.compute
                    
                    c_x1, c_x2, c_x1x2, c_x2x1 =  self.compute_compression_len(
                        x1=sequence,
                        x2=sequence,
                        compressor_index=c_idx
                    )
                    data = CompressionData(
                        c_x1=c_x1,
                        c_x2=c_x2,
                        c_x1x2=c_x1x2,
                        c_x2x1=c_x2x1
                    )
                    
                    distance = compute(
                        data
                    )
                    
                    simgas.append(distance)
            
            return np.mean(simgas).item()
        
        compute_sigma_vec = np.vectorize(compute_sigma)
        sigmas = compute_sigma_vec(all_data)
        
        return np.mean(sigmas)
    
    
    def compute_compression_len(self, x1: str, x2: str, compressor_index: int) -> Tuple[int, int, int, int]:
        
        x1x2: bytes = f'{x1}{self._concat_value}{x2}'.encode("utf-8")
        x2x1: bytes = f'{x2}{self._concat_value}{x1}'.encode("utf-8")
        
        _, _, c_x1 = self._compressors[compressor_index](x1.encode("utf-8"))
        _, _, c_x2 = self._compressors[compressor_index](x2.encode("utf-8"))
        _, _, c_x1x2 = self._compressors[compressor_index](x1x2)
        _, _, c_x2x1 = self._compressors[compressor_index](x2x1)
        
        return c_x1, c_x2, c_x1x2, c_x2x1

    @abstractmethod
    def compute_compression_matrix(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def build_matrix(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    
    def __call__(self, *args, **kwargs) -> Any:
        return self.build_matrix(*args, **kwargs)


class CompressionMatrixFactory(_BaseMatrixFactory):
    
    def __init__(self,
                 compressor_names: Union[str, List[str]],
                 compression_metric_names: Union[str, List[str]],
                 symmetric: bool = True,
                 concat_value: str = " ",
                 compression_level: int = 9,
                 min_size_threshold: int = 0):
        
        super().__init__(
            compressor_names=compressor_names,
            compression_metric_names=compression_metric_names,
            symmetric=symmetric,
            concat_value=concat_value,
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )
    
    def compute_compression_matrix(self, samples: List[str]) -> np.ndarray:
        
        n_compressors = len(self._compressors)
        n_metrics = len(self._metrics)
        n_samples = len(samples)

        compression_matrix = np.zeros(
            shape=(n_compressors, n_metrics, n_samples, n_samples),
            dtype=np.float32
        )
        
        if self._symmetric:
            index_pairs = list(combinations(range(n_samples), 2))
            index_pairs += [(i, i) for i in range(n_samples)]
        else:
            index_pairs = list(product(range(n_samples), repeat=2))

        for c_idx, _ in enumerate(self._compressors):
            for m_idx, metric in enumerate(self._metrics):
                compute = metric.compute
                for i, j in index_pairs:
                    
                    c_x1, c_x2, c_x1x2, c_x2x1 =  self.compute_compression_len(
                        x1=samples[i],
                        x2=samples[j],
                        compressor_index=c_idx
                    )
                    data = CompressionData(
                        c_x1=c_x1,
                        c_x2=c_x2,
                        c_x1x2=c_x1x2,
                        c_x2x1=c_x2x1
                    )
                    
                    distance = compute(
                        data
                    )
                    
                    compression_matrix[c_idx, m_idx, i, j] = distance
                    if self._symmetric and i != j:
                        compression_matrix[c_idx, m_idx, j, i] = distance
        
        return compression_matrix
        
            
    def build_matrix(self,  
                     sample: str,
                     kw_samples: Union[Dict[Union[int, str], str], Tuple, List],
                     get_sigma: bool = False) -> Dict[str, np.ndarray]:
        
        if isinstance(kw_samples, dict):
            cluster_samples = copy.deepcopy(kw_samples)
        
        else:
            cluster_samples = {
                index: value for index, value in enumerate(kw_samples)
            }
        
        results = {}
        all_sigmas = []

        for cluster_key, cluster_values in cluster_samples.items():
            
            cluster_values += [sample]
            
            compression_matrix = self.compute_compression_matrix(cluster_values)
            
            results[f"ScOPE_KwSamples_{cluster_key}"] = compression_matrix[:, :, :-1, :]
            results[f"ScOPE_UkSample_{cluster_key}"] = compression_matrix[:, :, -1:, :]

            if self.get_best_sigma:
                sigma = self.get_best_sigma(sample, *cluster_values[:-1])
                all_sigmas.append(sigma)
        
        if get_sigma:
            results["sigma"] = np.mean(all_sigmas)
            
        return results


if __name__ == "__main__":

    test_samples = {
        'class_0': ['Hola', 'Adios', 'Buenos dias'],
        'class_1': ['Hello', 'Goodbye', 'Good morning']
    }
    test_sample = 'hola'
       
    matrix = CompressionMatrixFactory(
        compressor_names=['gzip','bz2'],
        compression_metric_names=['ncd', 'clm'],
        concat_value=" ",
        min_size_threshold=80
    )
    
    data = matrix.build_matrix(
        sample=test_sample,
        kw_samples=test_samples,
        get_sigma=True
    )
        
    print(data)