import json
import numpy as np
from copy import deepcopy
from typing import Dict, List, Union, Any, Generator, Optional
from collections import defaultdict

from scope.predictor import PredictorRegistry
from scope.compression import CompressionMatrixFactory


class ScOPE:
    def __init__(self,
                 compressor_names: Union[str, List[str]],
                 compression_metric_names: Union[str, List[str]],
                 use_symmetric_matrix: bool = True,
                 compression_level: int = 9,
                 min_size_threshold: int = 0,
                 use_best_sigma: bool = True,
                 concat_value: str = ' ',
                 get_probas: bool = True,
                 use_prototypes: bool = False,
                 model_type: str = "ot",
                 **model_kwargs) -> None:
        
        if isinstance(compressor_names, str):
            compressor_names = [compressor_names]
        if isinstance(compression_metric_names, str):
            compression_metric_names = [compression_metric_names]
                    
        self._compressor_names = compressor_names
        self._compression_metric_names = compression_metric_names

        self._using_symmetric_matrix = use_symmetric_matrix
        self._compression_level = compression_level
        self._min_size_threshold = min_size_threshold
        self._concat_value = concat_value
        self._using_sigma = use_best_sigma
        self._use_prototypes = use_prototypes
        self._get_probas = get_probas
        self._model_type = model_type
        self._model_kwargs = model_kwargs
        
        self.predictor = PredictorRegistry.create(
            name=model_type,
            get_probas=get_probas,
            use_prototypes=use_prototypes,
            **model_kwargs
        )
        
        self.matrix_generators = CompressionMatrixFactory(
            compressor_names=compressor_names,
            compression_metric_names=compression_metric_names,
            symmetric=use_symmetric_matrix,
            concat_value=concat_value,
            compression_level=compression_level,
            min_size_threshold=min_size_threshold,
        )
    
    def to_dict(self) -> dict:
        return {
            'compressor_names': self._compressor_names,
            'compression_metrics': self._compression_metric_names,
            'use_symmetric_matrix': self._using_symmetric_matrix,
            'compression_level': self._compression_level,
            'min_size_threshold': self._min_size_threshold,
            'concat_value': self._concat_value,
            'use_best_sigma': self._using_sigma,
            'get_probas': self._get_probas,
            'use_prototypes': self._use_prototypes,
            'model_type': self._model_type,
            'model_kwargs': self._coerce_kwargs_to_serializable(self._model_kwargs)
        }
    
    def _coerce_kwargs_to_serializable(self, kwargs: dict) -> dict:
        try:
            json.dumps(kwargs)
            return deepcopy(kwargs)
        except Exception:
            return {k: str(v) for k, v in kwargs.items()}

    def __forward__(self, sample: str, kw_samples: Dict[str, str], detailed: bool = False) -> Dict[str, Any]:
        
        matrix: dict = self.matrix_generators(
            sample=sample,
            kw_samples=kw_samples,
            get_sigma=self._using_sigma
        )
        return self.predictor(
            matrix
        )[0]
    
    def forward(self, list_samples: List[str], list_kw_samples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        
        predictions = []
        
        for sample, kw_samples in zip(list_samples, list_kw_samples):
            prediction = self.__forward__(sample, kw_samples)
            
            predictions.append(prediction)
        
        return predictions    
    
    def __call__(self, 
                 list_samples: Union[List[str], str],
                 list_kw_samples: Union[
                     List[
                         Dict[str, str]
                        ],
                     Dict[str, str]]
                     ) -> List[Dict[str, Any]]:
        
        if not isinstance(list_samples, list):
            list_samples = [list_samples]
        
        if not isinstance(list_kw_samples, list):
            list_kw_samples = [list_kw_samples]
        
        
        if len(list_samples) != len(list_kw_samples):
            raise ValueError(
                "The number of samples and keyword samples must be the same."
            )
            
        if len(list_samples) == 1:
            return self.__forward__(
                sample=list_samples[0],
                kw_samples=list_kw_samples[0]
            )
        
        return self.forward(
            list_kw_samples=list_kw_samples,
            list_samples=list_samples
        )
    
    def __str__(self):
        return (
            f"ScOPE(compressor_names={self._compressor_names}, "
            f"use_symmetric_matrix={self._using_symmetric_matrix}, "
            f"compression_metrics={self._compression_metric_names}, "
            f"compression_level={self._compression_level}, "
            f"min_size_threshold={self._min_size_threshold}, "
            f"use_best_sigma={self._using_sigma}, "
            f"get_probas={self._get_probas}, "
            f"use_prototypes={self._use_prototypes}, "
            f"model_type='{self._model_type}', "
            f"model_kwargs={self._model_kwargs})"
        )

    def __repr__(self):
        return self.__str__()