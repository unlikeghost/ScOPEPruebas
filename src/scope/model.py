import json
import numpy as np
from copy import deepcopy
from typing import Dict, List, Union, Any, Tuple
from collections import defaultdict

from scope.predictor import PredictorRegistry
from scope.compression import CompressionMatrixFactory


class ScOPE:
    valid_ensemble_strategies = ['max', 'median', 'hard', 'borda', 'soft']
    
    def __init__(self,
                 compressor_names: Union[str, List[str]],
                 compression_metric_names: Union[str, List[str]],
                 ensemble_strategy: str = None,
                 use_symmetric_matrix: bool = True,
                 compression_level: int = 9,
                 min_size_threshold: int = 0,
                 use_best_sigma: bool = True,
                 concat_value: str = ' ',
                 use_prototypes: bool = False,
                 model_type: str = "ot",
                 **model_kwargs) -> None:
        
        _is_ensemble = False
        if isinstance(compressor_names, str):
            compressor_names = [compressor_names]
        if isinstance(compression_metric_names, str):
            compression_metric_names = [compression_metric_names]
        
        
        if len(compression_metric_names) > 1 or len(compressor_names) > 1:
            _is_ensemble = True
            
            if not ensemble_strategy:
                ensemble_strategy = 'max'
            
            if ensemble_strategy.lower() not in self.valid_ensemble_strategies:
                raise ValueError(
                    f"Invalid ensemble_strategy '{ensemble_strategy}'. "
                    f"Must be one of: {self.valid_ensemble_strategies}. "
                    f"Got: '{ensemble_strategy}'"
                )
            
            
            self._ensemble_strategy = ensemble_strategy
            
        self._compressor_names = compressor_names
        self._compression_metric_names = compression_metric_names
        self._is_ensemble = _is_ensemble
        self._ensemble_strategy = ensemble_strategy
        self._using_symmetric_matrix = use_symmetric_matrix
        self._compression_level = compression_level
        self._min_size_threshold = min_size_threshold
        self._concat_value = concat_value
        self._using_sigma = use_best_sigma
        self._use_prototypes = use_prototypes
        self._model_type = model_type
        self._model_kwargs = model_kwargs
        
        self.predictor = PredictorRegistry.create(
            name=model_type,
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
        model = {
            'compressor_names': self._compressor_names,
            'compression_metrics': self._compression_metric_names,
            'use_symmetric_matrix': self._using_symmetric_matrix,
            'compression_level': self._compression_level,
            'min_size_threshold': self._min_size_threshold,
            'concat_value': self._concat_value,
            'use_best_sigma': self._using_sigma,
            'use_prototypes': self._use_prototypes,
            'model_type': self._model_type,
            'model_kwargs': self._coerce_kwargs_to_serializable(self._model_kwargs)
        }
        
        if self._is_ensemble:
            model.update(
                {
                    'ensemble_strategy': self._ensemble_strategy
                }
            )
            
        return model
    
    def _coerce_kwargs_to_serializable(self, kwargs: dict) -> dict:
        try:
            json.dumps(kwargs)
            return deepcopy(kwargs)
        except Exception:
            return {k: str(v) for k, v in kwargs.items()}

    def __compute_probas__(self, dists: np.ndarray) -> np.ndarray:
        # dists = np.array(dists, dtype=float) ** 2

        # tau = np.median(dists)
        # delta = 0.2 * (np.max(dists) - np.min(dists))
        # alpha = np.log(9) / (2 * delta)
        
        # return 1.0 / (1.0 + np.exp(alpha * (dists - tau)))
        
        dists = np.array(dists, dtype=float)
        scores = 1.0 / (dists + 1e-8) ** 2
        probas = scores / np.sum(scores)
        return probas
    
    
    @staticmethod
    def __ensemble_max__(all_probas: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], str]:
        
        max_index_flat = np.argmax(all_probas)

        _, col_idx = np.unravel_index(max_index_flat, all_probas.shape)
        
        probas = all_probas[:, col_idx]
        
        probs_dict = dict(zip(classes, probas))
    
        prediction = classes[np.argmax(probas)]

        return probs_dict, prediction

    @staticmethod
    def __ensemble_median__(all_probas: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], str]:
        
        probs_final = np.median(all_probas.T, axis=0)
        
        return dict(zip(classes, probs_final)), classes[np.argmax(probs_final)]

    @staticmethod
    def __ensemble_hard__(all_probas: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], str]:
                
        votes = np.argmax(all_probas.T, axis=1)
        counts = np.bincount(votes, minlength=len(classes))
        probs_final = counts / len(votes)
        return dict(zip(classes, probs_final)), classes[np.argmax(probs_final)]
    
    
    @staticmethod
    def __ensemble_soft__(all_probas: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], str]:
        probs_final = np.mean(all_probas.T, axis=0)        
        return dict(zip(classes, probs_final)), classes[np.argmax(probs_final)]
    
    @staticmethod
    def __ensemble_borda__(all_probas: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], str]:
        all_probas_T = all_probas.T  # shape (n_modelos, n_clases)
        n_classes = all_probas_T.shape[1]
        
        ranks = np.argsort(np.argsort(-all_probas_T, axis=1), axis=1)
        
        scores = n_classes - ranks
        
        total_scores = np.sum(scores, axis=0)
        
        probs_final = total_scores / (n_classes * all_probas_T.shape[0])
        
        return dict(zip(classes, probs_final)), classes[np.argmax(total_scores)]
    
    def __forward__(self, sample: str, kw_samples: Dict[str, str], detailed: bool = False) -> Dict[str, Any]:
        
        matrix: dict = self.matrix_generators(
            sample=sample,
            kw_samples=kw_samples,
            get_sigma=self._using_sigma
        )
        results = self.predictor(
            matrix
        )[0]
        
        class_distances = results['scores']
                
        classes = list(class_distances.keys())
        
        distances_matrix = np.array([class_distances[cls] for cls in classes])
        
        all_probas = np.apply_along_axis(self.__compute_probas__, 0, distances_matrix)
        
        if self._is_ensemble:
            if self._ensemble_strategy == 'soft':
                probas, predicted_class = self.__ensemble_soft__(all_probas, classes)
            
            elif self._ensemble_strategy == 'max':
                probas, predicted_class = self.__ensemble_max__(all_probas, classes)
            
            elif self._ensemble_strategy == 'median':
                probas, predicted_class = self.__ensemble_median__(all_probas, classes)
            
            elif self._ensemble_strategy == 'hard':
                probas, predicted_class = self.__ensemble_hard__(all_probas, classes)
            
            elif self._ensemble_strategy == 'borda':
                probas, predicted_class = self.__ensemble_borda__(all_probas, classes)
                
        else:
            all_probas =  all_probas.flatten()
            probas, predicted_class = dict(zip(classes, all_probas)), classes[np.argmax(all_probas)]
        
        results['probas'] = probas
        results['predicted_class'] = predicted_class
        
        return results
    
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
            f"use_prototypes={self._use_prototypes}, "
            f"model_type='{self._model_type}', "
            f"model_kwargs={self._model_kwargs})"
        )

    def __repr__(self):
        return self.__str__()()