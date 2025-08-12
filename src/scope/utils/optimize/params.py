from typing import List, Dict
from dataclasses import dataclass, field
from itertools import chain, combinations

from scope.compression.metrics import MetricType
from scope.compression.compressors import CompressorType



def all_subsets(elements: List[str]) -> List[List[str]]:
    """Genera todas las combinaciones posibles de tamaño 1 a N."""
    return list(
        map(list, chain.from_iterable(combinations(elements, r) for r in range(1, len(elements) + 1)))
    )


@dataclass
class ParameterSpace:
    compressor_names_combinations: List[List[str]] = field(
        default_factory=lambda: all_subsets([c.value for c in CompressorType])
    )

    compression_metric_names_combinations: List[List[str]] = field(
        default_factory=lambda: all_subsets([m.value for m in MetricType])
    )
    
    concat_value: List[str] = field(
        default_factory=lambda: [' ', '']
    )
    
    model_types: List[str] = field(
        default_factory=lambda: ["ot", "pd"]
    )
    
    ensemble_strategy: List[str] = field(
        default_factory= lambda: ['max', 'median', 'hard', 'borda', 'soft', None]
    )
    
    agregation_strategy: List[str] = field(
        default_factory= lambda: ["mean", "median", "min", "max", "sum", None]
    )

    # Enteros
    compression_levels: List[int] = field(
        default_factory=lambda: [9]
    )
    min_size_thresholds: List[int] = field(
        # default_factory=lambda: [0, 20, 50]
        default_factory=lambda: [0]
    )

    # Booleanos
    use_best_sigma_options: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    use_symmetric_matrix_options: List[bool] = field(
        default_factory=lambda: [True, False]
    )

    # Específicos por tipo de modelo
    ot_matching_metrics: List[str] = field(
        default_factory=lambda: ["jaccard", "dice", "overlap", None]
    )
    
    pd_distance_metrics: List[str] = field(
        default_factory=lambda: ["cosine", "minkowski", "euclidean"]
    )


   