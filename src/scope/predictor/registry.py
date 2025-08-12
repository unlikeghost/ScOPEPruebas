from scope.predictor.base import _BasePredictor
from scope.predictor.ot import ScOPEOT
from scope.predictor.pd import ScOPEPD

from typing import List, Dict, Any, Type


class PredictorRegistry:
    
    _predictors: Dict[str, Type[_BasePredictor]] = {}
    _defaults: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[_BasePredictor], defaults: Dict[str, Any] = None):
        cls._predictors[name] = model_class
        cls._defaults[name] = defaults or {}
    
    @classmethod
    def create(cls, name: str, epsilon: float = 1e-12, use_prototypes: bool = False, **kwargs) -> _BasePredictor:
        if name not in cls._predictors:
            available = list(cls._predictors.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        
        config = cls._defaults[name].copy()
        
        config['epsilon'] = epsilon
        config['use_prototypes'] = use_prototypes
        
        config.update(kwargs)
        return cls._predictors[name](**config)
    
    @classmethod
    def available(cls) -> List[str]:
        return list(cls._predictors.keys())
    

PredictorRegistry.register(
    name="ot",
    model_class=ScOPEOT,
)

PredictorRegistry.register(
    name="pd",
    model_class=ScOPEPD,
)