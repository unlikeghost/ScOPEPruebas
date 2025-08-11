# -*- coding: utf-8 -*-

from enum import Enum
from abc import ABC, abstractmethod
from typing import Union
from dataclasses import dataclass


class MetricType(Enum):
    NCD = "ncd"
    CDM = "cdm"
    NRC = "nrc"
    CLM = "clm"


@dataclass
class CompressionData:
    """Estructura para pasar datos de compresión precalculados"""
    c_x1: int
    c_x2: int
    c_x1x2: int
    c_x2x1: int = None  # Solo para métricas que lo necesiten
    
    def __post_init__(self):
        """Validaciones básicas"""
        if any(val < 0 for val in [self.c_x1, self.c_x2, self.c_x1x2] if val is not None):
            raise ValueError("Compression lengths cannot be negative")


class _BaseMetric(ABC):
    
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metric='{self.name}')"
    
    def __str__(self):
        return self.__repr__()
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def compute(self, compression_data: CompressionData) -> float:
        raise NotImplementedError()
    
    @staticmethod
    def _safe_divide(numerator: float, denominator: float, metric_name: str) -> float:
        if denominator == 0:
            raise ZeroDivisionError(f"Denominator in {metric_name} is zero.")
        return numerator / denominator


class NCDMetric(_BaseMetric):
    name = "NCD"
    
    def compute(self, compression_data: CompressionData) -> float:
        c_x1, c_x2, c_x1x2 = compression_data.c_x1, compression_data.c_x2, compression_data.c_x1x2
        
        numerator = c_x1x2 - min(c_x1, c_x2)
        denominator = max(c_x1, c_x2)
        
        return self._safe_divide(numerator, denominator, self.name)


class CDMMetric(_BaseMetric):
    name = "CDM"
    
    def compute(self, compression_data: CompressionData) -> float:
        c_x1, c_x2, c_x1x2 = compression_data.c_x1, compression_data.c_x2, compression_data.c_x1x2
        
        denominator = c_x1 + c_x2
        
        return self._safe_divide(c_x1x2, denominator, self.name)


class NRCMetric(_BaseMetric):
    name = "NRC"
    
    def compute(self, compression_data: CompressionData) -> float:
        c_x1, c_x1x2 = compression_data.c_x1, compression_data.c_x1x2
        
        return self._safe_divide(c_x1x2, c_x1, self.name)


class CLMMetric(_BaseMetric):
    name = "CLM"
    
    def compute(self, compression_data: CompressionData) -> float:
        c_x1, c_x2, c_x1x2 = compression_data.c_x1, compression_data.c_x2, compression_data.c_x1x2
        
        numerator = 1 - (c_x1 + c_x2 - c_x1x2)
        
        return self._safe_divide(numerator, c_x1x2, self.name)


METRIC_STRATEGIES = {
    MetricType.NCD: NCDMetric,
    MetricType.NRC: NRCMetric,
    MetricType.CDM: CDMMetric,
    MetricType.CLM: CLMMetric,
}


def get_metric(name: Union[str, MetricType]) -> _BaseMetric:
    """Factory function para crear métricas"""
    if isinstance(name, str):
        try:
            metric_enum = MetricType(name.lower())
        except ValueError:
            allowed = sorted(m.value for m in MetricType)
            raise ValueError(
                f"'{name}' is not a valid metric name. "
                f"Expected one of: {', '.join(allowed)}"
            )
    elif isinstance(name, MetricType):
        metric_enum = name
    else:
        raise TypeError("Expected 'name' to be str or MetricType.")
    
    metric_class = METRIC_STRATEGIES[metric_enum]
    return metric_class()


