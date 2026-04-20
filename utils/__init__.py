"""
Utilidades generales para el sistema FEDformer.
"""

from .metrics import MetricsTracker
from .probabilistic_metrics import (
    pinball_loss,
    multi_quantile_pinball_loss,
    empirical_coverage,
    interval_width,
    calibration_gap,
    coverage_by_quantile_pair,
    crps_from_samples,
    sharpness_from_quantiles,
)
from .helpers import _select_amp_dtype, setup_cuda_optimizations, get_device
from .calibration import conformal_quantile, apply_conformal_interval
from .model_registry import (
    load_registry,
    save_registry,
    register_specialist,
    get_specialist,
    list_specialists,
)

__all__ = [
    "MetricsTracker",
    "_select_amp_dtype",
    "setup_cuda_optimizations",
    "get_device",
    "conformal_quantile",
    "apply_conformal_interval",
    "load_registry",
    "save_registry",
    "register_specialist",
    "get_specialist",
    "list_specialists",
    # Métricas probabilísticas
    "pinball_loss",
    "multi_quantile_pinball_loss",
    "empirical_coverage",
    "interval_width",
    "calibration_gap",
    "coverage_by_quantile_pair",
    "crps_from_samples",
    "sharpness_from_quantiles",
]
