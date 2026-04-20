# -*- coding: utf-8 -*-
"""
ForecastOutput: contenedor dual-space para predicciones escaladas y reales.
Permite transportar predicciones en espacio escalado (raw del modelo) y en
espacio real (precios o retornos no escalados) a través del pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ForecastOutput:
    """Contenedor dual-space para predicciones, muestras y cuantiles.

    La clase preserva tanto el espacio escalado del modelo como el espacio real
    usado por metricas e interpretacion. `preds_real` sigue siendo la ruta p50
    de compatibilidad, mientras que los cuantiles explicitos viven en
    `quantiles_scaled`, `quantiles_real` y `quantile_levels`.
    """

    # Espacio escalado (raw del modelo)
    preds_scaled: np.ndarray  # (n_windows, pred_len, n_targets)
    gt_scaled: np.ndarray  # (n_windows, pred_len, n_targets)
    samples_scaled: np.ndarray  # (n_samples, n_windows, pred_len, n_targets) o similar

    # Espacio real (precios o retornos no escalados)
    preds_real: np.ndarray
    gt_real: np.ndarray
    samples_real: np.ndarray
    quantiles_scaled: np.ndarray | None = (
        None  # (n_quantiles, n_windows, pred_len, n_targets)
    )
    quantiles_real: np.ndarray | None = None
    quantile_levels: np.ndarray | None = None  # (n_quantiles,)

    # Metadatos
    metric_space: str = "returns"  # "returns" | "prices"
    return_transform: str = "none"  # "none" | "log_return" | "simple_return"
    target_names: list[str] | None = None
    # Índice de fold por ventana — shape (n_windows,), dtype int32; None si no disponible
    window_fold_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.target_names is None:
            self.target_names = []

    @property
    def preds_for_metrics(self) -> np.ndarray:
        """Predicciones desescaladas para métricas financieras.

        Siempre devuelve preds_real: el espacio interpretable (retornos o precios
        dependiendo de return_transform). metric_space controla qué contiene
        preds_real (vía _inverse_transform_all), no qué array se selecciona aquí.
        """
        return self.preds_real

    @property
    def gt_for_metrics(self) -> np.ndarray:
        """Ground truth desescalado para métricas financieras."""
        return self.gt_real

    @property
    def samples_for_metrics(self) -> np.ndarray:
        """Muestras desescaladas para métricas financieras."""
        return self.samples_real

    @property
    def quantiles_for_metrics(self) -> np.ndarray | None:
        """Cuantiles desescalados para métricas financieras, si están disponibles."""
        return self.quantiles_real

    # ------------------------------------------------------------------
    # Helpers de cuantil por nivel
    # ------------------------------------------------------------------

    def _get_quantile_index(self, level: float) -> int:
        """Índice del nivel de cuantil en quantile_levels.

        Args:
            level: Nivel del cuantil (p.ej. 0.1, 0.5, 0.9).

        Raises:
            ValueError: Si quantile_levels es None o el nivel no está disponible.
        """
        if self.quantile_levels is None:
            raise ValueError(
                "quantile_levels no está disponible en este ForecastOutput."
            )
        idx = int(np.argmin(np.abs(self.quantile_levels.astype(float) - level)))
        if abs(float(self.quantile_levels[idx]) - level) > 1e-6:
            raise ValueError(
                f"Nivel de cuantil {level} no disponible. "
                f"Niveles: {list(self.quantile_levels)}"
            )
        return idx

    def get_quantile(self, level: float, real: bool = True) -> np.ndarray:
        """Devuelve el cuantil para el nivel especificado.

        Args:
            level: Nivel del cuantil (p.ej. 0.1, 0.5, 0.9).
            real: Si True devuelve del espacio real; si False del espacio escalado.

        Returns:
            Array de cuantiles, shape (n_windows, pred_len, n_targets).

        Raises:
            ValueError: Si quantiles_real/scaled es None o level no está disponible.
        """
        idx = self._get_quantile_index(level)
        arr = self.quantiles_real if real else self.quantiles_scaled
        if arr is None:
            space = "real" if real else "scaled"
            raise ValueError(
                f"quantiles_{space} no está disponible en este ForecastOutput."
            )
        return arr[idx]

    @property
    def p10_real(self) -> np.ndarray:
        """Cuantil p10 en espacio real."""
        return self.get_quantile(0.1, real=True)

    @property
    def p50_real(self) -> np.ndarray:
        """Cuantil p50 en espacio real (equivalente a preds_real)."""
        return self.get_quantile(0.5, real=True)

    @property
    def p90_real(self) -> np.ndarray:
        """Cuantil p90 en espacio real."""
        return self.get_quantile(0.9, real=True)

    @property
    def p10_scaled(self) -> np.ndarray:
        """Cuantil p10 en espacio escalado."""
        return self.get_quantile(0.1, real=False)

    @property
    def p50_scaled(self) -> np.ndarray:
        """Cuantil p50 en espacio escalado."""
        return self.get_quantile(0.5, real=False)

    @property
    def p90_scaled(self) -> np.ndarray:
        """Cuantil p90 en espacio escalado."""
        return self.get_quantile(0.9, real=False)
