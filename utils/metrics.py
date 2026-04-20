# -*- coding: utf-8 -*-
"""
Sistema de seguimiento analítico para el ciclo de vida del entrenamiento.
Refactorizado a Python 3.10+ PEP 8.
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Monitor temporal in-memory local de logs algorítmicos."""

    def __init__(self) -> None:
        # Formato interno: (fold, step, value) — permite reconstruir curvas por fold y época
        self.metrics: dict[str, list[tuple[int, int, float]]] = defaultdict(list)

    def log_metrics(self, metrics: dict[str, Any], step: int, fold: int = 0) -> None:
        """Adiciona métricas para su correlación en base al fold y bloque (step) en ejecución."""
        for key, value in metrics.items():
            self.metrics[key].append((fold, step, value))
            logger.info(
                "Fold %s Iteración Computada %s - [%s]: %.4f", fold, step, key, value
            )

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Devuelve un objeto paramétrico purgado con la consolidación global histórica."""
        summary: dict[str, dict[str, float]] = {}
        for key, values in self.metrics.items():
            vals = [float(v[2]) for v in values]
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Exporta el histórico como DataFrame con columnas fold, epoch, metric, value.

        Útil para persistir curvas de entrenamiento por fold a CSV.
        """
        rows = []
        for metric_name, entries in self.metrics.items():
            for fold, epoch, value in entries:
                rows.append(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "metric": metric_name,
                        "value": value,
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["fold", "epoch", "metric", "value"])
        return pd.DataFrame(rows)
