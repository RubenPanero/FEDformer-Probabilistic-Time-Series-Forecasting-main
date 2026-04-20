# -*- coding: utf-8 -*-
"""
Utilidades de entrada/salida para artefactos experimentales.

Funciones para serializar configuraciones, construir manifiestos de run
y exportar métricas probabilísticas y por fold como CSVs/JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from config import FEDformerConfig

logger = logging.getLogger(__name__)

# Campos de configuración exportados al manifiesto
_CONFIG_EXPORT_FIELDS = (
    "seq_len",
    "pred_len",
    "label_len",
    "d_model",
    "n_heads",
    "e_layers",
    "d_layers",
    "batch_size",
    "n_epochs_per_fold",
    "return_transform",
    "metric_space",
    "monitor_metric",
    "monitor_mode",
    "patience",
    "min_delta",
    "gradient_clip_norm",
    "seed",
)


def _to_python(val: Any) -> Any:
    """Convierte tipos numpy a equivalentes Python nativos para serialización JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_to_python(v) for v in val]
    return val


def serialize_config(config: FEDformerConfig) -> dict[str, Any]:
    """Convierte FEDformerConfig a dict JSON-serializable.

    Solo exporta los campos más relevantes para reproducir el experimento.
    No incluye tensores, callables ni objetos no serializables.

    Args:
        config: Configuración del experimento.

    Returns:
        Dict con tipos Python nativos, verificado como JSON-serializable.
    """
    result: dict[str, Any] = {}
    for field in _CONFIG_EXPORT_FIELDS:
        try:
            val = getattr(config, field, None)
            result[field] = _to_python(val)
        except Exception:  # noqa: BLE001
            pass
    # Verificar serializabilidad antes de devolver
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:  # pragma: no cover
        logger.warning(
            "serialize_config: algunos campos no serializables omitidos: %s", exc
        )
    return result


def build_run_manifest(
    config: FEDformerConfig,
    ticker: str,
    metrics_agg: dict[str, float],
    monitor_metric: str,
    seed: int,
    dataset_path: str,
    timestamp: str,
) -> dict[str, Any]:
    """Construye el manifiesto completo de un run experimental.

    Args:
        config: Configuración del experimento.
        ticker: Identificador del ticker procesado.
        metrics_agg: Métricas agregadas del run (p.ej. Sharpe, VaR, etc.).
        monitor_metric: Métrica usada para checkpoint selection.
        seed: Semilla aleatoria del run.
        dataset_path: Ruta al CSV de datos.
        timestamp: Marca temporal del run (YYYYMMDD_HHMMSS).

    Returns:
        Dict con todos los metadatos del run, JSON-serializable.
    """
    return {
        "timestamp": timestamp,
        "ticker": ticker,
        "dataset_path": dataset_path,
        "seed": seed,
        "monitor_metric": monitor_metric,
        "config": serialize_config(config),
        "metrics": {k: _to_python(v) for k, v in metrics_agg.items()},
    }


def save_run_manifest(
    manifest: dict[str, Any], results_dir: Path, timestamp: str
) -> Path:
    """Guarda el manifiesto del run como JSON en results_dir.

    Args:
        manifest: Dict construido con build_run_manifest.
        results_dir: Directorio de destino.
        timestamp: Marca temporal (usado en el nombre del archivo).

    Returns:
        Ruta al archivo JSON generado.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"run_manifest_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    logger.info("Manifiesto del run exportado a: %s", path)
    return path


def save_probabilistic_metrics(
    prob_metrics: list[dict[str, float]],
    results_dir: Path,
    timestamp: str,
    ticker: str,
) -> Path:
    """Exporta métricas probabilísticas por fold en formato largo.

    Columnas del CSV: fold, metric, value, space, aggregation.

    Args:
        prob_metrics: Lista de dicts con métricas por fold (de trainer.fold_probabilistic_metrics).
        results_dir: Directorio de destino.
        timestamp: Marca temporal.
        ticker: Nombre del ticker (incluido en el nombre del archivo).

    Returns:
        Ruta al CSV generado.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for fold_idx, fold_dict in enumerate(prob_metrics):
        for metric, value in fold_dict.items():
            rows.append(
                {
                    "fold": fold_idx,
                    "metric": metric,
                    "value": _to_python(value),
                    "space": "real",
                    "aggregation": "mean",
                }
            )
    path = results_dir / f"probabilistic_metrics_{timestamp}_{ticker}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Métricas probabilísticas exportadas a: %s", path)
    return path


def save_fold_metrics(
    fold_metrics: list[dict[str, float]],
    results_dir: Path,
    timestamp: str,
    ticker: str,
) -> Path:
    """Exporta métricas por fold en formato ancho.

    Columnas del CSV: fold + todas las métricas disponibles.
    Métricas ausentes en algún fold aparecen como NaN.

    Args:
        fold_metrics: Lista de dicts con métricas por fold.
        results_dir: Directorio de destino.
        timestamp: Marca temporal.
        ticker: Nombre del ticker.

    Returns:
        Ruta al CSV generado.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"fold": i, **{k: _to_python(v) for k, v in d.items()}}
        for i, d in enumerate(fold_metrics)
    ]
    path = results_dir / f"fold_metrics_{timestamp}_{ticker}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Métricas por fold exportadas a: %s", path)
    return path
