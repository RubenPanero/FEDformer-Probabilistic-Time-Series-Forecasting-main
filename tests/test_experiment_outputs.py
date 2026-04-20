# -*- coding: utf-8 -*-
"""Tests para utils/io_experiment.py (PR 4 — Artefactos experimentales)."""

import json

import numpy as np
import pandas as pd

from utils.io_experiment import (
    build_run_manifest,
    save_fold_metrics,
    save_probabilistic_metrics,
    save_run_manifest,
    serialize_config,
)


def _make_fold_prob_metrics(n_folds: int = 3) -> list[dict[str, float]]:
    """Factory de métricas probabilísticas por fold."""
    keys = [
        "pinball_p10",
        "pinball_p50",
        "pinball_p90",
        "coverage_80",
        "interval_width_80",
        "crps",
    ]
    rng = np.random.default_rng(0)
    return [{k: float(rng.uniform(0.0, 1.0)) for k in keys} for _ in range(n_folds)]


def test_serialize_config_is_json_serializable(config):
    """serialize_config devuelve un dict JSON-serializable con campos clave."""
    result = serialize_config(config)

    # Debe ser JSON-serializable (no debe lanzar)
    encoded = json.dumps(result)
    decoded = json.loads(encoded)

    # Los campos más relevantes deben estar presentes
    for field in ("seq_len", "pred_len", "batch_size", "seed"):
        assert field in decoded, f"Campo '{field}' faltante en serialize_config"

    # Todos los valores deben ser tipos Python nativos
    for key, val in decoded.items():
        assert isinstance(val, (int, float, str, bool, list, type(None))), (
            f"Campo '{key}' tiene tipo no JSON-nativo: {type(val)}"
        )


def test_run_manifest_contains_expected_fields(config, tmp_path):
    """build_run_manifest y save_run_manifest generan el JSON correcto."""
    metrics_agg = {"sharpe_ratio": 0.65, "max_drawdown": 0.12, "var_95": 0.05}
    manifest = build_run_manifest(
        config=config,
        ticker="NVDA",
        metrics_agg=metrics_agg,
        monitor_metric="val_loss",
        seed=42,
        dataset_path="data/NVDA_features.csv",
        timestamp="20260309_120000",
    )

    # Campos obligatorios en el manifiesto
    for field in (
        "timestamp",
        "ticker",
        "dataset_path",
        "seed",
        "monitor_metric",
        "config",
        "metrics",
    ):
        assert field in manifest, f"Campo '{field}' faltante en manifiesto"

    assert manifest["ticker"] == "NVDA"
    assert manifest["seed"] == 42
    assert "sharpe_ratio" in manifest["metrics"]

    # save_run_manifest debe crear el archivo JSON
    path = save_run_manifest(manifest, tmp_path, "20260309_120000")
    assert path.exists(), "El archivo JSON del manifiesto debe existir"
    assert path.name == "run_manifest_20260309_120000.json"

    # El JSON guardado debe ser válido y contener los mismos datos
    with open(path) as fh:
        loaded = json.load(fh)
    assert loaded["ticker"] == "NVDA"
    assert "config" in loaded


def test_probabilistic_metrics_csv_shape(tmp_path):
    """save_probabilistic_metrics crea un CSV con las columnas y filas correctas."""
    n_folds = 3
    prob_metrics = _make_fold_prob_metrics(n_folds)
    n_metrics_per_fold = len(prob_metrics[0])

    path = save_probabilistic_metrics(prob_metrics, tmp_path, "20260309_120000", "NVDA")

    assert path.exists(), "El CSV de métricas probabilísticas debe existir"
    assert path.name == "probabilistic_metrics_20260309_120000_NVDA.csv"

    df = pd.read_csv(path)
    # Columnas esperadas
    for col in ("fold", "metric", "value", "space", "aggregation"):
        assert col in df.columns, f"Columna '{col}' faltante en el CSV"

    # Una fila por (fold × métrica)
    assert len(df) == n_folds * n_metrics_per_fold, (
        f"Esperadas {n_folds * n_metrics_per_fold} filas, obtenidas {len(df)}"
    )

    # Los folds deben ser 0-indexed
    assert set(df["fold"].unique()) == set(range(n_folds))


def test_fold_metrics_csv_contains_fold_rows(tmp_path):
    """save_fold_metrics crea un CSV en formato ancho con una fila por fold."""
    n_folds = 4
    fold_metrics = _make_fold_prob_metrics(n_folds)

    path = save_fold_metrics(fold_metrics, tmp_path, "20260309_120000", "NVDA")

    assert path.exists(), "El CSV de métricas por fold debe existir"
    assert path.name == "fold_metrics_20260309_120000_NVDA.csv"

    df = pd.read_csv(path)

    # Una fila por fold
    assert len(df) == n_folds, f"Esperadas {n_folds} filas, obtenidas {len(df)}"

    # Columna fold presente
    assert "fold" in df.columns
    assert list(df["fold"]) == list(range(n_folds))

    # Las métricas del primer fold deben estar como columnas
    for metric in fold_metrics[0]:
        assert metric in df.columns, f"Métrica '{metric}' faltante en CSV"
