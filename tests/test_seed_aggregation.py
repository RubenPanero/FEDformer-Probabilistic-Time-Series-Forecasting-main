# -*- coding: utf-8 -*-
"""Tests para Épica 9 — Robustez por seeds (multi-seed).

Cubre:
- aggregate_seed_metrics: agregación de manifiestos por directorio
- summarize_seed_stability: resumen estadístico de estabilidad
- run_single_seed / run_multi_seed_experiment: runner de subprocesos (mockeado)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.experiment_registry import (
    aggregate_seed_metrics,
    summarize_seed_stability,
)
from scripts.run_multi_seed import run_multi_seed_experiment, run_single_seed

# ---------------------------------------------------------------------------
# Helpers de fixtures
# ---------------------------------------------------------------------------

_EXPECTED_AGG_COLUMNS = [
    "seed",
    "sharpe",
    "sortino",
    "var_95",
    "coverage_80",
    "pinball_p50",
    "crps",
]

_EXPECTED_STABILITY_COLUMNS = ["metric", "mean", "std", "min", "max", "best", "worst"]


def _write_manifest(
    directory: Path,
    seed: int,
    timestamp: str = "20260309_120000",
    ticker: str = "NVDA",
    metrics: dict | None = None,
) -> Path:
    """Escribe un run_manifest_*.json sintético en el directorio dado."""
    if metrics is None:
        metrics = {
            "sharpe_ratio": 0.65 + seed * 0.01,
            "sortino_ratio": 1.05 + seed * 0.01,
            "var_95": 0.05,
            "coverage_80": 0.82,
            "pinball_p50": 0.03,
            "crps": 0.42,
        }
    manifest = {
        "timestamp": timestamp,
        "ticker": ticker,
        "seed": seed,
        "monitor_metric": "val_loss",
        "dataset_path": f"data/{ticker}_features.csv",
        "config": {"seq_len": 96, "pred_len": 20},
        "metrics": metrics,
    }
    path = directory / f"run_manifest_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return path


# ---------------------------------------------------------------------------
# Tests de aggregate_seed_metrics
# ---------------------------------------------------------------------------


def test_aggregate_seed_metrics_empty_returns_empty_df():
    """run_paths=[] → DataFrame vacío con las columnas correctas."""
    result = aggregate_seed_metrics([])

    assert isinstance(result, pd.DataFrame)
    assert result.empty, "Se esperaba DataFrame vacío"
    assert list(result.columns) == _EXPECTED_AGG_COLUMNS, (
        f"Columnas incorrectas: {list(result.columns)}"
    )


def test_aggregate_seed_metrics_reads_manifests(tmp_path):
    """Crear 3 directorios tmp con run_manifest_*.json sintéticos → 3 filas."""
    seeds = [42, 123, 7]
    run_dirs = []

    for seed in seeds:
        run_dir = tmp_path / f"run_seed_{seed}"
        run_dir.mkdir()
        _write_manifest(run_dir, seed=seed, timestamp=f"2026030{seed % 10}_120000")
        run_dirs.append(run_dir)

    result = aggregate_seed_metrics(run_dirs)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3, f"Se esperaban 3 filas, se obtuvieron {len(result)}"
    assert list(result.columns) == _EXPECTED_AGG_COLUMNS

    # Verificar que los seeds están presentes
    seeds_found = set(result["seed"].tolist())
    assert seeds_found == set(seeds), f"Seeds incorrectos: {seeds_found}"


def test_aggregate_seed_metrics_fills_nan_for_missing(tmp_path):
    """Un manifest sin 'sharpe' → esa fila tiene NaN en sharpe."""
    run_dir = tmp_path / "run_seed_42"
    run_dir.mkdir()

    # Manifiesto sin sharpe_ratio ni sharpe
    _write_manifest(
        run_dir,
        seed=42,
        metrics={
            "sortino_ratio": 1.05,
            "crps": 0.42,
            # sharpe ausente deliberadamente
        },
    )

    result = aggregate_seed_metrics([run_dir])

    assert len(result) == 1
    # sharpe debe ser NaN
    assert pd.isna(result.iloc[0]["sharpe"]), (
        f"Se esperaba NaN en sharpe, se obtuvo: {result.iloc[0]['sharpe']}"
    )
    # sortino sí debe estar presente
    assert not pd.isna(result.iloc[0]["sortino"]), "sortino no debería ser NaN"
    # crps sí debe estar presente
    assert not pd.isna(result.iloc[0]["crps"]), "crps no debería ser NaN"


# ---------------------------------------------------------------------------
# Tests de summarize_seed_stability
# ---------------------------------------------------------------------------


def test_summarize_seed_stability_basic():
    """DataFrame con 3 filas de métricas → resumen con mean/std/min/max/best/worst."""
    df = pd.DataFrame(
        {
            "seed": [42, 123, 7],
            "sharpe": [0.65, 0.70, 0.60],
            "sortino": [1.05, 1.10, 1.00],
        }
    )

    summary = summarize_seed_stability(df)

    assert isinstance(summary, pd.DataFrame)
    # Debe tener una fila por métrica numérica (sharpe, sortino) — seed excluido
    assert len(summary) == 2, f"Se esperaban 2 filas, se obtuvieron {len(summary)}"

    sharpe_row = summary[summary["metric"] == "sharpe"].iloc[0]
    assert sharpe_row["mean"] == pytest.approx(0.65, abs=1e-6)
    assert sharpe_row["min"] == pytest.approx(0.60, abs=1e-6)
    assert sharpe_row["max"] == pytest.approx(0.70, abs=1e-6)
    assert sharpe_row["best"] == pytest.approx(0.70, abs=1e-6)
    assert sharpe_row["worst"] == pytest.approx(0.60, abs=1e-6)


def test_summarize_seed_stability_empty_df_returns_empty():
    """DataFrame vacío → resumen vacío con columnas correctas."""
    df = pd.DataFrame(columns=_EXPECTED_AGG_COLUMNS)

    summary = summarize_seed_stability(df)

    assert isinstance(summary, pd.DataFrame)
    assert summary.empty, "Se esperaba DataFrame vacío"
    assert list(summary.columns) == _EXPECTED_STABILITY_COLUMNS


def test_summarize_seed_stability_output_columns():
    """Verificar que el output tiene las columnas requeridas."""
    df = pd.DataFrame(
        {
            "seed": [42, 123],
            "sharpe": [0.65, 0.70],
            "crps": [0.42, 0.38],
        }
    )

    summary = summarize_seed_stability(df)

    assert list(summary.columns) == _EXPECTED_STABILITY_COLUMNS, (
        f"Columnas incorrectas: {list(summary.columns)}"
    )


def test_summarize_seed_stability_raises_no_numeric_cols():
    """DataFrame sin columnas numéricas (excepto seed) → ValueError."""
    df = pd.DataFrame(
        {
            "ticker": ["NVDA", "GOOGL"],
            "status": ["ok", "ok"],
        }
    )

    with pytest.raises(ValueError, match="columnas numéricas"):
        summarize_seed_stability(df)


# ---------------------------------------------------------------------------
# Tests de run_single_seed (con mock de subprocess.run)
# ---------------------------------------------------------------------------


def test_run_single_seed_success():
    """Mockear subprocess.run con returncode=0 → result['success'] == True."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "Entrenamiento completado\n"
    mock_proc.stderr = ""

    with patch("scripts.run_multi_seed.subprocess.run", return_value=mock_proc):
        result = run_single_seed(
            csv_path="data/NVDA_features.csv",
            targets=["Close"],
            seed=42,
        )

    assert result["success"] is True, f"Se esperaba success=True, got: {result}"
    assert result["returncode"] == 0
    assert result["seed"] == 42


def test_run_single_seed_failure():
    """Mockear subprocess.run con returncode=1 → result['success'] == False, sin excepción."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout = ""
    mock_proc.stderr = "Error en el entrenamiento"

    with patch("scripts.run_multi_seed.subprocess.run", return_value=mock_proc):
        result = run_single_seed(
            csv_path="data/NVDA_features.csv",
            targets=["Close"],
            seed=123,
        )

    assert result["success"] is False, f"Se esperaba success=False, got: {result}"
    assert result["returncode"] == 1
    assert result["seed"] == 123
    # No debe lanzar excepción


# ---------------------------------------------------------------------------
# Tests de run_multi_seed_experiment (con mock de subprocess.run)
# ---------------------------------------------------------------------------


def test_run_multi_seed_experiment_calls_each_seed():
    """Mockear subprocess.run → verificar que se llamó len(seeds) veces."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "OK"
    mock_proc.stderr = ""

    seeds = [42, 123, 7]

    with patch(
        "scripts.run_multi_seed.subprocess.run", return_value=mock_proc
    ) as mock_run:
        results = run_multi_seed_experiment(
            csv_path="data/NVDA_features.csv",
            targets=["Close"],
            seeds=seeds,
        )

    assert mock_run.call_count == len(seeds), (
        f"Se esperaban {len(seeds)} llamadas, se hicieron {mock_run.call_count}"
    )
    assert len(results) == len(seeds)

    # Todos deben ser exitosos (mock returncode=0)
    for r in results:
        assert r["success"] is True


def test_run_multi_seed_experiment_returns_one_result_per_seed():
    """Cada seed genera exactamente un resultado en la lista retornada."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = ""
    mock_proc.stderr = ""

    seeds = [1, 2, 3, 4, 5]

    with patch("scripts.run_multi_seed.subprocess.run", return_value=mock_proc):
        results = run_multi_seed_experiment(
            csv_path="data/NVDA_features.csv",
            targets=["Close"],
            seeds=seeds,
        )

    assert len(results) == len(seeds)
    result_seeds = [r["seed"] for r in results]
    assert result_seeds == seeds, f"Orden de seeds incorrecto: {result_seeds}"
