# -*- coding: utf-8 -*-
"""Tests para utils/experiment_registry.py (Épica 5 — Consolidado de corridas)."""

import json

import pandas as pd
import pytest

from utils.experiment_registry import (
    build_experiment_table,
    load_portfolio_metrics,
    load_probabilistic_metrics,
    load_run_manifests,
    rank_runs,
)

# ---------------------------------------------------------------------------
# Helpers de fixtures
# ---------------------------------------------------------------------------


def _write_manifest(path, timestamp: str, ticker: str = "NVDA", seed: int = 42):
    """Escribe un manifiesto JSON sintético en el path dado."""
    manifest = {
        "timestamp": timestamp,
        "ticker": ticker,
        "seed": seed,
        "monitor_metric": "val_loss",
        "dataset_path": f"data/{ticker}_features.csv",
        "config": {"seq_len": 96, "pred_len": 20},
        "metrics": {"sharpe_ratio": 0.65, "max_drawdown": 0.12},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return manifest


def _write_prob_metrics_csv(path, timestamp: str = "", ticker: str = "NVDA"):
    """Escribe un CSV sintético de métricas probabilísticas (formato largo)."""
    rows = [
        {
            "fold": 0,
            "metric": "crps",
            "value": 0.42,
            "space": "real",
            "aggregation": "mean",
        },
        {
            "fold": 1,
            "metric": "crps",
            "value": 0.38,
            "space": "real",
            "aggregation": "mean",
        },
        {
            "fold": 0,
            "metric": "pinball_p90",
            "value": 0.11,
            "space": "real",
            "aggregation": "mean",
        },
    ]
    df = pd.DataFrame(rows)
    if timestamp:
        df["timestamp"] = timestamp
        df["ticker"] = ticker
    df.to_csv(path, index=False)
    return df


def _write_portfolio_metrics_csv(path, timestamp: str = "", ticker: str = "NVDA"):
    """Escribe un CSV sintético de métricas de portfolio."""
    rows = [
        {"fold": 0, "sharpe_ratio": 0.65, "sortino_ratio": 1.05, "max_drawdown": 0.12},
        {"fold": 1, "sharpe_ratio": 0.70, "sortino_ratio": 1.10, "max_drawdown": 0.09},
    ]
    df = pd.DataFrame(rows)
    if timestamp:
        df["timestamp"] = timestamp
        df["ticker"] = ticker
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Tests de load_run_manifests
# ---------------------------------------------------------------------------


def test_load_run_manifests_empty_dir(tmp_path):
    """Directorio sin JSONs retorna lista vacía."""
    result = load_run_manifests(tmp_path)
    assert result == [], f"Se esperaba lista vacía, se obtuvo: {result}"


def test_load_run_manifests_nonexistent_dir(tmp_path):
    """Directorio inexistente retorna lista vacía sin lanzar excepción."""
    result = load_run_manifests(tmp_path / "no_existe")
    assert result == []


def test_load_run_manifests_reads_json(tmp_path):
    """Carga un manifiesto JSON sintético correctamente."""
    json_path = tmp_path / "run_manifest_20260309_120000.json"
    expected = _write_manifest(json_path, "20260309_120000", ticker="NVDA", seed=42)

    result = load_run_manifests(tmp_path)

    assert len(result) == 1, f"Se esperaba 1 manifiesto, se obtuvieron {len(result)}"
    assert result[0]["ticker"] == "NVDA"
    assert result[0]["seed"] == 42
    assert result[0]["timestamp"] == "20260309_120000"
    assert "metrics" in result[0]
    assert result[0]["metrics"]["sharpe_ratio"] == expected["metrics"]["sharpe_ratio"]


def test_load_run_manifests_reads_multiple(tmp_path):
    """Carga múltiples manifiestos y los devuelve todos."""
    _write_manifest(
        tmp_path / "run_manifest_20260309_100000.json", "20260309_100000", "NVDA"
    )
    _write_manifest(
        tmp_path / "run_manifest_20260309_110000.json", "20260309_110000", "GOOGL"
    )

    result = load_run_manifests(tmp_path)

    assert len(result) == 2
    tickers = {m["ticker"] for m in result}
    assert tickers == {"NVDA", "GOOGL"}


# ---------------------------------------------------------------------------
# Tests de load_probabilistic_metrics
# ---------------------------------------------------------------------------


def test_load_probabilistic_metrics_empty(tmp_path):
    """Sin CSVs probabilísticos retorna DataFrame vacío."""
    result = load_probabilistic_metrics(tmp_path)
    assert isinstance(result, pd.DataFrame)
    assert result.empty, "Se esperaba DataFrame vacío"


def test_load_probabilistic_metrics_nonexistent_dir(tmp_path):
    """Directorio inexistente retorna DataFrame vacío."""
    result = load_probabilistic_metrics(tmp_path / "no_existe")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_load_probabilistic_metrics_concatenates(tmp_path):
    """Dos CSVs sintéticos son concatenados en un único DataFrame."""
    _write_prob_metrics_csv(
        tmp_path / "probabilistic_metrics_20260309_100000_NVDA.csv",
        "20260309_100000",
        "NVDA",
    )
    _write_prob_metrics_csv(
        tmp_path / "probabilistic_metrics_20260309_110000_GOOGL.csv",
        "20260309_110000",
        "GOOGL",
    )

    result = load_probabilistic_metrics(tmp_path)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Debe tener filas de ambos CSVs (3 filas cada uno = 6 total)
    assert len(result) == 6, f"Se esperaban 6 filas, se obtuvieron {len(result)}"
    # Columnas mínimas
    for col in ("fold", "metric", "value"):
        assert col in result.columns, f"Columna '{col}' faltante"


def test_load_probabilistic_metrics_single_file(tmp_path):
    """Un solo CSV es cargado correctamente con columnas esperadas."""
    _write_prob_metrics_csv(tmp_path / "probabilistic_metrics_20260309_120000_NVDA.csv")

    result = load_probabilistic_metrics(tmp_path)

    assert len(result) == 3
    assert set(result["metric"].unique()) == {"crps", "pinball_p90"}


# ---------------------------------------------------------------------------
# Tests de load_portfolio_metrics
# ---------------------------------------------------------------------------


def test_load_portfolio_metrics_basic(tmp_path):
    """Carga básica de un portfolio_metrics CSV."""
    _write_portfolio_metrics_csv(
        tmp_path / "portfolio_metrics_20260309_120000_NVDA.csv",
        "20260309_120000",
        "NVDA",
    )

    result = load_portfolio_metrics(tmp_path)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert len(result) == 2, (
        f"Se esperaban 2 filas (folds), se obtuvieron {len(result)}"
    )
    assert "sharpe_ratio" in result.columns


def test_load_portfolio_metrics_empty(tmp_path):
    """Sin archivos portfolio retorna DataFrame vacío."""
    result = load_portfolio_metrics(tmp_path)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# Tests de build_experiment_table
# ---------------------------------------------------------------------------


def test_build_experiment_table_empty_manifests():
    """Lista de manifiestos vacía retorna DataFrame vacío."""
    result = build_experiment_table([], pd.DataFrame(), pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_build_experiment_table_smoke(tmp_path):
    """Con manifiestos y métricas sintéticas construye DataFrame con al menos 1 fila."""
    # Crear manifiesto
    ts = "20260309_120000"
    manifest_path = tmp_path / f"run_manifest_{ts}.json"
    _write_manifest(manifest_path, ts, ticker="NVDA", seed=42)

    # Cargar manifiesto
    manifests = load_run_manifests(tmp_path)

    # Crear métricas probabilísticas y de portfolio sin filtrado por timestamp/ticker
    prob_df = pd.DataFrame(
        [
            {
                "fold": 0,
                "metric": "crps",
                "value": 0.42,
                "space": "real",
                "aggregation": "mean",
            },
            {
                "fold": 1,
                "metric": "crps",
                "value": 0.38,
                "space": "real",
                "aggregation": "mean",
            },
        ]
    )
    port_df = pd.DataFrame(
        [
            {"fold": 0, "sharpe_ratio": 0.65},
            {"fold": 1, "sharpe_ratio": 0.70},
        ]
    )

    result = build_experiment_table(manifests, prob_df, port_df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) >= 1, "La tabla debe tener al menos una fila"
    # Columnas básicas presentes
    for col in ("timestamp", "ticker", "seed", "monitor_metric"):
        assert col in result.columns, f"Columna '{col}' faltante"
    assert result.iloc[0]["ticker"] == "NVDA"
    assert result.iloc[0]["seed"] == 42


def test_build_experiment_table_with_empty_metrics():
    """DataFrames de métricas vacíos rellenan con NaN sin error."""
    manifests = [
        {
            "timestamp": "20260309_120000",
            "ticker": "NVDA",
            "seed": 42,
            "monitor_metric": "val_loss",
            "metrics": {"sharpe_ratio": 0.65},
        }
    ]

    result = build_experiment_table(manifests, pd.DataFrame(), pd.DataFrame())

    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "NVDA"
    assert result.iloc[0]["metric_sharpe_ratio"] == pytest.approx(0.65)


def test_build_experiment_table_multiple_runs():
    """Múltiples manifiestos generan múltiples filas."""
    manifests = [
        {
            "timestamp": "20260309_100000",
            "ticker": "NVDA",
            "seed": 42,
            "monitor_metric": "val_loss",
            "metrics": {"sharpe_ratio": 0.65},
        },
        {
            "timestamp": "20260309_110000",
            "ticker": "GOOGL",
            "seed": 123,
            "monitor_metric": "crps",
            "metrics": {"sharpe_ratio": 1.20},
        },
    ]

    result = build_experiment_table(manifests, pd.DataFrame(), pd.DataFrame())

    assert len(result) == 2
    assert set(result["ticker"].tolist()) == {"NVDA", "GOOGL"}


# ---------------------------------------------------------------------------
# Tests de rank_runs
# ---------------------------------------------------------------------------


def test_rank_runs_sorts_descending():
    """Mayor score primero con ascending=False (defecto)."""
    df = pd.DataFrame(
        {
            "timestamp": ["A", "B", "C"],
            "ticker": ["NVDA", "GOOGL", "AAPL"],
            "metric_sharpe_ratio": [0.65, 1.20, 0.30],
        }
    )

    ranked = rank_runs(df, "metric_sharpe_ratio")

    assert ranked.iloc[0]["metric_sharpe_ratio"] == pytest.approx(1.20)
    assert ranked.iloc[1]["metric_sharpe_ratio"] == pytest.approx(0.65)
    assert ranked.iloc[2]["metric_sharpe_ratio"] == pytest.approx(0.30)
    # Índice reseteado
    assert list(ranked.index) == [0, 1, 2]


def test_rank_runs_sorts_ascending():
    """Menor score primero con ascending=True (útil para métricas de error)."""
    df = pd.DataFrame(
        {
            "timestamp": ["A", "B", "C"],
            "ticker": ["NVDA", "GOOGL", "AAPL"],
            "prob_crps": [0.42, 0.30, 0.55],
        }
    )

    ranked = rank_runs(df, "prob_crps", ascending=True)

    assert ranked.iloc[0]["prob_crps"] == pytest.approx(0.30)
    assert ranked.iloc[2]["prob_crps"] == pytest.approx(0.55)


def test_rank_runs_raises_on_missing_column():
    """ValueError si la columna de score no existe en el DataFrame."""
    df = pd.DataFrame({"timestamp": ["A"], "ticker": ["NVDA"]})

    with pytest.raises(ValueError, match="metric_sharpe_ratio"):
        rank_runs(df, "metric_sharpe_ratio")
