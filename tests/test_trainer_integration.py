# -*- coding: utf-8 -*-
"""
Tests de integración para WalkForwardTrainer y ForecastOutput.
Verifica que run_backtest() devuelve un ForecastOutput bien formado,
que las propiedades del dual-space son correctas, y que el entrenador
funciona sin W&B activo.

Todos los tests están marcados con @pytest.mark.slow porque involucran
forward passes reales del modelo Flow_FEDformer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Asegurar que el directorio raíz del proyecto está en sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers para construir el CSV sintético y la config mínima
# ---------------------------------------------------------------------------


def _crear_csv_sintetico(ruta: Path, n_filas: int = 120) -> None:
    """Crea un CSV OHLCV sintético con datos aleatorios en la ruta indicada."""
    rng = np.random.default_rng(seed=0)
    precios_base = 100.0 + rng.normal(0, 1, n_filas).cumsum()
    precios_base = np.maximum(precios_base, 10.0)  # evitar precios negativos

    df = pd.DataFrame(
        {
            "Open": precios_base + rng.uniform(-0.5, 0.5, n_filas),
            "High": precios_base + rng.uniform(0.0, 1.5, n_filas),
            "Low": precios_base - rng.uniform(0.0, 1.5, n_filas),
            "Close": precios_base,
            "Volume": rng.integers(500_000, 2_000_000, n_filas).astype(float),
        }
    )
    df.to_csv(ruta, index=False)


def _crear_config_minima(csv_path: str):
    """
    Instancia FEDformerConfig con parámetros reducidos para que los tests
    sean rápidos y no dependan de la GPU.
    """
    from config import FEDformerConfig

    return FEDformerConfig(
        target_features=["Close"],
        file_path=csv_path,
        # Secuencias cortas para reducir tiempo de cómputo
        seq_len=24,
        label_len=8,
        pred_len=4,  # par → requisito del affine coupling
        # Transformer mínimo
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=64,
        modes=4,
        # Flows mínimos
        n_flow_layers=2,
        flow_hidden_dim=32,
        # Entrenamiento de una sola época
        n_epochs_per_fold=1,
        batch_size=4,
        # Sin AMP ni compilación (evitar problemas en CPU)
        use_amp=False,
        compile_mode="",
        pin_memory=False,
        num_workers=0,
        # Sin early stopping
        patience=0,
        # Sin W&B
        wandb_project="test-integration",
    )


def _crear_dataset(config) -> "TimeSeriesDataset":  # noqa: F821
    """Crea un TimeSeriesDataset en modo 'all' usando la config proporcionada."""
    from data import TimeSeriesDataset

    return TimeSeriesDataset(config, flag="all", strict=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_backtest_returns_forecast_output(tmp_path: Path) -> None:
    """
    Verifica que run_backtest() devuelve una instancia de ForecastOutput
    con las dimensiones correctas de targets.

    Se usa n_splits=2 para ejecutar exactamente 1 fold (range(1, 2)).
    """
    from training import WalkForwardTrainer
    from training.forecast_output import ForecastOutput

    # Preparar datos sintéticos y config
    csv_path = tmp_path / "data_sintetico.csv"
    _crear_csv_sintetico(csv_path, n_filas=120)
    config = _crear_config_minima(str(csv_path))
    dataset = _crear_dataset(config)

    n_targets = len(config.target_features)  # 1 (solo "Close")

    # Instanciar el trainer con los únicos dos argumentos del constructor
    trainer = WalkForwardTrainer(config=config, full_dataset=dataset)

    # Ejecutar backtest con n_splits=2 → 1 fold efectivo
    resultado = trainer.run_backtest(n_splits=2)

    # Verificar tipo de retorno
    assert isinstance(resultado, ForecastOutput), (
        f"run_backtest() debe retornar ForecastOutput, obtuvo {type(resultado)}"
    )

    # Verificar que hay predicciones (array no vacío)
    assert resultado.preds_scaled.size > 0, (
        "preds_scaled no debe estar vacío tras un fold exitoso"
    )
    assert resultado.quantiles_scaled is not None
    assert resultado.quantiles_real is not None
    assert resultado.quantile_levels is not None
    assert resultado.quantiles_scaled.shape[0] == 3
    assert np.allclose(
        resultado.quantile_levels, np.array([0.1, 0.5, 0.9], dtype=np.float32)
    )
    assert np.allclose(resultado.preds_scaled, resultado.quantiles_scaled[1])

    # Verificar dimensión de targets en el eje -1
    assert resultado.preds_scaled.shape[-1] == n_targets, (
        f"preds_scaled.shape[-1] debe ser {n_targets}, "
        f"obtuvo {resultado.preds_scaled.shape[-1]}"
    )

    # preds_scaled y gt_scaled deben tener la misma forma
    assert resultado.gt_scaled.shape == resultado.preds_scaled.shape, (
        f"gt_scaled.shape {resultado.gt_scaled.shape} != "
        f"preds_scaled.shape {resultado.preds_scaled.shape}"
    )


@pytest.mark.slow
def test_run_backtest_respects_mc_dropout_eval_samples(tmp_path: Path) -> None:
    """run_backtest usa el knob trainer-only de MC Dropout sin romper ForecastOutput."""
    from training import WalkForwardTrainer

    csv_path = tmp_path / "data_sintetico.csv"
    _crear_csv_sintetico(csv_path, n_filas=120)
    config = _crear_config_minima(str(csv_path))
    config.mc_dropout_eval_samples = 4
    dataset = _crear_dataset(config)

    trainer = WalkForwardTrainer(config=config, full_dataset=dataset)
    resultado = trainer.run_backtest(n_splits=2)

    assert resultado.samples_scaled.shape[0] == 4
    assert resultado.quantiles_scaled is not None
    assert np.allclose(resultado.preds_scaled, resultado.quantiles_scaled[1])


@pytest.mark.slow
def test_forecast_output_properties(tmp_path: Path) -> None:
    """
    Verifica que las propiedades de ForecastOutput devuelven numpy arrays
    válidos y que los metadatos tienen los valores esperados.
    """
    from training import WalkForwardTrainer

    # Preparar datos sintéticos y config
    csv_path = tmp_path / "data_sintetico.csv"
    _crear_csv_sintetico(csv_path, n_filas=120)
    config = _crear_config_minima(str(csv_path))
    dataset = _crear_dataset(config)

    trainer = WalkForwardTrainer(config=config, full_dataset=dataset)
    resultado = trainer.run_backtest(n_splits=2)

    # preds_for_metrics debe ser numpy array
    preds_metrics = resultado.preds_for_metrics
    assert isinstance(preds_metrics, np.ndarray), (
        f"preds_for_metrics debe ser np.ndarray, obtuvo {type(preds_metrics)}"
    )

    # gt_for_metrics debe ser numpy array
    gt_metrics = resultado.gt_for_metrics
    assert isinstance(gt_metrics, np.ndarray), (
        f"gt_for_metrics debe ser np.ndarray, obtuvo {type(gt_metrics)}"
    )

    # metric_space debe ser uno de los valores válidos
    assert resultado.metric_space in ("returns", "prices"), (
        f"metric_space debe ser 'returns' o 'prices', obtuvo '{resultado.metric_space}'"
    )

    # target_names debe ser lista no vacía
    assert isinstance(resultado.target_names, list), (
        f"target_names debe ser list, obtuvo {type(resultado.target_names)}"
    )
    assert len(resultado.target_names) > 0, "target_names no debe estar vacío"

    # target_names debe contener el target configurado
    assert "Close" in resultado.target_names, (
        f"target_names debe contener 'Close', obtuvo {resultado.target_names}"
    )


@pytest.mark.slow
def test_run_backtest_no_crash_with_wandb_disabled(tmp_path: Path) -> None:
    """
    Verifica que el trainer completa run_backtest() sin lanzar excepciones
    cuando W&B está deshabilitado o no disponible.

    La config usa wandb_project genérico; si wandb no está instalado o
    la inicialización falla, _initialize_wandb() lo captura silenciosamente
    (self.wandb_run = None) y el entrenamiento continúa sin errores.
    """
    from training import WalkForwardTrainer
    from training.forecast_output import ForecastOutput

    # Preparar datos sintéticos y config (wandb_project genérico no autenticado)
    csv_path = tmp_path / "data_sintetico.csv"
    _crear_csv_sintetico(csv_path, n_filas=120)
    config = _crear_config_minima(str(csv_path))

    # Deshabilitar explícitamente cualquier integración con W&B mediante
    # la variable de entorno estándar que wandb respeta
    import os

    env_original = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "disabled"

    try:
        dataset = _crear_dataset(config)
        trainer = WalkForwardTrainer(config=config, full_dataset=dataset)

        # No debe lanzar ninguna excepción
        resultado = trainer.run_backtest(n_splits=2)

        # El resultado debe ser siempre un ForecastOutput (incluso si está vacío)
        assert isinstance(resultado, ForecastOutput), (
            f"run_backtest() debe retornar ForecastOutput incluso con W&B deshabilitado, "
            f"obtuvo {type(resultado)}"
        )
    finally:
        # Restaurar variable de entorno original
        if env_original is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = env_original


@pytest.mark.slow
def test_preds_real_equals_p50_real(tmp_path: Path) -> None:
    """Verifica invariante: preds_real == p50_real en ForecastOutput."""
    from training import WalkForwardTrainer

    csv_path = tmp_path / "data_sintetico.csv"
    _crear_csv_sintetico(csv_path, n_filas=120)
    config = _crear_config_minima(str(csv_path))
    dataset = _crear_dataset(config)

    trainer = WalkForwardTrainer(config=config, full_dataset=dataset)
    resultado = trainer.run_backtest(n_splits=2)

    assert resultado.quantiles_real is not None
    assert resultado.quantile_levels is not None
    # preds_real debe ser igual al cuantil p50 en espacio real
    np.testing.assert_array_almost_equal(
        resultado.preds_real,
        resultado.p50_real,
        decimal=6,
        err_msg="preds_real debe coincidir con p50_real",
    )


@pytest.mark.slow
def test_fold_probabilistic_metrics_populated(tmp_path: Path) -> None:
    """Verifica que trainer.fold_probabilistic_metrics tiene n_folds entradas con claves correctas."""
    from training import WalkForwardTrainer

    csv_path = tmp_path / "data_sintetico.csv"
    _crear_csv_sintetico(csv_path, n_filas=120)
    config = _crear_config_minima(str(csv_path))
    dataset = _crear_dataset(config)

    n_splits = 3  # genera 2 folds efectivos (range(1, 3))
    trainer = WalkForwardTrainer(config=config, full_dataset=dataset)
    trainer.run_backtest(n_splits=n_splits)

    expected_keys = {
        "pinball_p10",
        "pinball_p50",
        "pinball_p90",
        "coverage_80",
        "interval_width_80",
        "crps",
    }
    assert len(trainer.fold_probabilistic_metrics) == n_splits - 1, (
        f"Esperado {n_splits - 1} folds, obtenido {len(trainer.fold_probabilistic_metrics)}"
    )
    for i, fold_metrics in enumerate(trainer.fold_probabilistic_metrics):
        for key in expected_keys:
            assert key in fold_metrics, (
                f"Fold {i}: clave '{key}' faltante. Claves: {set(fold_metrics.keys())}"
            )
