# -*- coding: utf-8 -*-
"""Tests para utils/probabilistic_metrics.py."""

import numpy as np
import pytest

from utils.probabilistic_metrics import (
    calibration_gap,
    coverage_by_quantile_pair,
    crps_from_samples,
    empirical_coverage,
    interval_width,
    multi_quantile_pinball_loss,
    pinball_loss,
)

# Dimensiones comunes para los tests
N_WIN, PRED, N_TGT = 8, 5, 1


def _make_data(seed: int = 0) -> tuple:
    """Factory de datos de prueba."""
    rng = np.random.default_rng(seed)
    y_true = rng.normal(0, 1, (N_WIN, PRED, N_TGT))
    levels = np.array([0.1, 0.5, 0.9])
    quantiles = np.sort(rng.normal(0, 1, (3, N_WIN, PRED, N_TGT)), axis=0)
    samples = rng.normal(0, 1, (100, N_WIN, PRED, N_TGT))
    return y_true, quantiles, levels, samples


def test_pinball_loss_scalar_case():
    """pinball_loss reproduce la fórmula analítica para un caso escalar conocido."""
    # y_true=1, y_pred=0, q=0.9 → diff=1>0 → loss = q*diff = 0.9
    y_true = np.array([[[1.0]]])
    y_pred = np.array([[[0.0]]])
    result = pinball_loss(y_true, y_pred, 0.9)
    assert abs(result - 0.9) < 1e-7, f"Esperado 0.9, obtenido {result}"

    # y_true=0, y_pred=1, q=0.1 → diff=-1<0 → loss = (q-1)*diff = (-0.9)*(-1) = 0.9
    y_true2 = np.array([[[0.0]]])
    y_pred2 = np.array([[[1.0]]])
    result2 = pinball_loss(y_true2, y_pred2, 0.1)
    assert abs(result2 - 0.9) < 1e-7, f"Esperado 0.9, obtenido {result2}"


def test_pinball_loss_multidim_case():
    """pinball_loss es broadcastable y retorna escalar no negativo para arrays multidim."""
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, (N_WIN, PRED, N_TGT))
    y_pred = rng.normal(0, 1, (N_WIN, PRED, N_TGT))
    result = pinball_loss(y_true, y_pred, 0.5)
    assert isinstance(result, float), "Debe retornar escalar Python float"
    assert result >= 0.0, "Pinball loss no puede ser negativa"

    # La pinball p50 es MAE/2 por definición
    mae_half = float(np.mean(np.abs(y_true - y_pred))) / 2.0
    assert abs(result - mae_half) < 1e-6, (
        f"Pinball p50 debe ser MAE/2: {result:.6f} vs {mae_half:.6f}"
    )


def test_empirical_coverage_basic():
    """empirical_coverage retorna 1.0 cuando todos los puntos están dentro del intervalo."""
    y_true = np.zeros((N_WIN, PRED, N_TGT))
    lower = -np.ones((N_WIN, PRED, N_TGT))
    upper = np.ones((N_WIN, PRED, N_TGT))
    assert abs(empirical_coverage(y_true, lower, upper) - 1.0) < 1e-9

    # Cobertura 0 cuando todos están fuera
    lower_out = np.full_like(y_true, 5.0)
    upper_out = np.full_like(y_true, 10.0)
    assert abs(empirical_coverage(y_true, lower_out, upper_out) - 0.0) < 1e-9


def test_interval_width_basic():
    """interval_width retorna la anchura media correcta."""
    lower = np.zeros((N_WIN, PRED, N_TGT))
    upper = np.full((N_WIN, PRED, N_TGT), 2.0)
    assert abs(interval_width(lower, upper) - 2.0) < 1e-9

    # Anchura cero (intervalos degenerados)
    assert abs(interval_width(lower, lower) - 0.0) < 1e-9


def test_calibration_gap_perfect_case():
    """Con cuantiles exactamente calibrados (cuantiles empíricos), gap ≈ 0."""
    n_windows = 1000
    # Distribución uniforme determinista en [0,1]: cuantiles teóricos conocidos
    y_true = np.linspace(0.0, 1.0, n_windows).reshape(n_windows, 1, 1)
    levels = np.array([0.1, 0.5, 0.9])

    # Para uniforme [0,1], el cuantil q es exactamente q
    theoretical_q = levels  # Q(0.1)=0.1, Q(0.5)=0.5, Q(0.9)=0.9
    quantiles = np.broadcast_to(
        theoretical_q[:, np.newaxis, np.newaxis, np.newaxis],
        (3, n_windows, 1, 1),
    ).copy()

    gaps = calibration_gap(y_true, quantiles, levels)
    assert set(gaps.keys()) == {
        "calibration_gap_p10",
        "calibration_gap_p50",
        "calibration_gap_p90",
    }
    for key, val in gaps.items():
        assert val < 0.002, f"{key}={val:.4f} debería ser ≈0 con cuantiles perfectos"


def test_crps_from_samples_basic():
    """Para samples tipo Dirac (todas iguales a y_pred), CRPS = MAE."""
    rng = np.random.default_rng(0)
    n_windows, pred_len, n_targets = 10, 5, 1
    y_true = rng.normal(0, 1, (n_windows, pred_len, n_targets))
    y_pred = rng.normal(0, 1, (n_windows, pred_len, n_targets))

    # Dirac: todas las n_samples muestras son iguales a y_pred
    n_samples = 100
    samples = np.repeat(y_pred[np.newaxis], n_samples, axis=0)

    crps = crps_from_samples(y_true, samples)
    mae = float(np.mean(np.abs(y_pred - y_true)))

    # CRPS de distribución Dirac en y_pred = |y_pred - y_true| = MAE
    assert abs(crps - mae) < 1e-5, f"CRPS Dirac={crps:.6f}, MAE={mae:.6f}"


def test_multi_quantile_pinball_loss_keys():
    """multi_quantile_pinball_loss retorna las claves correctas para levels=[0.1,0.5,0.9]."""
    y_true, quantiles, levels, _ = _make_data()
    result = multi_quantile_pinball_loss(y_true, quantiles, levels)
    assert set(result.keys()) == {"pinball_p10", "pinball_p50", "pinball_p90"}
    for v in result.values():
        assert isinstance(v, float)
        assert v >= 0.0


def test_coverage_by_quantile_pair_raises_on_missing_level():
    """coverage_by_quantile_pair lanza ValueError si el nivel no existe."""
    y_true, quantiles, levels, _ = _make_data()
    with pytest.raises(ValueError, match="no encontrado"):
        coverage_by_quantile_pair(y_true, quantiles, levels, 0.05, 0.95)
