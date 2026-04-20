"""Tests para validate_forecast — métricas de validación probabilística."""

import numpy as np
import pandas as pd
import pytest

from scripts.validate_forecast import (
    compute_coverage,
    directional_accuracy,
    interval_score,
    mae,
    pinball_loss,
)


# ── compute_coverage ─────────────────────────────────────────────────────────


def test_coverage_todos_dentro():
    gt = pd.Series([5.0, 6.0, 7.0])
    lower = pd.Series([4.0, 5.0, 6.0])
    upper = pd.Series([6.0, 7.0, 8.0])
    assert compute_coverage(gt, lower, upper) == pytest.approx(1.0)


def test_coverage_ninguno_dentro():
    gt = pd.Series([10.0, 10.0, 10.0])
    lower = pd.Series([4.0, 5.0, 6.0])
    upper = pd.Series([6.0, 7.0, 8.0])
    assert compute_coverage(gt, lower, upper) == pytest.approx(0.0)


def test_coverage_mitad():
    gt = pd.Series([5.0, 10.0])  # primero dentro, segundo fuera
    lower = pd.Series([4.0, 4.0])
    upper = pd.Series([6.0, 6.0])
    assert compute_coverage(gt, lower, upper) == pytest.approx(0.5)


def test_coverage_en_borde_incluido():
    """Los bordes exactos del intervalo cuentan como 'dentro'."""
    gt = pd.Series([4.0, 6.0])
    lower = pd.Series([4.0, 4.0])
    upper = pd.Series([6.0, 6.0])
    assert compute_coverage(gt, lower, upper) == pytest.approx(1.0)


# ── pinball_loss ──────────────────────────────────────────────────────────────


def test_pinball_perfecto_es_cero():
    gt = np.array([1.0, 2.0, 3.0])
    q_hat = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(gt, q_hat, 0.5) == pytest.approx(0.0)


def test_pinball_mediana_igual_a_mae_mitad():
    """pinball(q=0.5) == MAE / 2."""
    gt = np.array([0.0, 0.0, 0.0])
    q_hat = np.array([1.0, 2.0, 3.0])
    expected = np.mean([1.0, 2.0, 3.0]) / 2  # 1.0
    assert pinball_loss(gt, q_hat, 0.5) == pytest.approx(expected)


def test_pinball_sobreestimacion_penaliza_con_1_menos_q():
    """Cuando q_hat > gt, penalidad es (1-q)*(q_hat-gt)."""
    gt = np.array([0.0])
    q_hat = np.array([1.0])
    # q=0.1: penalidad = (1-0.1)*1.0 = 0.9
    assert pinball_loss(gt, q_hat, 0.1) == pytest.approx(0.9)


def test_pinball_subestimacion_penaliza_con_q():
    """Cuando q_hat < gt, penalidad es q*(gt-q_hat)."""
    gt = np.array([1.0])
    q_hat = np.array([0.0])
    # q=0.9: penalidad = 0.9*1.0 = 0.9
    assert pinball_loss(gt, q_hat, 0.9) == pytest.approx(0.9)


# ── mae ───────────────────────────────────────────────────────────────────────


def test_mae_perfecto():
    assert mae(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)


def test_mae_basico():
    gt = np.array([100.0, 200.0])
    p50 = np.array([90.0, 210.0])
    assert mae(gt, p50) == pytest.approx(10.0)


# ── interval_score ────────────────────────────────────────────────────────────


def test_interval_score_perfecto_es_solo_anchura():
    """Si gt dentro del intervalo, IS = (upper - lower)."""
    gt = np.array([5.0])
    lower = np.array([4.0])
    upper = np.array([6.0])
    alpha = 0.20  # intervalo 80%
    expected = 2.0  # solo anchura
    assert interval_score(gt, lower, upper, alpha) == pytest.approx(expected)


def test_interval_score_penaliza_miss_por_abajo():
    """GT por debajo del intervalo añade penalización (2/alpha)*(lower-gt)."""
    gt = np.array([2.0])
    lower = np.array([4.0])
    upper = np.array([6.0])
    alpha = 0.20
    # anchura=2, penalización=(2/0.2)*(4-2)=20 → total=22
    assert interval_score(gt, lower, upper, alpha) == pytest.approx(22.0)


def test_interval_score_penaliza_miss_por_arriba():
    """GT por encima del intervalo añade penalización (2/alpha)*(gt-upper)."""
    gt = np.array([8.0])
    lower = np.array([4.0])
    upper = np.array([6.0])
    alpha = 0.20
    # anchura=2, penalización=(2/0.2)*(8-6)=20 → total=22
    assert interval_score(gt, lower, upper, alpha) == pytest.approx(22.0)


# ── directional_accuracy ──────────────────────────────────────────────────────


def test_directional_accuracy_perfecto():
    last_price = np.array([100.0, 100.0])
    gt_next = np.array([105.0, 95.0])  # sube, baja
    p50_next = np.array([103.0, 92.0])  # predice sube, baja
    assert directional_accuracy(gt_next, p50_next, last_price) == pytest.approx(1.0)


def test_directional_accuracy_cero():
    last_price = np.array([100.0, 100.0])
    gt_next = np.array([105.0, 95.0])  # sube, baja
    p50_next = np.array([95.0, 105.0])  # predice baja, sube
    assert directional_accuracy(gt_next, p50_next, last_price) == pytest.approx(0.0)


def test_directional_accuracy_mitad():
    last_price = np.array([100.0, 100.0])
    gt_next = np.array([105.0, 95.0])
    p50_next = np.array([103.0, 105.0])  # primero ok, segundo mal
    assert directional_accuracy(gt_next, p50_next, last_price) == pytest.approx(0.5)
