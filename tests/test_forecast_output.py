# -*- coding: utf-8 -*-
"""Tests para ForecastOutput: dual-space predictions y backward compat."""

import numpy as np
import pytest
from training.forecast_output import ForecastOutput
from simulations import RiskSimulator, PortfolioSimulator


def _make_forecast(
    metric_space: str = "returns", return_transform: str = "none"
) -> ForecastOutput:
    """Factory de ForecastOutput para tests."""
    n_windows, pred_len, n_targets = 10, 5, 1
    n_samples = 50
    rng = np.random.default_rng(42)
    preds_scaled = rng.normal(0, 1, (n_windows, pred_len, n_targets))
    gt_scaled = rng.normal(0, 1, (n_windows, pred_len, n_targets))
    samples_scaled = rng.normal(0, 1, (n_samples, n_windows, pred_len, n_targets))
    quantile_levels = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    quantiles_scaled = np.quantile(samples_scaled, quantile_levels, axis=0)
    # Espacio real: versión escalada x 100 (simulando inverse_transform)
    preds_real = preds_scaled * 100
    gt_real = gt_scaled * 100
    samples_real = samples_scaled * 100
    quantiles_real = quantiles_scaled * 100
    return ForecastOutput(
        preds_scaled=preds_scaled,
        gt_scaled=gt_scaled,
        samples_scaled=samples_scaled,
        preds_real=preds_real,
        gt_real=gt_real,
        samples_real=samples_real,
        quantiles_scaled=quantiles_scaled,
        quantiles_real=quantiles_real,
        quantile_levels=quantile_levels,
        metric_space=metric_space,
        return_transform=return_transform,
        target_names=["Close"],
    )


def test_forecast_output_returns_space():
    """Con metric_space='returns', *_for_metrics devuelve preds_real (desescalados).

    Corrección: antes devolvía preds_scaled (standardizados), lo que hacía que
    Sharpe/VaR/CVaR se calcularan sobre valores z-score en vez de retornos reales.
    metric_space controla qué contiene preds_real (vía _inverse_transform_all),
    no qué array se selecciona en la propiedad.
    """
    fo = _make_forecast(metric_space="returns")
    assert np.array_equal(fo.preds_for_metrics, fo.preds_real)
    assert np.array_equal(fo.gt_for_metrics, fo.gt_real)
    assert np.array_equal(fo.samples_for_metrics, fo.samples_real)
    # Los valores son x100 (mock de inverse_transform), no los valores escalados
    assert not np.array_equal(fo.preds_for_metrics, fo.preds_scaled)


def test_forecast_output_prices_space():
    """Con metric_space='prices', *_for_metrics devuelve los arrays reales."""
    fo = _make_forecast(metric_space="prices", return_transform="log_return")
    assert np.array_equal(fo.preds_for_metrics, fo.preds_real)
    assert np.array_equal(fo.gt_for_metrics, fo.gt_real)
    assert np.array_equal(fo.samples_for_metrics, fo.samples_real)


def test_forecast_output_exposes_explicit_quantiles():
    """quantiles_for_metrics debe apuntar a quantiles_real con niveles explícitos."""
    fo = _make_forecast(metric_space="returns")
    assert fo.quantiles_for_metrics is not None
    assert fo.quantile_levels is not None
    assert np.array_equal(fo.quantiles_for_metrics, fo.quantiles_real)
    assert fo.quantiles_for_metrics.shape[0] == 3
    assert np.allclose(fo.quantile_levels, np.array([0.1, 0.5, 0.9], dtype=np.float32))


def test_risk_simulator_backward_compat():
    """RiskSimulator acepta np.ndarray directamente (backward compat)."""
    samples = np.random.randn(100, 10, 1)
    risk = RiskSimulator(samples)
    var = risk.calculate_var()
    assert var.shape == (10, 1)


def test_portfolio_simulator_backward_compat():
    """PortfolioSimulator acepta np.ndarray directamente (backward compat)."""
    preds = np.random.randn(20, 10, 1)
    gt = np.random.randn(20, 10, 1)
    ps = PortfolioSimulator(preds, gt)
    returns = ps.run_simple_strategy()
    assert returns.shape[0] == 19


def test_inverse_transform_noop_when_no_return_transform():
    """Con return_transform='none', preds_for_metrics devuelve preds_real (precios desescalados).

    El dataclass almacena ambos arrays correctamente; preds_for_metrics siempre
    apunta al espacio interpretable (preds_real), independiente del return_transform.
    """
    fo = _make_forecast(metric_space="returns", return_transform="none")
    # preds_for_metrics siempre devuelve preds_real
    assert np.array_equal(fo.preds_for_metrics, fo.preds_real)
    # preds_real es diferente de preds_scaled (x100 en nuestro mock de inverse_transform)
    assert not np.array_equal(fo.preds_real, fo.preds_scaled)


def test_window_fold_ids_shape_and_default():
    """window_fold_ids tiene shape (n_windows,) cuando se proporciona; None por defecto."""
    fo_sin_ids = _make_forecast()
    assert fo_sin_ids.window_fold_ids is None

    n_windows = 10
    fold_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
    fo_con_ids = _make_forecast()
    # Reconstruir con window_fold_ids explícito
    fo_con_ids = ForecastOutput(
        preds_scaled=fo_con_ids.preds_scaled,
        gt_scaled=fo_con_ids.gt_scaled,
        samples_scaled=fo_con_ids.samples_scaled,
        preds_real=fo_con_ids.preds_real,
        gt_real=fo_con_ids.gt_real,
        samples_real=fo_con_ids.samples_real,
        metric_space="returns",
        return_transform="none",
        target_names=["Close"],
        window_fold_ids=fold_ids,
    )
    assert fo_con_ids.window_fold_ids is not None
    assert fo_con_ids.window_fold_ids.shape == (n_windows,)
    assert fo_con_ids.window_fold_ids.dtype == np.int32
    assert set(fo_con_ids.window_fold_ids.tolist()) == {1, 2, 3}


def test_inverse_transform_log_return_produces_positive_prices():
    """Con return_transform='log_return' y metric_space='prices', samples_for_metrics
    devuelve samples_real (que debe ser positivo si representan precios)."""
    rng = np.random.default_rng(0)
    n_windows, pred_len, n_targets = 8, 4, 1
    n_samples = 20
    # Retornos pequeños positivos (simula log-returns)
    returns = rng.normal(0.001, 0.01, (n_samples, n_windows, pred_len, n_targets))
    # Precios reconstruidos (siempre positivos si last_price > 0)
    last_price = 150.0
    prices = last_price * np.exp(np.cumsum(returns, axis=-2))

    fo = ForecastOutput(
        preds_scaled=returns[:1, :, :, :].mean(axis=0),  # (8,4,1)
        gt_scaled=returns[:1, :, :, :].mean(axis=0),
        samples_scaled=returns,
        preds_real=prices[:1, :, :, :].mean(axis=0),
        gt_real=prices[:1, :, :, :].mean(axis=0),
        samples_real=prices,
        metric_space="prices",
        return_transform="log_return",
        target_names=["Close"],
    )

    assert fo.samples_for_metrics.shape == prices.shape
    assert np.all(fo.samples_for_metrics > 0), (
        "Los precios reconstruidos deben ser positivos"
    )


def test_forecast_output_quantile_helpers():
    """p10/p50/p90_real devuelven el cuantil correcto; get_quantile es consistente."""
    fo = _make_forecast(metric_space="returns")

    # Shapes correctos
    assert fo.p10_real.shape == fo.preds_real.shape
    assert fo.p50_real.shape == fo.preds_real.shape
    assert fo.p90_real.shape == fo.preds_real.shape

    # Invariante: p50_real corresponde al cuantil 0.5 en quantiles_real
    assert fo.quantile_levels is not None
    assert fo.quantiles_real is not None
    levels_list = [float(lev) for lev in fo.quantile_levels]
    idx_50 = min(range(len(levels_list)), key=lambda i: abs(levels_list[i] - 0.5))
    np.testing.assert_array_equal(fo.p50_real, fo.quantiles_real[idx_50])

    # get_quantile es consistente con las properties
    np.testing.assert_array_equal(fo.get_quantile(0.1), fo.p10_real)
    np.testing.assert_array_equal(fo.get_quantile(0.5), fo.p50_real)
    np.testing.assert_array_equal(fo.get_quantile(0.9), fo.p90_real)

    # scaled también funciona
    np.testing.assert_array_equal(fo.get_quantile(0.1, real=False), fo.p10_scaled)


def test_get_quantile_raises_when_missing():
    """get_quantile lanza ValueError si el nivel no está o quantiles es None."""
    fo = _make_forecast()

    # Nivel no disponible (0.25 no está en [0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="no disponible"):
        fo.get_quantile(0.25)

    # ForecastOutput con quantile_levels pero sin quantiles_real
    fo_no_q = ForecastOutput(
        preds_scaled=fo.preds_scaled,
        gt_scaled=fo.gt_scaled,
        samples_scaled=fo.samples_scaled,
        preds_real=fo.preds_real,
        gt_real=fo.gt_real,
        samples_real=fo.samples_real,
        quantile_levels=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        # quantiles_real=None por defecto
    )
    with pytest.raises(ValueError, match="no está disponible"):
        fo_no_q.get_quantile(0.5)
