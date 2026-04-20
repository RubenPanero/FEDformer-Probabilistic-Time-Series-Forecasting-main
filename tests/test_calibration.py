# -*- coding: utf-8 -*-
"""Tests para utils/calibration.py y _apply_cp_calibration en main.py."""

import numpy as np
import pytest

from utils.calibration import (
    apply_conformal_interval,
    conformal_calibration_walkforward,
    conformal_quantile,
)


# ---------------------------------------------------------------------------
# Tests de conformal_quantile y apply_conformal_interval
# ---------------------------------------------------------------------------


def test_conformal_quantile_basic() -> None:
    """conformal_quantile devuelve un float no negativo para datos simples."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    q = conformal_quantile(y_true, y_pred, alpha=0.2)
    assert isinstance(q, float)
    assert q >= 0.0


def test_conformal_quantile_invalid_alpha() -> None:
    """conformal_quantile lanza ValueError para alpha fuera de (0, 1)."""
    y = np.ones(5)
    with pytest.raises(ValueError):
        conformal_quantile(y, y, alpha=0.0)
    with pytest.raises(ValueError):
        conformal_quantile(y, y, alpha=1.0)


def test_conformal_quantile_shape_mismatch() -> None:
    """conformal_quantile lanza ValueError cuando las formas no coinciden."""
    with pytest.raises(ValueError):
        conformal_quantile(np.ones(5), np.ones(6), alpha=0.1)


def test_apply_conformal_interval_shape() -> None:
    """apply_conformal_interval devuelve lower y upper con el mismo shape que y_pred."""
    y_pred = np.linspace(-1.0, 1.0, 20)
    lower, upper = apply_conformal_interval(y_pred, q_hat=0.5)
    assert lower.shape == y_pred.shape
    assert upper.shape == y_pred.shape
    assert np.all(upper >= lower)


# ---------------------------------------------------------------------------
# Test de _apply_cp_calibration (Enfoque 2 — global)
# ---------------------------------------------------------------------------


def test_apply_cp_calibration_returns_expected_coverage() -> None:
    """_apply_cp_calibration devuelve q_hat > 0 y cobertura en [0, 1]."""
    from training.forecast_output import ForecastOutput

    rng = np.random.default_rng(42)
    n_windows, pred_len, n_targets = 50, 5, 1
    preds = rng.standard_normal((n_windows, pred_len, n_targets)).astype(np.float32)
    gt = (
        preds
        + rng.standard_normal((n_windows, pred_len, n_targets)).astype(np.float32) * 0.3
    )

    forecast = ForecastOutput(
        preds_scaled=preds,
        gt_scaled=gt,
        preds_real=preds,
        gt_real=gt,
        samples_scaled=np.zeros((n_windows, pred_len, 10, n_targets)),
        samples_real=np.zeros((n_windows, pred_len, 10, n_targets)),
        target_names=["Close"],
        metric_space="returns",
        return_transform="log_return",
    )

    from main import _apply_cp_calibration

    result = _apply_cp_calibration(forecast, alpha=0.2)
    assert result["q_hat"] > 0
    assert 0.0 <= result["cp_coverage_80"] <= 1.0
    assert result["cp_lower"].shape == (n_windows * pred_len * n_targets,)
    assert result["cp_upper"].shape == (n_windows * pred_len * n_targets,)


def test_apply_cp_calibration_coverage_near_target() -> None:
    """Con residuos pequeños la cobertura 80% debe ser razonablemente alta."""
    from training.forecast_output import ForecastOutput

    rng = np.random.default_rng(0)
    n_windows, pred_len, n_targets = 200, 10, 1
    preds = rng.standard_normal((n_windows, pred_len, n_targets)).astype(np.float32)
    # Residuos pequeños → intervalo ampliado ≥ 80% cobertura
    gt = (
        preds
        + rng.standard_normal((n_windows, pred_len, n_targets)).astype(np.float32) * 0.1
    )

    forecast = ForecastOutput(
        preds_scaled=preds,
        gt_scaled=gt,
        preds_real=preds,
        gt_real=gt,
        samples_scaled=np.zeros((n_windows, pred_len, 10, n_targets)),
        samples_real=np.zeros((n_windows, pred_len, 10, n_targets)),
        target_names=["Close"],
        metric_space="returns",
        return_transform="log_return",
    )

    from main import _apply_cp_calibration

    result = _apply_cp_calibration(forecast, alpha=0.2)
    # La cobertura empírica debe ser ≥ 0.75 con residuos pequeños y suficientes muestras
    assert result["cp_coverage_80"] >= 0.75


# ---------------------------------------------------------------------------
# Tests de conformal_calibration_walkforward (Enfoque 1)
# ---------------------------------------------------------------------------


def test_conformal_calibration_walkforward_fold0_returns_none() -> None:
    """El fold con el id más bajo debe retornar q_hat=None."""
    rng = np.random.default_rng(42)
    residuals_by_fold = {k: rng.uniform(0, 1, 50) for k in range(3)}
    result = conformal_calibration_walkforward(residuals_by_fold, alpha=0.2)
    assert result[0] is None


def test_conformal_calibration_walkforward_subsequent_folds_have_qhat() -> None:
    """Los folds 1, 2, ... deben tener q_hat válido (float)."""
    rng = np.random.default_rng(42)
    residuals_by_fold = {k: rng.uniform(0, 1, 50) for k in range(3)}
    result = conformal_calibration_walkforward(residuals_by_fold, alpha=0.2)
    assert isinstance(result[1], float)
    assert isinstance(result[2], float)


def test_conformal_calibration_walkforward_coverage_increases_with_folds() -> None:
    """q_hat de folds posteriores usa más residuos acumulados."""
    rng = np.random.default_rng(7)
    residuals_by_fold = {k: rng.uniform(0, 1, 30) for k in range(5)}
    result = conformal_calibration_walkforward(residuals_by_fold, alpha=0.2)
    for fold_k in range(1, 5):
        assert result[fold_k] is not None
        assert np.isfinite(result[fold_k])


def test_conformal_calibration_walkforward_invalid_alpha() -> None:
    """Lanza ValueError para alpha fuera de (0, 1)."""
    residuals_by_fold = {0: np.ones(10), 1: np.ones(10)}
    with pytest.raises(ValueError):
        conformal_calibration_walkforward(residuals_by_fold, alpha=0.0)
    with pytest.raises(ValueError):
        conformal_calibration_walkforward(residuals_by_fold, alpha=1.5)


def test_conformal_calibration_walkforward_single_fold() -> None:
    """Con un solo fold, el resultado tiene q_hat=None."""
    result = conformal_calibration_walkforward(
        {0: np.array([0.1, 0.2, 0.3])}, alpha=0.2
    )
    assert len(result) == 1
    assert result[0] is None


def test_conformal_calibration_walkforward_returns_all_fold_ids() -> None:
    """El resultado contiene exactamente los mismos fold_ids que el input."""
    rng = np.random.default_rng(123)
    fold_ids = [0, 1, 3, 7]
    residuals_by_fold = {k: rng.uniform(0, 1, 20) for k in fold_ids}
    result = conformal_calibration_walkforward(residuals_by_fold, alpha=0.2)
    assert set(result.keys()) == set(fold_ids)


# ---------------------------------------------------------------------------
# Tests de _apply_cp_walkforward (Enfoque 1 - walk-forward)
# ---------------------------------------------------------------------------


def _make_forecast_with_folds(
    n_windows_per_fold: int = 30,
    pred_len: int = 5,
    n_folds: int = 3,
    noise_scale: float = 0.3,
    seed: int = 42,
) -> object:
    """Crea ForecastOutput sintético con window_fold_ids por fold."""
    from training.forecast_output import ForecastOutput

    rng = np.random.default_rng(seed)
    n_windows = n_windows_per_fold * n_folds
    n_targets = 1
    preds = rng.standard_normal((n_windows, pred_len, n_targets)).astype(np.float32)
    gt = preds + rng.standard_normal(preds.shape).astype(np.float32) * noise_scale
    fold_ids = np.repeat(np.arange(n_folds, dtype=np.int32), n_windows_per_fold)
    return ForecastOutput(
        preds_scaled=preds,
        gt_scaled=gt,
        preds_real=preds,
        gt_real=gt,
        samples_scaled=np.zeros((n_windows, pred_len, 10, n_targets)),
        samples_real=np.zeros((n_windows, pred_len, 10, n_targets)),
        target_names=["Close"],
        metric_space="returns",
        return_transform="log_return",
        window_fold_ids=fold_ids,
    )


def test_cp_walkforward_excludes_fold0_from_coverage() -> None:
    """folds_calibrated debe ser n_folds - 1 (fold 0 excluido)."""
    from main import _apply_cp_walkforward

    result = _apply_cp_walkforward(_make_forecast_with_folds(n_folds=3), alpha=0.2)
    assert result["cp_wf_folds_calibrated"] == 2
    assert 0.0 <= result["cp_wf_coverage_80"] <= 1.0


def test_cp_walkforward_returns_nan_when_no_fold_ids() -> None:
    """Si window_fold_ids es None retorna coverage=nan y folds_calibrated=0."""
    from training.forecast_output import ForecastOutput

    from main import _apply_cp_walkforward

    rng = np.random.default_rng(42)
    n_windows, pred_len, n_targets = 30, 5, 1
    preds = rng.standard_normal((n_windows, pred_len, n_targets)).astype(np.float32)
    forecast = ForecastOutput(
        preds_scaled=preds,
        gt_scaled=preds + 0.1,
        preds_real=preds,
        gt_real=preds + 0.1,
        samples_scaled=np.zeros((n_windows, pred_len, 10, n_targets)),
        samples_real=np.zeros((n_windows, pred_len, 10, n_targets)),
        target_names=["Close"],
        metric_space="returns",
        return_transform="log_return",
        window_fold_ids=None,
    )
    result = _apply_cp_walkforward(forecast, alpha=0.2)
    assert np.isnan(result["cp_wf_coverage_80"])
    assert result["cp_wf_folds_calibrated"] == 0


def test_cp_walkforward_q_hat_by_fold_structure() -> None:
    """cp_wf_q_hat_by_fold contiene todos los fold ids; fold 0 es None."""
    from main import _apply_cp_walkforward

    n_folds = 4
    result = _apply_cp_walkforward(
        _make_forecast_with_folds(n_folds=n_folds), alpha=0.2
    )
    q_hat_by_fold = result["cp_wf_q_hat_by_fold"]
    assert set(q_hat_by_fold.keys()) == set(range(n_folds))
    assert q_hat_by_fold[0] is None
    for k in range(1, n_folds):
        assert q_hat_by_fold[k] is not None
        assert q_hat_by_fold[k] >= 0.0


def test_cp_walkforward_high_coverage_with_small_noise() -> None:
    """Con residuos pequeños la cobertura walk-forward debe ser alta."""
    from main import _apply_cp_walkforward

    forecast = _make_forecast_with_folds(
        n_windows_per_fold=100, n_folds=4, noise_scale=0.05
    )
    result = _apply_cp_walkforward(forecast, alpha=0.2)
    assert result["cp_wf_coverage_80"] >= 0.75
