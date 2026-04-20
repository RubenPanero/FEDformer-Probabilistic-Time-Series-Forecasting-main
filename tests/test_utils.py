# -*- coding: utf-8 -*-
"""
Tests para los módulos auxiliares de utils/:
  - utils/helpers.py   → get_device, set_seed, setup_cuda_optimizations, _select_amp_dtype
  - utils/calibration.py → conformal_quantile
  - utils/metrics.py   → MetricsTracker
"""

import numpy as np
import pytest
import torch

from utils.calibration import conformal_quantile
from utils.helpers import (
    _select_amp_dtype,
    get_device,
    set_seed,
    setup_cuda_optimizations,
)
from utils.metrics import MetricsTracker

# ---------------------------------------------------------------------------
# Grupo 1: utils/helpers.py
# ---------------------------------------------------------------------------


class TestGetDevice:
    """Pruebas para get_device()."""

    def test_get_device_returns_valid(self) -> None:
        """get_device() debe retornar un torch.device válido (cpu o cuda)."""
        device = get_device()

        # Verificar tipo correcto
        assert isinstance(device, torch.device)

        # Verificar que es uno de los valores esperados
        assert device.type in ("cpu", "cuda"), (
            f"Se esperaba 'cpu' o 'cuda', pero se obtuvo: {device.type}"
        )

    def test_get_device_matches_cuda_availability(self) -> None:
        """get_device() debe retornar cuda solo si CUDA está disponible."""
        device = get_device()

        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"


class TestSetSeed:
    """Pruebas para set_seed()."""

    def test_set_seed_determinism(self) -> None:
        """Llamar set_seed(42) dos veces debe producir la misma secuencia de torch.randn()."""
        set_seed(42)
        secuencia_1 = torch.randn(10)

        set_seed(42)
        secuencia_2 = torch.randn(10)

        assert torch.allclose(secuencia_1, secuencia_2), (
            "set_seed(42) no produce secuencias deterministas en torch.randn()"
        )

    def test_set_seed_different_seeds_differ(self) -> None:
        """Semillas distintas deben producir secuencias distintas (con alta probabilidad)."""
        set_seed(42)
        secuencia_a = torch.randn(20)

        set_seed(99)
        secuencia_b = torch.randn(20)

        # Es estadísticamente imposible que sean idénticas con semillas distintas
        assert not torch.allclose(secuencia_a, secuencia_b), (
            "Semillas distintas produjeron secuencias idénticas (fallo de aleatoriedad)"
        )

    def test_set_seed_affects_numpy(self) -> None:
        """set_seed() también debe fijar la semilla de NumPy."""
        set_seed(7)
        arr_1 = np.random.randn(5)

        set_seed(7)
        arr_2 = np.random.randn(5)

        np.testing.assert_array_equal(arr_1, arr_2)

    def test_set_seed_no_crash_with_deterministic_flag(self) -> None:
        """set_seed() con deterministic=True no debe lanzar excepciones."""
        # No se hace assert sobre el estado global; solo verificar que no falla
        set_seed(0, deterministic=True)


class TestSetupCudaOptimizations:
    """Pruebas para setup_cuda_optimizations()."""

    def test_setup_cuda_optimizations_no_crash(self) -> None:
        """setup_cuda_optimizations() no debe lanzar excepciones en ningún entorno (incluyendo CPU)."""
        # En CPU es un no-op seguro; en GPU activa TF32/benchmark
        try:
            setup_cuda_optimizations()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"setup_cuda_optimizations() lanzó una excepción inesperada: {exc}"
            )


class TestSelectAmpDtype:
    """Pruebas para _select_amp_dtype() (función interna expuesta para testing)."""

    def test_select_amp_dtype_returns_torch_dtype(self) -> None:
        """_select_amp_dtype() debe retornar un torch.dtype válido."""
        dtype = _select_amp_dtype()
        assert isinstance(dtype, torch.dtype)

    def test_select_amp_dtype_valid_values(self) -> None:
        """_select_amp_dtype() debe retornar float16 o bfloat16."""
        dtype = _select_amp_dtype()
        assert dtype in (torch.float16, torch.bfloat16), (
            f"Se esperaba float16 o bfloat16, pero se obtuvo: {dtype}"
        )


# ---------------------------------------------------------------------------
# Grupo 2: utils/calibration.py
# ---------------------------------------------------------------------------


class TestConformalQuantile:
    """Pruebas para conformal_quantile()."""

    def test_conformal_quantile_basic(self) -> None:
        """El resultado debe estar entre 0 y max(residuals)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        q = conformal_quantile(y_true, y_pred, alpha=0.1)

        residuals = np.abs(y_true - y_pred)
        assert isinstance(q, float)
        assert 0.0 <= q <= float(np.max(residuals)), (
            f"El cuantil conforme {q} está fuera del rango [0, {np.max(residuals)}]"
        )

    def test_conformal_quantile_alpha_zero_raises(self) -> None:
        """alpha=0 debe lanzar ValueError (fuera del intervalo abierto (0, 1))."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.raises(ValueError):
            conformal_quantile(y_true, y_pred, alpha=0.0)

    def test_conformal_quantile_alpha_one_raises(self) -> None:
        """alpha=1 debe lanzar ValueError (fuera del intervalo abierto (0, 1))."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.raises(ValueError):
            conformal_quantile(y_true, y_pred, alpha=1.0)

    def test_conformal_quantile_alpha_validation(self) -> None:
        """Verificar que alpha=0 y alpha=1 lanzan ValueError en una sola prueba parametrizada."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        for invalid_alpha in (0.0, 1.0):
            with pytest.raises(ValueError, match="Alpha"):
                conformal_quantile(y_true, y_pred, alpha=invalid_alpha)

    def test_conformal_quantile_shape_mismatch(self) -> None:
        """y_true e y_pred con shapes distintos deben lanzar ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])  # shape diferente

        with pytest.raises(ValueError):
            conformal_quantile(y_true, y_pred, alpha=0.1)

    def test_conformal_quantile_empty_raises(self) -> None:
        """Arrays vacíos deben lanzar ValueError."""
        y_true = np.array([])
        y_pred = np.array([])

        with pytest.raises(ValueError):
            conformal_quantile(y_true, y_pred, alpha=0.1)

    def test_conformal_quantile_perfect_predictions(self) -> None:
        """Con predicciones perfectas todos los residuos son 0; el cuantil debe ser 0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        q = conformal_quantile(y_true, y_pred, alpha=0.1)
        assert q == pytest.approx(0.0)

    def test_conformal_quantile_2d_arrays(self) -> None:
        """Debe funcionar con arrays 2D (reshape interno a 1D)."""
        y_true = np.ones((4, 3))
        y_pred = np.zeros((4, 3))  # residuos todos = 1

        q = conformal_quantile(y_true, y_pred, alpha=0.1)
        assert isinstance(q, float)
        assert q == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Grupo 3: utils/metrics.py — MetricsTracker
# ---------------------------------------------------------------------------


class TestMetricsTracker:
    """Pruebas para MetricsTracker."""

    def test_metrics_tracker_log_and_retrieve(self) -> None:
        """log_metrics() y get_summary() deben devolver valores estadísticos correctos."""
        tracker = MetricsTracker()

        # Registrar una sola métrica con un valor conocido
        tracker.log_metrics({"loss": 0.5}, step=1)

        summary = tracker.get_summary()

        assert "loss" in summary, "La clave 'loss' no aparece en get_summary()"
        stats = summary["loss"]

        # Con un solo valor, mean == min == max == 0.5 y std == 0
        assert stats["mean"] == pytest.approx(0.5)
        assert stats["min"] == pytest.approx(0.5)
        assert stats["max"] == pytest.approx(0.5)
        assert stats["std"] == pytest.approx(0.0)

    def test_metrics_tracker_multiple_epochs(self) -> None:
        """Acumular varias épocas y verificar que el historial es correcto."""
        tracker = MetricsTracker()

        valores_loss = [1.0, 0.8, 0.6, 0.4]
        for step, val in enumerate(valores_loss, start=1):
            tracker.log_metrics({"train_loss": val}, step=step)

        summary = tracker.get_summary()

        assert "train_loss" in summary
        stats = summary["train_loss"]

        # Verificar estadísticos contra valores conocidos
        assert stats["mean"] == pytest.approx(np.mean(valores_loss))
        assert stats["min"] == pytest.approx(min(valores_loss))
        assert stats["max"] == pytest.approx(max(valores_loss))
        assert stats["std"] == pytest.approx(float(np.std(valores_loss)))

    def test_metrics_tracker_multiple_keys(self) -> None:
        """log_metrics() acepta varios keys en una misma llamada."""
        tracker = MetricsTracker()

        tracker.log_metrics({"loss": 0.3, "accuracy": 0.95}, step=1)
        tracker.log_metrics({"loss": 0.2, "accuracy": 0.97}, step=2)

        summary = tracker.get_summary()

        assert "loss" in summary
        assert "accuracy" in summary

        # Verificar medias
        assert summary["loss"]["mean"] == pytest.approx(0.25)
        assert summary["accuracy"]["mean"] == pytest.approx(0.96)

    def test_metrics_tracker_empty_summary(self) -> None:
        """Un tracker recién creado sin logs debe devolver un dict vacío."""
        tracker = MetricsTracker()
        summary = tracker.get_summary()
        assert summary == {}

    def test_metrics_tracker_step_order_preserved(self) -> None:
        """El historial interno debe conservar el orden de inserción (fold, step, value)."""
        tracker = MetricsTracker()

        pasos = [(1, 1.0), (2, 0.5), (3, 0.25)]
        for step, val in pasos:
            tracker.log_metrics({"loss": val}, step=step)

        # Acceso directo al historial interno: formato (fold=0, step, value)
        historial = tracker.metrics["loss"]
        esperado = [(0, step, val) for step, val in pasos]
        assert historial == esperado, (
            f"El orden del historial interno no coincide. Se obtuvo: {historial}"
        )

    def test_metrics_tracker_to_dataframe(self) -> None:
        """to_dataframe() debe exportar el historial con columnas fold, epoch, metric, value."""
        tracker = MetricsTracker()
        tracker.log_metrics({"train_loss": 0.8, "val_loss": 0.9}, step=0, fold=1)
        tracker.log_metrics({"train_loss": 0.6, "val_loss": 0.7}, step=1, fold=1)

        df = tracker.to_dataframe()

        assert set(df.columns) == {"fold", "epoch", "metric", "value"}
        assert len(df) == 4  # 2 métricas × 2 pasos
        assert set(df["metric"].unique()) == {"train_loss", "val_loss"}
        assert list(df[df["metric"] == "train_loss"]["value"]) == [0.8, 0.6]
        assert all(df["fold"] == 1)

    def test_metrics_tracker_to_dataframe_empty(self) -> None:
        """to_dataframe() sin datos debe devolver DataFrame vacío con columnas correctas."""
        tracker = MetricsTracker()
        df = tracker.to_dataframe()
        assert list(df.columns) == ["fold", "epoch", "metric", "value"]
        assert len(df) == 0
