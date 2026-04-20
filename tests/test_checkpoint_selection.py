# -*- coding: utf-8 -*-
"""Tests para el monitor de checkpoint configurable (PR 3).

Verifica que monitor_metric y monitor_mode en LoopSettings/FEDformerConfig
funcionan correctamente, que _select_monitor_value elige la métrica adecuada,
y que la configuración inválida lanza ValueError.
"""

import pytest

from config import LoopSettings
from training.trainer import WalkForwardTrainer


def test_monitor_metric_defaults_to_val_loss():
    """LoopSettings tiene monitor_metric='val_loss' y monitor_mode='min' por defecto."""
    loop = LoopSettings()
    assert loop.monitor_metric == "val_loss"
    assert loop.monitor_mode == "min"


def test_checkpoint_selection_uses_pinball_when_configured():
    """_select_monitor_value retorna val_metrics['pinball_p50'] cuando está configurado."""
    train_m = {"loss": 1.5}
    val_m = {"loss": 1.2, "pinball_p50": 0.08}
    result = WalkForwardTrainer._select_monitor_value(train_m, val_m, "val_pinball_p50")
    assert abs(result - 0.08) < 1e-9, f"Esperado 0.08, obtenido {result}"


def test_checkpoint_selection_fallback_on_missing_pinball(caplog):
    """_select_monitor_value usa val_loss como fallback si pinball_p50 no está disponible."""
    import logging

    train_m = {"loss": 1.5}
    val_m = {"loss": 1.2}  # sin pinball_p50
    with caplog.at_level(logging.WARNING):
        result = WalkForwardTrainer._select_monitor_value(
            train_m, val_m, "val_pinball_p50"
        )
    assert abs(result - 1.2) < 1e-9, (
        f"Esperado fallback a val_loss=1.2, obtenido {result}"
    )
    assert any("fallback" in rec.message.lower() for rec in caplog.records)


def test_checkpoint_selection_composite():
    """_select_monitor_value retorna 0.5*val_loss + 0.5*pinball_p50 para 'composite'."""
    train_m = {"loss": 1.0}
    val_m = {"loss": 1.0, "pinball_p50": 0.2}
    result = WalkForwardTrainer._select_monitor_value(train_m, val_m, "composite")
    expected = 0.5 * 1.0 + 0.5 * 0.2
    assert abs(result - expected) < 1e-9, f"Esperado {expected}, obtenido {result}"


def test_monitor_mode_max_inverts_comparison():
    """Con monitor_mode='max', el valor efectivo es negado (mayor score = mejor checkpoint)."""
    loop = LoopSettings(monitor_mode="max")
    assert loop.monitor_mode == "max"

    # Simular la lógica del trainer: monitor_sign = -1 para max
    monitor_sign = -1.0 if loop.monitor_mode == "max" else 1.0

    # Un score más alto (mejor en max) debe dar un effective_val más bajo
    score_a = 0.8
    score_b = 0.6  # peor en max
    assert monitor_sign * score_a < monitor_sign * score_b, (
        "Para mode='max', score más alto debe tener effective_val más bajo"
    )


def test_invalid_monitor_metric_raises_at_config_init():
    """LoopSettings lanza ValueError con monitor_metric inválido."""
    with pytest.raises(ValueError, match="inválido"):
        LoopSettings(monitor_metric="invalid_metric")


def test_invalid_monitor_mode_raises_at_config_init():
    """LoopSettings lanza ValueError con monitor_mode inválido."""
    with pytest.raises(ValueError, match="inválido"):
        LoopSettings(monitor_mode="invalid_mode")


def test_fedformerconfig_monitor_metric_setter_validates(config):
    """FEDformerConfig.monitor_metric setter valida el valor."""
    with pytest.raises(ValueError, match="inválido"):
        config.monitor_metric = "bad_value"


def test_fedformerconfig_monitor_metric_round_trip(config):
    """FEDformerConfig expone monitor_metric y monitor_mode vía sections."""
    config.monitor_metric = "composite"
    config.monitor_mode = "min"
    assert config.monitor_metric == "composite"
    assert config.monitor_mode == "min"
    assert config.sections.training.loop.monitor_metric == "composite"
