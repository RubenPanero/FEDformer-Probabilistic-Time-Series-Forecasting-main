# -*- coding: utf-8 -*-
"""Tests para TRAINING_PRESETS y apply_preset en config.py (Épica 7)."""

import pytest

from config import TRAINING_PRESETS, FEDformerConfig, apply_preset

FIXTURE_CSV = "tests/fixtures/NVDA_features.csv"


def _base_config(**kwargs: object) -> FEDformerConfig:
    """Factory de configuración mínima usando el fixture de tests."""
    return FEDformerConfig(
        target_features=["Close"],
        file_path=FIXTURE_CSV,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests de apply_preset por preset individual
# ---------------------------------------------------------------------------


def test_debug_preset_reduces_epochs() -> None:
    """apply_preset con 'debug' reduce n_epochs_per_fold a 3."""
    cfg = _base_config()
    assert cfg.n_epochs_per_fold == 20  # default
    apply_preset(cfg, "debug")
    assert cfg.n_epochs_per_fold == 3


def test_debug_preset_reduces_seq_len() -> None:
    """apply_preset con 'debug' reduce seq_len a 32."""
    cfg = _base_config()
    apply_preset(cfg, "debug")
    assert cfg.seq_len == 32


def test_debug_preset_reduces_batch_size() -> None:
    """apply_preset con 'debug' establece batch_size a 16."""
    cfg = _base_config()
    apply_preset(cfg, "debug")
    assert cfg.batch_size == 16


def test_debug_preset_disables_amp() -> None:
    """apply_preset con 'debug' desactiva AMP (use_amp=False)."""
    cfg = _base_config()
    apply_preset(cfg, "debug")
    assert cfg.use_amp is False


def test_debug_preset_sets_compile_mode_none() -> None:
    """apply_preset con 'debug' pone compile_mode='none'."""
    cfg = _base_config()
    apply_preset(cfg, "debug")
    assert cfg.compile_mode == "none"


def test_cpu_safe_preset_disables_amp() -> None:
    """apply_preset con 'cpu_safe' desactiva AMP."""
    cfg = _base_config()
    apply_preset(cfg, "cpu_safe")
    assert cfg.use_amp is False


def test_cpu_safe_preset_sets_num_workers_zero() -> None:
    """apply_preset con 'cpu_safe' pone num_workers=0."""
    cfg = _base_config()
    apply_preset(cfg, "cpu_safe")
    assert cfg.num_workers == 0


def test_cpu_safe_preset_disables_pin_memory() -> None:
    """apply_preset con 'cpu_safe' desactiva pin_memory."""
    cfg = _base_config()
    apply_preset(cfg, "cpu_safe")
    assert cfg.pin_memory is False


def test_cpu_safe_preset_sets_compile_mode_none() -> None:
    """apply_preset con 'cpu_safe' pone compile_mode='none'."""
    cfg = _base_config()
    apply_preset(cfg, "cpu_safe")
    assert cfg.compile_mode == "none"


def test_gpu_research_preset_sets_larger_batch() -> None:
    """apply_preset con 'gpu_research' aumenta batch_size a 128."""
    cfg = _base_config()
    apply_preset(cfg, "gpu_research")
    assert cfg.batch_size == 128


def test_gpu_research_preset_enables_amp() -> None:
    """apply_preset con 'gpu_research' activa AMP."""
    cfg = _base_config()
    # Forzar use_amp=False para verificar que el preset lo activa
    cfg.use_amp = False
    apply_preset(cfg, "gpu_research")
    assert cfg.use_amp is True


def test_gpu_research_preset_sets_compile_mode() -> None:
    """apply_preset con 'gpu_research' pone compile_mode='default'."""
    cfg = _base_config()
    apply_preset(cfg, "gpu_research")
    assert cfg.compile_mode == "default"


def test_gpu_research_preset_sets_num_workers() -> None:
    """apply_preset con 'gpu_research' pone num_workers=4."""
    cfg = _base_config()
    apply_preset(cfg, "gpu_research")
    assert cfg.num_workers == 4


def test_gpu_research_preset_enables_pin_memory() -> None:
    """apply_preset con 'gpu_research' activa pin_memory."""
    cfg = _base_config()
    apply_preset(cfg, "gpu_research")
    assert cfg.pin_memory is True


def test_probabilistic_eval_preset_sets_monitor_metric() -> None:
    """apply_preset con 'probabilistic_eval' pone monitor_metric='val_pinball_p50'."""
    cfg = _base_config()
    apply_preset(cfg, "probabilistic_eval")
    assert cfg.monitor_metric == "val_pinball_p50"


def test_probabilistic_eval_preset_sets_monitor_mode() -> None:
    """apply_preset con 'probabilistic_eval' pone monitor_mode='min'."""
    cfg = _base_config()
    apply_preset(cfg, "probabilistic_eval")
    assert cfg.monitor_mode == "min"


def test_probabilistic_eval_preset_increases_patience() -> None:
    """apply_preset con 'probabilistic_eval' aumenta patience a 10."""
    cfg = _base_config()
    apply_preset(cfg, "probabilistic_eval")
    assert cfg.patience == 10


def test_fourier_optimized_preset_sets_modes() -> None:
    """apply_preset con 'fourier_optimized' fija modes=48."""
    cfg = _base_config()
    apply_preset(cfg, "fourier_optimized")
    assert cfg.modes == 48


# ---------------------------------------------------------------------------
# Tests de comportamiento de apply_preset
# ---------------------------------------------------------------------------


def test_invalid_preset_raises_value_error() -> None:
    """apply_preset con nombre inexistente lanza ValueError."""
    cfg = _base_config()
    with pytest.raises(ValueError, match="no reconocido"):
        apply_preset(cfg, "nonexistent")


def test_apply_preset_returns_same_config_instance() -> None:
    """apply_preset retorna la misma instancia de config (mutación in-place)."""
    cfg = _base_config()
    result = apply_preset(cfg, "cpu_safe")
    assert result is cfg


def test_cli_override_takes_priority_over_preset() -> None:
    """Un override aplicado después del preset prevalece sobre el preset."""
    cfg = _base_config()
    apply_preset(cfg, "debug")
    assert cfg.batch_size == 16  # valor del preset
    # Simular override CLI aplicado después
    cfg.batch_size = 64
    assert cfg.batch_size == 64  # el override prevaleció


def test_preset_does_not_affect_unrelated_fields() -> None:
    """El preset 'cpu_safe' no modifica campos de sequencia ni learning_rate."""
    cfg = _base_config()
    lr_before = cfg.learning_rate
    apply_preset(cfg, "cpu_safe")
    assert cfg.learning_rate == lr_before


def test_preset_debug_does_not_affect_monitor_metric() -> None:
    """El preset 'debug' no modifica monitor_metric."""
    cfg = _base_config()
    metric_before = cfg.monitor_metric
    apply_preset(cfg, "debug")
    assert cfg.monitor_metric == metric_before


# ---------------------------------------------------------------------------
# Tests de integridad del diccionario TRAINING_PRESETS
# ---------------------------------------------------------------------------


def test_all_preset_names_are_valid() -> None:
    """Todos los nombres de preset en TRAINING_PRESETS son strings no vacíos."""
    assert len(TRAINING_PRESETS) > 0
    for name in TRAINING_PRESETS:
        assert isinstance(name, str) and len(name) > 0


def test_all_preset_values_are_dicts() -> None:
    """Todos los valores de TRAINING_PRESETS son diccionarios no vacíos."""
    for name, overrides in TRAINING_PRESETS.items():
        assert isinstance(overrides, dict), f"Preset '{name}' debe ser un dict"
        assert len(overrides) > 0, f"Preset '{name}' no debe estar vacío"


def test_expected_presets_present() -> None:
    """Los presets requeridos están presentes en TRAINING_PRESETS."""
    expected = {
        "debug",
        "cpu_safe",
        "gpu_research",
        "probabilistic_eval",
        "fourier_optimized",
    }
    assert expected.issubset(set(TRAINING_PRESETS.keys()))


def test_all_presets_are_applicable() -> None:
    """Todos los presets en TRAINING_PRESETS pueden aplicarse sin error."""
    for preset_name in TRAINING_PRESETS:
        cfg = _base_config()
        result = apply_preset(cfg, preset_name)
        assert result is cfg, f"Preset '{preset_name}' no retornó la misma instancia"
