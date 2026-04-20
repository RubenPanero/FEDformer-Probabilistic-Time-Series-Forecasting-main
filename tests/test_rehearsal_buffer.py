# -*- coding: utf-8 -*-
"""Tests unitarios para RehearsalBuffer y su integración con config/trainer."""

import torch

from config import FEDformerConfig
from data import TimeSeriesDataset
from training.rehearsal_buffer import RehearsalBuffer
from training.trainer import WalkForwardTrainer

FIXTURE_CSV = "tests/fixtures/NVDA_features.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    """Crea un batch sintético con la estructura esperada por el buffer."""
    return {
        "x_enc": torch.randn(batch_size, 10, 3),
        "x_dec": torch.randn(batch_size, 5, 3),
        "y_true": torch.randn(batch_size, 5, 1),
        "x_regime": torch.zeros(batch_size, dtype=torch.long),
    }


def _base_config(**kwargs: object) -> FEDformerConfig:
    return FEDformerConfig(
        target_features=["Close"],
        file_path=FIXTURE_CSV,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests del buffer
# ---------------------------------------------------------------------------


def test_buffer_empty_on_init() -> None:
    """El buffer comienza vacío tras la construcción."""
    buf = RehearsalBuffer(capacity=100)
    assert len(buf) == 0
    assert not buf.is_ready


def test_add_batch_increases_len() -> None:
    """add_batch de un batch de 4 aumenta el buffer en 4 muestras."""
    buf = RehearsalBuffer(capacity=100)
    buf.add_batch(_make_batch(batch_size=4))
    assert len(buf) == 4
    assert buf.is_ready


def test_capacity_limit_fifo() -> None:
    """Con capacity=5, añadir 8 muestras mantiene solo las 5 más recientes."""
    buf = RehearsalBuffer(capacity=5)
    buf.add_batch(_make_batch(batch_size=8))
    assert len(buf) == 5


def test_sample_returns_correct_shape() -> None:
    """sample(k=2) devuelve tensores con shape[0] == 2."""
    buf = RehearsalBuffer(capacity=100)
    buf.add_batch(_make_batch(batch_size=10))
    result = buf.sample(k=2)
    assert result is not None
    assert result["x_enc"].shape[0] == 2
    assert result["y_true"].shape[0] == 2


def test_sample_clamps_k_to_available() -> None:
    """sample(k=100) cuando solo hay 3 muestras devuelve shape[0] == 3."""
    buf = RehearsalBuffer(capacity=100)
    buf.add_batch(_make_batch(batch_size=3))
    result = buf.sample(k=100)
    assert result is not None
    assert result["x_enc"].shape[0] == 3


def test_sample_empty_returns_none() -> None:
    """El buffer vacío devuelve None al hacer sample."""
    buf = RehearsalBuffer(capacity=100)
    assert buf.sample(k=5) is None


def test_update_alias_for_add_batch() -> None:
    """update() produce el mismo efecto que add_batch()."""
    buf = RehearsalBuffer(capacity=100)
    buf.update(_make_batch(batch_size=4))
    assert len(buf) == 4


# ---------------------------------------------------------------------------
# Tests de integración config / trainer
# ---------------------------------------------------------------------------


def test_rehearsal_settings_defaults() -> None:
    """RehearsalSettings tiene los defaults correctos."""
    from config import RehearsalSettings

    rs = RehearsalSettings()
    assert rs.enabled is False
    assert rs.buffer_size == 1000
    assert rs.rehearsal_epochs == 1
    assert rs.rehearsal_lr_mult == 0.1


def test_config_rehearsal_disabled_by_default() -> None:
    """FEDformerConfig por defecto tiene rehearsal desactivado."""
    cfg = _base_config()
    assert cfg.rehearsal_enabled is False
    assert cfg.rehearsal_buffer_size == 1000
    assert cfg.rehearsal_lr_mult == 0.1


def test_config_rehearsal_setters() -> None:
    """Los setters de rehearsal actualizan correctamente la config anidada."""
    cfg = _base_config()
    cfg.rehearsal_enabled = True
    cfg.rehearsal_buffer_size = 500
    cfg.rehearsal_lr_mult = 0.05
    assert cfg.sections.training.rehearsal.enabled is True
    assert cfg.sections.training.rehearsal.buffer_size == 500
    assert cfg.sections.training.rehearsal.rehearsal_lr_mult == 0.05


def test_trainer_buffer_none_when_disabled() -> None:
    """WalkForwardTrainer con config default tiene rehearsal_buffer is None."""
    cfg = _base_config(batch_size=8)
    ds = TimeSeriesDataset(cfg, flag="all")
    trainer = WalkForwardTrainer(cfg, ds)
    assert trainer.rehearsal_buffer is None


def test_trainer_buffer_initialized_when_enabled() -> None:
    """WalkForwardTrainer con rehearsal_enabled=True inicializa el buffer."""
    cfg = _base_config(batch_size=8)
    cfg.rehearsal_enabled = True
    cfg.rehearsal_buffer_size = 200
    ds = TimeSeriesDataset(cfg, flag="all")
    trainer = WalkForwardTrainer(cfg, ds)
    assert trainer.rehearsal_buffer is not None
    assert len(trainer.rehearsal_buffer) == 0
    assert trainer.rehearsal_buffer._capacity == 200
