# -*- coding: utf-8 -*-
"""Tests para sequential_finetuner y main CLI: flags, propagación de config."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from config import FEDformerConfig
from data import TimeSeriesDataset
from models.fedformer import Flow_FEDformer
from training.forecast_output import ForecastOutput
from training.sequential_finetuner import _load_symbols_from_file, finetune_sequence
from training.trainer import WalkForwardTrainer

FIXTURE_CSV = "tests/fixtures/NVDA_features.csv"


def _base_config(**kwargs: object) -> FEDformerConfig:
    return FEDformerConfig(
        target_features=["Close"],
        file_path=FIXTURE_CSV,
        **kwargs,
    )


def _make_forecast_output() -> ForecastOutput:
    """Factory mínima de ForecastOutput para mocks del finetuner."""
    return ForecastOutput(
        preds_scaled=np.zeros((1, 1, 1)),
        gt_scaled=np.zeros((1, 1, 1)),
        samples_scaled=np.zeros((1, 1, 1, 1)),
        preds_real=np.zeros((1, 1, 1)),
        gt_real=np.zeros((1, 1, 1)),
        samples_real=np.zeros((1, 1, 1, 1)),
        metric_space="returns",
        return_transform="none",
        target_names=["Close"],
    )


# ---------------------------------------------------------------------------
# Tests de configuración base
# ---------------------------------------------------------------------------


def test_config_accepts_finetune_runtime_fields() -> None:
    cfg = _base_config(
        finetune_from="checkpoints/mock.pt",
        freeze_backbone=True,
        finetune_lr=1e-5,
    )
    assert cfg.finetune_from == "checkpoints/mock.pt"
    assert cfg.freeze_backbone is True
    assert cfg.finetune_lr == 1e-5


def test_trainer_finetune_checkpoint_and_freeze(tmp_path: Path) -> None:
    source_cfg = _base_config()
    _ = TimeSeriesDataset(source_cfg, flag="all")
    source_model = Flow_FEDformer(source_cfg)
    state = source_model.state_dict()
    state["components.regime_embedding.weight"] = torch.full_like(
        state["components.regime_embedding.weight"], 0.1234
    )
    ckpt_path = tmp_path / "ft.pt"
    torch.save({"model_state_dict": state}, ckpt_path)

    cfg = _base_config(
        finetune_from=str(ckpt_path),
        freeze_backbone=True,
        finetune_lr=1e-5,
        batch_size=8,
    )
    ds = TimeSeriesDataset(cfg, flag="all")
    trainer = WalkForwardTrainer(cfg, ds)
    train_subset = Subset(ds, list(range(min(32, len(ds)))))
    test_subset = Subset(ds, list(range(min(32, len(ds)))))
    components = trainer._build_training_components(
        train_subset, test_subset, fold_idx=1
    )

    loaded_value = components.model.state_dict()["components.regime_embedding.weight"][
        0, 0
    ]
    assert torch.isclose(loaded_value, torch.tensor(0.1234), atol=1e-6)

    trainable = {
        name: p.requires_grad for name, p in components.model.named_parameters()
    }
    assert trainable["flows.0.layers.0.conditioner.0.weight"] is True
    assert trainable["sequence_layers.encoders.0.layers.conv.0.weight"] is False


# ---------------------------------------------------------------------------
# Tests de _load_symbols_from_file
# ---------------------------------------------------------------------------


def test_load_symbols_from_file(tmp_path: Path) -> None:
    """Ignora líneas vacías y comentarios; preserva orden."""
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\n# comentario\nMSFT\n\nGOOGL\n", encoding="utf-8")
    assert _load_symbols_from_file(str(symbols_file)) == ["AAPL", "MSFT", "GOOGL"]


def test_load_symbols_from_file_only_comments(tmp_path: Path) -> None:
    """Devuelve lista vacía si el archivo solo contiene comentarios o blancos."""
    symbols_file = tmp_path / "empty.txt"
    symbols_file.write_text("# solo comentarios\n\n", encoding="utf-8")
    assert _load_symbols_from_file(str(symbols_file)) == []


# ---------------------------------------------------------------------------
# Tests de finetune_sequence
# ---------------------------------------------------------------------------


def test_finetune_sequence_passes_new_params_to_config(tmp_path: Path) -> None:
    """finetune_sequence propaga los nuevos parámetros al FEDformerConfig."""
    base_ckpt = tmp_path / "base.pt"
    base_ckpt.touch()
    ckpt_dir = tmp_path / "out" / "MOCK"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "best_model_fold_2.pt").touch()

    captured: dict = {}

    def fake_config(*_args: object, **kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    with (
        patch("training.sequential_finetuner.FEDformerConfig", side_effect=fake_config),
        patch("training.sequential_finetuner.TimeSeriesDataset"),
        patch("training.sequential_finetuner.WalkForwardTrainer") as MockTrainer,
        patch("os.path.exists", return_value=True),
    ):
        mock_trainer = MagicMock()
        mock_trainer.checkpoint_dir = ckpt_dir
        mock_trainer.run_backtest.return_value = _make_forecast_output()
        MockTrainer.return_value = mock_trainer

        finetune_sequence(
            symbols=["MOCK"],
            base_checkpoint=str(base_ckpt),
            output_dir=str(tmp_path / "out"),
            n_splits=3,
            main_epochs=10,
            scheduler_type="cosine",
            warmup_epochs=2,
            patience=3,
            min_delta=1e-4,
            return_transform="log_return",
            metric_space="prices",
            time_features=["month"],
        )

    assert captured["n_epochs_per_fold"] == 10
    assert captured["scheduler_type"] == "cosine"
    assert captured["warmup_epochs"] == 2
    assert captured["patience"] == 3
    assert captured["min_delta"] == 1e-4
    assert captured["return_transform"] == "log_return"
    assert captured["metric_space"] == "prices"
    assert captured["time_features"] == ["month"]


def test_finetune_sequence_wires_rehearsal_settings(tmp_path: Path) -> None:
    """finetune_sequence activa el rehearsal buffer en la config cuando se pasa rehearsal_k."""
    base_ckpt = tmp_path / "base.pt"
    base_ckpt.touch()
    ckpt_dir = tmp_path / "out" / "MOCK"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "best_model_fold_2.pt").touch()

    captured_config: list[MagicMock] = []

    def fake_config(*_args: object, **kwargs: object) -> MagicMock:
        mock = MagicMock()
        mock.sections.training.rehearsal.rehearsal_epochs = 1
        captured_config.append(mock)
        return mock

    with (
        patch("training.sequential_finetuner.FEDformerConfig", side_effect=fake_config),
        patch("training.sequential_finetuner.TimeSeriesDataset"),
        patch("training.sequential_finetuner.WalkForwardTrainer") as MockTrainer,
        patch("os.path.exists", return_value=True),
    ):
        mock_trainer = MagicMock()
        mock_trainer.checkpoint_dir = ckpt_dir
        mock_trainer.run_backtest.return_value = _make_forecast_output()
        MockTrainer.return_value = mock_trainer

        finetune_sequence(
            symbols=["MOCK"],
            base_checkpoint=str(base_ckpt),
            output_dir=str(tmp_path / "out"),
            n_splits=3,
            rehearsal_k=200,
            rehearsal_epochs=2,
            rehearsal_lr_mult=0.05,
        )

    cfg = captured_config[0]
    # Verificar que se activó el rehearsal buffer en la config
    assert cfg.rehearsal_enabled is True
    assert cfg.rehearsal_buffer_size == 200
    assert cfg.rehearsal_lr_mult == 0.05
    assert cfg.sections.training.rehearsal.rehearsal_epochs == 2


def test_dynamic_fold_checkpoint(tmp_path: Path) -> None:
    """El checkpoint propagado al siguiente ticker usa el índice del último fold."""
    for n_splits in [2, 3, 5]:
        expected_filename = f"best_model_fold_{n_splits - 1}.pt"

        base_ckpt = tmp_path / "base.pt"
        base_ckpt.touch()
        ckpt_dir = tmp_path / f"out_{n_splits}" / "MOCK"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / expected_filename).touch()

        with (
            patch(
                "training.sequential_finetuner.FEDformerConfig",
                return_value=MagicMock(),
            ),
            patch("training.sequential_finetuner.TimeSeriesDataset"),
            patch("training.sequential_finetuner.WalkForwardTrainer") as MockTrainer,
            patch("os.path.exists", return_value=True),
        ):
            mock_trainer = MagicMock()
            mock_trainer.checkpoint_dir = ckpt_dir
            mock_trainer.run_backtest.return_value = _make_forecast_output()
            MockTrainer.return_value = mock_trainer

            result_ckpt = finetune_sequence(
                symbols=["MOCK"],
                base_checkpoint=str(base_ckpt),
                output_dir=str(tmp_path / f"out_{n_splits}"),
                n_splits=n_splits,
            )

        assert result_ckpt is not None
        assert expected_filename in result_ckpt, (
            f"n_splits={n_splits}: esperado '{expected_filename}', obtenido '{result_ckpt}'"
        )


# ---------------------------------------------------------------------------
# Tests de wiring main.py → _create_config
# ---------------------------------------------------------------------------


def test_create_config_wires_architecture_flags() -> None:
    """_create_config propaga flags de arquitectura al FEDformerConfig."""
    import argparse

    import main as main_module

    args = argparse.Namespace(
        pred_len=20,
        seq_len=96,
        label_len=48,
        batch_size=8,
        use_checkpointing=False,
        grad_accum_steps=1,
        finetune_from=None,
        freeze_backbone=False,
        finetune_lr=None,
        wandb_project="test",
        wandb_entity=None,
        date_col=None,
        seed=42,
        deterministic=False,
        epochs=None,
        return_transform="log_return",
        metric_space="prices",
        dropout=None,
        weight_decay=None,
        learning_rate=None,
        scheduler_type=None,
        warmup_epochs=None,
        patience=None,
        min_delta=None,
        gradient_clip_norm=None,
        rehearsal_k=None,
        rehearsal_epochs=None,
        rehearsal_lr_mult=None,
        preset=None,
        conformal_calibration=False,
        cp_walkforward=False,
        compile_mode=None,
        mc_dropout_eval_samples=None,
        e_layers=5,
        d_layers=3,
        n_flow_layers=4,
        flow_hidden_dim=128,
    )

    cfg = main_module._create_config(
        args,
        targets=["Close"],
        csv_path=FIXTURE_CSV,
    )

    assert cfg.e_layers == 5
    assert cfg.d_layers == 3
    assert cfg.n_flow_layers == 4
    assert cfg.flow_hidden_dim == 128


def test_create_config_wires_return_transform_and_metric_space() -> None:
    """_create_config propaga --return-transform y --metric-space a FEDformerConfig."""
    import argparse

    import main as main_module

    args = argparse.Namespace(
        pred_len=20,
        seq_len=96,
        label_len=48,
        batch_size=8,
        use_checkpointing=False,
        grad_accum_steps=1,
        finetune_from=None,
        freeze_backbone=False,
        finetune_lr=None,
        wandb_project="test",
        wandb_entity=None,
        date_col=None,
        seed=42,
        deterministic=False,
        epochs=None,
        return_transform="log_return",
        metric_space="prices",
        dropout=None,
        weight_decay=None,
        learning_rate=None,
        scheduler_type=None,
        warmup_epochs=None,
        patience=None,
        min_delta=None,
        gradient_clip_norm=None,
        rehearsal_k=None,
        rehearsal_epochs=None,
        rehearsal_lr_mult=None,
        preset=None,
        conformal_calibration=False,
        cp_walkforward=False,
        compile_mode=None,
        mc_dropout_eval_samples=None,
        e_layers=None,
        d_layers=None,
        n_flow_layers=None,
        flow_hidden_dim=None,
    )

    cfg = main_module._create_config(
        args,
        targets=["Close"],
        csv_path=FIXTURE_CSV,
    )

    assert cfg.return_transform == "log_return"
    assert cfg.metric_space == "prices"


def test_create_config_cli_wins_over_preset() -> None:
    """CLI flags siempre tienen prioridad sobre --preset (orden: config_base → preset → flags_CLI)."""
    import argparse

    import main as main_module

    # preset 'debug' establece seq_len=32, pred_len=8, batch_size=16
    # Los valores CLI (96, 20, 64) deben ganar
    args = argparse.Namespace(
        seq_len=96,
        pred_len=20,
        batch_size=64,
        label_len=48,
        preset="debug",
        e_layers=None,
        d_layers=None,
        n_flow_layers=None,
        flow_hidden_dim=None,
        epochs=None,
        dropout=None,
        weight_decay=None,
        learning_rate=None,
        scheduler_type=None,
        warmup_epochs=None,
        patience=None,
        min_delta=None,
        gradient_clip_norm=None,
        rehearsal_k=None,
        rehearsal_epochs=None,
        rehearsal_lr_mult=None,
        compile_mode=None,
        mc_dropout_eval_samples=None,
        use_checkpointing=False,
        grad_accum_steps=1,
        finetune_from=None,
        freeze_backbone=False,
        finetune_lr=None,
        wandb_project=None,
        wandb_entity=None,
        date_col=None,
        seed=7,
        deterministic=False,
        return_transform="log_return",
        metric_space="returns",
        conformal_calibration=False,
        cp_walkforward=False,
    )
    config = main_module._create_config(args, targets=["Close"], csv_path=FIXTURE_CSV)

    assert config.seq_len == 96, (
        f"CLI seq_len=96 debe ganar sobre preset debug seq_len=32, got {config.seq_len}"
    )
    assert config.pred_len == 20, (
        f"CLI pred_len=20 debe ganar sobre preset debug pred_len=8, got {config.pred_len}"
    )
    assert config.batch_size == 64, (
        f"CLI batch_size=64 debe ganar sobre preset debug batch_size=16, got {config.batch_size}"
    )


def test_create_config_wires_mc_dropout_eval_samples() -> None:
    """_create_config propaga el knob trainer-only de MC Dropout."""
    import argparse

    import main as main_module

    args = argparse.Namespace(
        pred_len=20,
        seq_len=96,
        label_len=48,
        batch_size=8,
        use_checkpointing=False,
        grad_accum_steps=1,
        finetune_from=None,
        freeze_backbone=False,
        finetune_lr=None,
        wandb_project="test",
        wandb_entity=None,
        date_col=None,
        seed=42,
        deterministic=False,
        epochs=None,
        return_transform="log_return",
        metric_space="prices",
        dropout=None,
        weight_decay=None,
        learning_rate=None,
        scheduler_type=None,
        warmup_epochs=None,
        patience=None,
        min_delta=None,
        gradient_clip_norm=None,
        rehearsal_k=None,
        rehearsal_epochs=None,
        rehearsal_lr_mult=None,
        preset=None,
        conformal_calibration=False,
        cp_walkforward=False,
        compile_mode=None,
        mc_dropout_eval_samples=11,
        e_layers=None,
        d_layers=None,
        n_flow_layers=None,
        flow_hidden_dim=None,
    )

    cfg = main_module._create_config(
        args,
        targets=["Close"],
        csv_path=FIXTURE_CSV,
    )

    assert cfg.mc_dropout_eval_samples == 11


def test_save_results_to_csv_generates_fallback_target_names(tmp_path: Path) -> None:
    """_save_results_to_csv exporta filas aunque ForecastOutput no traiga target_names."""
    import main as main_module

    forecast = ForecastOutput(
        preds_scaled=np.zeros((1, 2, 1)),
        gt_scaled=np.zeros((1, 2, 1)),
        samples_scaled=np.zeros((1, 1, 2, 1)),
        preds_real=np.array([[[1.0], [2.0]]]),
        gt_real=np.array([[[1.5], [2.5]]]),
        samples_real=np.zeros((1, 1, 2, 1)),
        metric_space="returns",
        return_transform="none",
        target_names=None,
    )

    main_module._save_results_to_csv(
        forecast=forecast,
        risk_metrics={},
        portfolio_metrics={},
        results_dir=tmp_path,
        timestamp="20260324_000000",
    )

    exported = pd.read_csv(tmp_path / "predictions_20260324_000000.csv")
    assert len(exported) == 2
    assert set(exported["target"]) == {"target_0"}
