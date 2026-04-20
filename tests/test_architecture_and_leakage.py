import numpy as np
import torch
from pathlib import Path

from config import FEDformerConfig
from data import TimeSeriesDataset
from models.fedformer import Flow_FEDformer
from models.layers import AttentionConfig, AttentionLayer
from training import WalkForwardTrainer
from utils import apply_conformal_interval, conformal_quantile

FIXTURE_CSV = str(Path("tests/fixtures/NVDA_features.csv"))


def test_model_uses_configured_depth() -> None:
    cfg = FEDformerConfig(
        target_features=["Close"],
        file_path=FIXTURE_CSV,
        e_layers=3,
        d_layers=2,
    )
    model = Flow_FEDformer(cfg)
    assert len(model.sequence_layers["encoders"]) == 3
    assert len(model.sequence_layers["decoders"]) == 2


def test_attention_output_depends_on_values() -> None:
    cfg = AttentionConfig(d_model=8, n_heads=2, seq_len=6, modes=3, dropout=0.0)
    attn = AttentionLayer(cfg).eval()
    q = torch.randn(2, 6, 8)
    k = torch.randn(2, 6, 8)
    v_zeros = torch.zeros(2, 6, 8)
    v_random = torch.randn(2, 6, 8)
    out_zeros = attn(q, k, v_zeros)
    out_random = attn(q, k, v_random)
    assert not torch.allclose(out_zeros, out_random)


def test_fold_indices_avoid_label_leakage() -> None:
    cfg = FEDformerConfig(
        target_features=["Close"],
        file_path=FIXTURE_CSV,
        seq_len=32,
        label_len=16,
        pred_len=8,
    )
    ds = TimeSeriesDataset(cfg, flag="all")
    trainer = WalkForwardTrainer(cfg, ds)

    train_end_idx = 100
    test_end_idx = 140
    train_indices, test_indices = trainer._build_fold_indices(
        train_end_idx, test_end_idx
    )

    assert train_indices
    assert test_indices
    assert max(train_indices) + cfg.seq_len + cfg.pred_len - 1 < train_end_idx
    assert min(test_indices) == max(0, train_end_idx - cfg.seq_len)
    assert min(test_indices) + cfg.seq_len >= train_end_idx
    assert max(test_indices) + cfg.seq_len + cfg.pred_len - 1 < test_end_idx


def test_fold_refit_changes_scaler_distribution() -> None:
    cfg = FEDformerConfig(
        target_features=["Close"],
        file_path=FIXTURE_CSV,
    )
    ds = TimeSeriesDataset(cfg, flag="all")
    if hasattr(ds.scaler, "center_"):
        first_mean = float(ds.scaler.center_[0])
    elif hasattr(ds.scaler, "mean_"):
        first_mean = float(ds.scaler.mean_[0])
    else:
        first_mean = float(ds.full_data_scaled[:, 0].mean())
    ds.refit_for_cutoff(50)
    if hasattr(ds.scaler, "center_"):
        second_mean = float(ds.scaler.center_[0])
    elif hasattr(ds.scaler, "mean_"):
        second_mean = float(ds.scaler.mean_[0])
    else:
        second_mean = float(ds.full_data_scaled[:, 0].mean())
    assert first_mean != second_mean


def test_conformal_interval_helpers() -> None:
    y_true = np.array([[1.0, 2.0], [1.1, 1.8]])
    y_pred = np.array([[1.0, 2.2], [0.9, 2.0]])
    q_hat = conformal_quantile(y_true, y_pred, alpha=0.2)
    lower, upper = apply_conformal_interval(y_pred, q_hat)
    assert q_hat >= 0
    assert np.all(lower <= upper)
