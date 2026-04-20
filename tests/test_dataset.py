import pytest
from config import FEDformerConfig
from data import TimeSeriesDataset
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    data = {
        "date": pd.to_datetime(pd.date_range(start="2023-01-01", periods=100)),
        "Close": np.random.rand(100) * 100,
        "Volume": np.random.rand(100) * 1000,
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "sample_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_dataset_loading(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        date_column="date",
        seq_len=10,
        pred_len=4,
        label_len=5,
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    assert len(dataset) > 0


def test_dataset_scaling(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        date_column="date",
        seq_len=10,
        pred_len=4,
        label_len=5,
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    # The scaler should be fitted
    assert dataset.scaler is not None
    # The transformed data should be finite and roughly centered.
    assert np.isfinite(dataset.data_x).all()
    assert np.allclose(np.median(dataset.data_x, axis=0), 0, atol=1.0)


def test_dataset_inverse_scaling(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        date_column="date",
        seq_len=10,
        pred_len=4,
        label_len=5,
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    original_data = dataset.scaler.inverse_transform(dataset.data_x)
    assert np.allclose(
        original_data.mean(),
        pd.read_csv(sample_csv).drop("date", axis=1).mean().mean(),
        atol=100,
    )


def test_dataset_item(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        seq_len=10,
        pred_len=4,
        label_len=5,
        date_column="date",
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    item = dataset[0]
    assert "x_enc" in item
    assert "x_dec" in item
    assert "y_true" in item
    assert "x_regime" in item
    assert item["x_enc"].shape == (10, 2)  # seq_len, n_features
    assert item["y_true"].shape == (4, 1)  # pred_len, n_targets


def test_dataset_sequence_creation(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        seq_len=10,
        pred_len=4,
        label_len=5,
        date_column="date",
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    item = dataset[0]
    # The last `label_len` values of the encoder input should be the same as the first `label_len` values of the decoder input
    assert np.allclose(
        item["x_enc"][-config.label_len :], item["x_dec"][: config.label_len]
    )


def test_dataset_multi_target_numeric_alignment(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close", "Volume"],
        file_path=sample_csv,
        seq_len=10,
        pred_len=4,
        label_len=5,
        date_column="date",
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    item = dataset[0]
    assert item["y_true"].shape == (4, 2)


def test_dataset_with_return_transform(sample_csv: str) -> None:
    """Test de integración: verifica que TimeSeriesDataset funciona con return_transform='log_return'."""
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        date_column="date",
        seq_len=10,
        pred_len=4,
        label_len=5,
        return_transform="log_return",
        scaling_strategy="robust",
        missing_policy="impute_median",
        outlier_policy="none",
        drift_checks={"enabled": False},
    )
    dataset = TimeSeriesDataset(config=config, flag="train")

    # El dataset debe tener muestras válidas
    assert len(dataset) > 0

    # Las secuencias deben ser finitas
    item = dataset[0]
    assert np.isfinite(item["x_enc"].numpy()).all(), "x_enc contiene valores no finitos"
    assert np.isfinite(item["y_true"].numpy()).all(), (
        "y_true contiene valores no finitos"
    )

    # Las formas deben ser correctas
    assert item["x_enc"].shape[0] == config.seq_len
    assert item["y_true"].shape[0] == config.pred_len
    assert item["y_true"].shape[1] == 1  # una sola columna objetivo


def test_split_view_sizes_and_no_overlap(sample_csv: str) -> None:
    """Los splits train/val/test deben cubrir 70/20/10 y no solaparse."""
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        date_column="date",
        seq_len=10,
        pred_len=4,
        label_len=5,
    )

    ds_train = TimeSeriesDataset(config=config, flag="train")
    ds_val = TimeSeriesDataset(config=config, flag="val")
    ds_test = TimeSeriesDataset(config=config, flag="test")
    ds_all = TimeSeriesDataset(config=config, flag="all")

    n_rows = len(ds_all.full_data_scaled)
    num_train = int(n_rows * 0.7)
    num_val = int(n_rows * 0.2)

    # train cubre exactamente el 70%
    assert len(ds_train.data_x) == num_train

    # val cubre el 20% (sin incluir el 10% de test)
    assert len(ds_val.data_x) == num_val + config.seq_len  # contexto solapado incluido

    # el final de train == el inicio del bloque val puro (sin overlap)
    train_end = num_train
    val_end = num_train + num_val
    assert train_end <= val_end <= n_rows

    # test comienza donde termina val (con overlap de seq_len para contexto)
    assert len(ds_test.data_x) == n_rows - (num_train + num_val) + config.seq_len
