from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import FEDformerConfig
from data import PreprocessingPipeline, TimeSeriesDataset


@pytest.fixture
def mixed_csv(tmp_path: Path) -> str:
    rng = np.random.default_rng(123)
    rows = 120
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "Close": rng.normal(100, 5, rows),
            "Volume": rng.normal(1_000, 50, rows),
            "sector": np.where(np.arange(rows) % 2 == 0, "tech", "energy"),
        }
    )
    df.loc[5, "Close"] = np.nan
    df.loc[15, "Volume"] = np.nan
    df.loc[20, "Close"] = 10_000.0
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_non_numeric_target_rejected(mixed_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["sector"],
        file_path=mixed_csv,
        date_column="date",
    )
    with pytest.raises(ValueError):
        TimeSeriesDataset(config=config, flag="train")


def test_missing_impute_is_deterministic(mixed_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=mixed_csv,
        date_column="date",
        missing_policy="impute_median",
        scaling_strategy="robust",
    )
    pipeline = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    pipeline.fit(raw, fit_end_idx=80)
    first = pipeline.transform(raw).values
    second = pipeline.transform(raw).values
    assert np.allclose(first, second, atol=1e-12)


def test_outlier_winsorize_clips_features_not_targets(tmp_path: Path) -> None:
    """winsorize clippea features no-target; las columnas target NO se clippean en transform()."""
    rng = np.random.default_rng(42)
    rows = 120
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "Close": rng.normal(100, 5, rows),
            "Volume": rng.normal(1_000, 50, rows),
        }
    )
    # Outlier de feature (Volume) en entrenamiento → debe ser clipeado en transform()
    df.loc[20, "Volume"] = 100_000.0
    # Outlier de target (Close) en PERIODO TEST (fila 90 > cutoff 80) → NO debe clipearse
    df.loc[90, "Close"] = 9_999.0

    csv_path = tmp_path / "outlier_test.csv"
    df.to_csv(csv_path, index=False)

    base_cfg = dict(
        target_features=["Close"],
        file_path=str(csv_path),
        date_column="date",
        missing_policy="ffill_bfill",
        scaling_strategy="none",
        drift_checks={"enabled": False},
    )
    pipe_win = PreprocessingPipeline.from_config(
        FEDformerConfig(**base_cfg, outlier_policy="winsorize"),
        target_features=["Close"],
        date_column="date",
    )
    pipe_win.fit(df, fit_end_idx=80)
    result = pipe_win.transform(df)

    # Feature no-target (Volume) SÍ se clippea en transform()
    assert result["Volume"].max() < 100_000.0
    # Target (Close) NO se clippea aunque supere el rango de entrenamiento
    assert result["Close"].iloc[90] == pytest.approx(9_999.0, rel=1e-3)


def test_last_prices_uses_training_cutoff(tmp_path: Path) -> None:
    """last_prices refleja el último precio del bloque de entrenamiento, no del dataset completo."""
    rng = np.random.default_rng(7)
    rows = 80
    prices = np.cumsum(rng.normal(0, 1, rows)) + 100.0
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "Close": prices,
        }
    )
    # El último precio del dataset completo es distinto del de training
    cutoff = 50
    last_train_price = float(df["Close"].iloc[cutoff])
    last_full_price = float(df["Close"].iloc[-1])
    assert last_train_price != pytest.approx(last_full_price, rel=1e-3)

    csv_path = tmp_path / "prices.csv"
    df.to_csv(csv_path, index=False)

    cfg = FEDformerConfig(
        target_features=["Close"],
        file_path=str(csv_path),
        date_column="date",
        scaling_strategy="none",
        outlier_policy="none",
        missing_policy="ffill_bfill",
        return_transform="log_return",
    )
    pipe = PreprocessingPipeline.from_config(
        cfg, target_features=["Close"], date_column="date"
    )
    pipe.fit(df, fit_end_idx=cutoff)

    # last_prices debe ser el precio en df.iloc[cutoff], no df.iloc[-1]
    assert pipe.last_prices["Close"] == pytest.approx(last_train_price, rel=1e-6)
    assert pipe.last_prices["Close"] != pytest.approx(last_full_price, rel=1e-3)


def test_persistence_roundtrip_equivalence(mixed_csv: str, tmp_path: Path) -> None:
    config = FEDformerConfig(
        target_features=["Close", "Volume"],
        file_path=mixed_csv,
        date_column="date",
        missing_policy="impute_median",
        scaling_strategy="standard",
        categorical_encoding="onehot",
        drift_checks={"enabled": False},
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    pipe = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    pipe.fit(raw, fit_end_idx=90)
    ref = pipe.transform(raw)

    artifact_dir = tmp_path / "prep"
    pipe.save_artifacts(artifact_dir)

    loaded = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    loaded.load_artifacts(artifact_dir)
    out = loaded.transform(raw)
    assert np.allclose(ref.values, out.values, atol=1e-8)


def test_drift_check_catches_missing_columns(mixed_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=mixed_csv,
        date_column="date",
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    pipe = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    pipe.fit(raw, fit_end_idx=80)
    with pytest.raises(ValueError):
        pipe.validate_input_schema(raw.drop(columns=["Volume"]))


def test_inverse_transform_multitarget(mixed_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close", "Volume"],
        file_path=mixed_csv,
        date_column="date",
        scaling_strategy="standard",
        missing_policy="impute_median",
        outlier_policy="none",
        drift_checks={"enabled": False},
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    pipe = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    pipe.fit(raw, fit_end_idx=100)
    transformed = pipe.transform(raw)

    y_scaled = transformed[["Close", "Volume"]].to_numpy().reshape(-1, 1, 2)
    y_inv = pipe.inverse_transform_targets(y_scaled, ["Close", "Volume"]).reshape(-1, 2)

    close_fill = pipe.fill_values.get("Close", float(raw["Close"].median()))
    volume_fill = pipe.fill_values.get("Volume", float(raw["Volume"].median()))
    close = raw["Close"].fillna(close_fill).to_numpy()
    volume = raw["Volume"].fillna(volume_fill).to_numpy()
    assert np.allclose(y_inv[:, 0], close, atol=1e-5)
    assert np.allclose(y_inv[:, 1], volume, atol=1e-5)


def test_log_return_transform(mixed_csv: str) -> None:
    """Verifica que los retornos logarítmicos se computan correctamente y el resultado tiene una fila menos que el input."""
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=mixed_csv,
        date_column="date",
        scaling_strategy="none",
        outlier_policy="none",
        missing_policy="impute_median",
        drift_checks={"enabled": False},
        return_transform="log_return",
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    pipe = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    pipe.fit(raw)
    result = pipe.transform(raw)

    # El resultado debe tener una fila menos que el input (primera fila NaN eliminada)
    assert len(result) == len(raw) - 1

    # Verificar que los retornos logarítmicos son finitos en la mayoría de los casos
    close_vals = result["Close"].to_numpy()
    assert np.isfinite(close_vals).sum() > len(close_vals) * 0.8


def test_return_inverse_transform(mixed_csv: str) -> None:
    """Verifica que los precios pueden recuperarse desde los retornos predichos."""
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=mixed_csv,
        date_column="date",
        scaling_strategy="none",
        outlier_policy="none",
        missing_policy="impute_median",
        drift_checks={"enabled": False},
        return_transform="log_return",
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    pipe = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    pipe.fit(raw)

    # Crear retornos logarítmicos sintéticos conocidos
    last_price = 100.0
    log_returns = np.array([0.01, -0.005, 0.02, 0.0])
    prices = pipe.inverse_transform_returns(log_returns, last_price)

    # Verificar que los precios se recuperan correctamente desde retornos log
    expected_0 = last_price * np.exp(log_returns[0])
    expected_1 = expected_0 * np.exp(log_returns[1])
    assert np.isclose(prices[0], expected_0, rtol=1e-6)
    assert np.isclose(prices[1], expected_1, rtol=1e-6)
    assert len(prices) == len(log_returns)

    # Verificar también simple_return
    config_sr = FEDformerConfig(
        target_features=["Close"],
        file_path=mixed_csv,
        date_column="date",
        scaling_strategy="none",
        outlier_policy="none",
        missing_policy="impute_median",
        drift_checks={"enabled": False},
        return_transform="simple_return",
    )
    pipe_sr = PreprocessingPipeline.from_config(
        config_sr,
        target_features=config_sr.target_features,
        date_column=config_sr.date_column,
    )
    pipe_sr.fit(raw)
    simple_returns = np.array([0.01, -0.005, 0.02])
    prices_sr = pipe_sr.inverse_transform_returns(simple_returns, last_price)
    expected_sr_0 = last_price * (1 + simple_returns[0])
    expected_sr_1 = expected_sr_0 * (1 + simple_returns[1])
    assert np.isclose(prices_sr[0], expected_sr_0, rtol=1e-6)
    assert np.isclose(prices_sr[1], expected_sr_1, rtol=1e-6)


def test_return_transform_leakage_safe(mixed_csv: str) -> None:
    """Verifica que el scaler se ajusta solo con retornos de entrenamiento, no con datos de test."""
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=mixed_csv,
        date_column="date",
        scaling_strategy="standard",
        outlier_policy="none",
        missing_policy="impute_median",
        drift_checks={"enabled": False},
        return_transform="log_return",
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    train_cutoff = 70

    pipe = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    # Ajustar solo con datos de entrenamiento
    pipe.fit(raw, fit_end_idx=train_cutoff)

    # Los estadísticos deben corresponder únicamente al split de entrenamiento
    train_df = raw.iloc[:train_cutoff].copy()
    # Calcular retornos de entrenamiento manualmente
    close_series = train_df["Close"].astype(float)
    train_log_returns = np.log(close_series / close_series.shift(1)).iloc[1:]
    train_mean = float(train_log_returns.mean())
    train_std = float(train_log_returns.std(ddof=0))

    # El scaler debe haberse ajustado con estadísticos del entrenamiento
    fit_stat = pipe.fit_stats.get("Close", {})
    assert abs(fit_stat.get("mean", float("nan")) - train_mean) < 0.05
    assert abs(fit_stat.get("std", float("nan")) - train_std) < 0.05


def test_last_prices_is_boundary_anchor(mixed_csv: str) -> None:
    """last_prices debe ser df.iloc[cutoff]: el precio frontera train/test.

    No es leakage: este precio ya está implícito en el último retorno de
    entrenamiento (log(P[cutoff]/P[cutoff-1])). Sirve como ancla para
    reconstruir precios desde retornos predichos en el período de test.
    """
    from config import FEDformerConfig
    from data import PreprocessingPipeline

    config = FEDformerConfig(
        target_features=["Close"],
        file_path=mixed_csv,
        date_column="date",
        scaling_strategy="standard",
        outlier_policy="none",
        missing_policy="impute_median",
        drift_checks={"enabled": False},
        return_transform="log_return",
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    cutoff = 70

    pipe = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    pipe.fit(raw, fit_end_idx=cutoff)

    # last_prices debe ser exactamente df.iloc[cutoff] — el precio en la frontera
    expected_anchor = float(raw["Close"].iloc[cutoff])
    assert pipe.last_prices is not None
    assert pipe.last_prices["Close"] == pytest.approx(expected_anchor, rel=1e-6)

    # Verificar que NO es el precio anterior (cutoff-1) ni el posterior (cutoff+1)
    assert pipe.last_prices["Close"] != pytest.approx(
        float(raw["Close"].iloc[cutoff - 1]), rel=1e-6
    )
