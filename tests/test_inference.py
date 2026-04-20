# -*- coding: utf-8 -*-
"""Tests para el paquete inference."""

import json
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def mock_registry(tmp_path):
    """Crea un registry self-contained en tmp_path con un especialista NVDA.

    IMPORTANTE: Crea un CSV sintético en tmp_path para que FEDformerConfig
    pueda leerlo al detectar enc_in/dec_in. El modelo y los datos usan
    las mismas 2 columnas (Close, Volume) para coherencia de dimensiones.
    """
    # 1. Crear CSV sintético — necesario para que FEDformerConfig.__init__ funcione
    rng = np.random.default_rng(42)
    n_rows = 200
    csv_path = tmp_path / "NVDA_features.csv"
    pd.DataFrame(
        {
            "Close": np.cumsum(rng.standard_normal(n_rows)) + 100,
            "Volume": rng.integers(1000, 10000, n_rows).astype(float),
        }
    ).to_csv(csv_path, index=False)

    # 2. Crear config y modelo con el CSV real → enc_in/dec_in = 2
    from config import FEDformerConfig

    config = FEDformerConfig(
        target_features=["Close"],
        file_path=str(csv_path),
        seq_len=20,  # Corto para tests rápidos
        label_len=10,
        pred_len=4,  # Par (requisito affine coupling)
        batch_size=8,
    )
    from models.fedformer import Flow_FEDformer

    model = Flow_FEDformer(config)

    # 3. Crear checkpoint
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": None,
        "epoch": 10,
        "fold": 3,
        "loss": 0.5,
        "config": {
            "seq_len": 20,
            "label_len": 10,
            "pred_len": 4,
            "n_splits": 4,
            "return_transform": "none",
            "metric_space": "returns",
            "gradient_clip_norm": 0.5,
            "batch_size": 8,
            "seed": 7,
            "target_features": ["Close"],
            # Parámetros de arquitectura (reflejan defaults con seq_len=20)
            "d_model": 512,
            "n_heads": 8,
            "d_ff": 2048,
            "e_layers": 2,
            "d_layers": 1,
            "modes": 10,  # clamped: max(1, seq_len//2) = 10
            "dropout": 0.1,
            "n_flow_layers": 4,
            "flow_hidden_dim": 64,
            "enc_in": 2,
            "dec_in": 2,
        },
    }
    torch.save(checkpoint, checkpoints / "nvda_canonical.pt")

    # 4. Crear preprocessing artifacts
    from sklearn.preprocessing import RobustScaler

    preproc_dir = checkpoints / "nvda_preprocessing"
    preproc_dir.mkdir()
    (preproc_dir / "schema.json").write_text(
        json.dumps(
            {
                "source_columns": ["Close", "Volume"],
                "feature_columns": ["Close", "Volume"],
                "target_features": ["Close"],
                "target_indices": [0],
                "numeric_columns": ["Close", "Volume"],
                "categorical_columns": [],
                "time_feature_columns": [],
                "onehot_columns": [],
                "category_mappings": {},
            }
        )
    )
    (preproc_dir / "metadata.json").write_text(
        json.dumps(
            {
                "fit_end_idx": 100,
                "fill_values": {},
                "outlier_bounds": {},
                "fit_stats": {},
                "return_transform": "none",
                "last_prices": {"Close": 100.0},
                "settings": {
                    "feature_roles": {},
                    "scaling_strategy": "robust",
                    "missing_policy": "impute_median",
                    "outlier_policy": "winsorize",
                    "fit_scope": "fold_train_only",
                    "persist_artifacts": False,
                    "drift_checks": {"enabled": False},
                    "strict_mode": False,
                    "categorical_encoding": "none",
                    "time_features": [],
                    "artifact_dir": "reports/preprocessing",
                    "return_transform": "none",
                },
            }
        )
    )
    scaler = RobustScaler()
    scaler.fit(rng.standard_normal((50, 2)))
    with (preproc_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    # 5. Crear registry JSON — file apunta al CSV en tmp_path
    registry = {
        "version": "1.0",
        "specialists": {
            "NVDA": {
                "checkpoint": str(checkpoints / "nvda_canonical.pt"),
                "config": checkpoint["config"],
                "data": {
                    "file": str(csv_path),
                    "rows": n_rows,
                    "features": 2,
                    "preprocessing_artifacts": str(preproc_dir),
                },
                "metrics": {"sharpe": 1.06},
            }
        },
    }
    registry_path = checkpoints / "model_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2))

    return registry_path


def _rewrite_registry_ticker_key(registry_path, new_key: str) -> None:
    """Reescribe la clave del especialista para probar compatibilidad por casing."""
    registry = json.loads(registry_path.read_text())
    specialist = registry["specialists"].pop("NVDA")
    registry["specialists"][new_key] = specialist
    registry_path.write_text(json.dumps(registry, indent=2))


def test_predict_returns_forecast_output(mock_registry):
    """predict() retorna ForecastOutput con shapes correctos."""
    from inference.loader import load_specialist
    from inference.predictor import predict
    from training.forecast_output import ForecastOutput

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)

    # CSV sintético con suficientes filas (seq_len=20, pred_len=4)
    rng = np.random.default_rng(99)
    n_rows = config.seq_len + config.pred_len + 10
    csv_path = mock_registry.parent / "test_data.csv"
    pd.DataFrame(
        {
            "Close": np.cumsum(rng.standard_normal(n_rows)) + 100,
            "Volume": rng.integers(1000, 10000, n_rows).astype(float),
        }
    ).to_csv(csv_path, index=False)

    forecast = predict(
        model=model,
        config=config,
        preprocessor=preprocessor,
        csv_path=str(csv_path),
        n_samples=3,  # mínimo para tests rápidos
    )

    assert isinstance(forecast, ForecastOutput)
    assert forecast.preds_real.size > 0
    assert forecast.quantiles_real is not None
    assert forecast.quantile_levels is not None
    assert forecast.preds_real.shape[1] == config.pred_len
    assert forecast.preds_real.shape[2] == len(config.target_features)


def test_predict_preserves_label_len(mock_registry):
    """predict() usa label_len del modelo entrenado, no el default del config."""
    from inference.loader import load_specialist
    from inference.predictor import _make_inference_config

    _, config, _ = load_specialist("NVDA", registry_path=mock_registry)
    # El mock usa label_len=10, seq_len=20 (seq_len//2 == label_len, latente)
    # Forzamos una discrepancia modificando label_len en config
    config.label_len = 7  # != seq_len//2 = 10

    csv_path = (
        mock_registry.parent.parent / "NVDA_features.csv"
    )  # tmp_path/NVDA_features.csv
    inference_cfg = _make_inference_config(config, str(csv_path))

    assert inference_cfg.label_len == 7, (
        f"label_len esperado=7, obtenido={inference_cfg.label_len}"
    )


def test_predict_uses_inference_default_mc_samples(
    mock_registry, monkeypatch: pytest.MonkeyPatch
):
    """predict() conserva 50 muestras por defecto y no hereda el knob del trainer."""
    from inference.loader import load_specialist
    from inference.predictor import predict

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)
    config.mc_dropout_eval_samples = 7

    captured: list[int] = []

    def fake_mc_dropout_inference(
        model, batch, n_samples, use_flow_sampling, mc_batch_size=1
    ):
        del model, use_flow_sampling, mc_batch_size
        captured.append(n_samples)
        batch_size = batch["x_enc"].shape[0]
        return torch.ones(
            n_samples,
            batch_size,
            config.pred_len,
            len(config.target_features),
        )

    monkeypatch.setattr(
        "inference.predictor.mc_dropout_inference", fake_mc_dropout_inference
    )

    rng = np.random.default_rng(99)
    n_rows = config.seq_len + config.pred_len + 10
    csv_path = mock_registry.parent / "test_default_samples.csv"
    pd.DataFrame(
        {
            "Close": np.cumsum(rng.standard_normal(n_rows)) + 100,
            "Volume": rng.integers(1000, 10000, n_rows).astype(float),
        }
    ).to_csv(csv_path, index=False)

    _ = predict(
        model=model,
        config=config,
        preprocessor=preprocessor,
        csv_path=str(csv_path),
    )

    assert captured
    assert all(sample_count == 50 for sample_count in captured)


def test_predict_does_not_refit_preprocessor(mock_registry, monkeypatch):
    """predict() no re-fittea el preprocessor — usa artefactos de entrenamiento."""
    from inference.loader import load_specialist
    from inference.predictor import predict

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)
    assert preprocessor.fitted, "El preprocessor debe estar fitted tras load_specialist"

    refit_calls = []
    original_fit = preprocessor.fit

    def tracking_fit(*args, **kwargs):
        refit_calls.append(1)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(preprocessor, "fit", tracking_fit)

    rng = np.random.default_rng(77)
    n_rows = config.seq_len + config.pred_len + 10
    csv_path = mock_registry.parent / "test_data2.csv"
    pd.DataFrame(
        {
            "Close": np.cumsum(rng.standard_normal(n_rows)) + 100,
            "Volume": rng.integers(1000, 10000, n_rows).astype(float),
        }
    ).to_csv(csv_path, index=False)

    predict(
        model=model,
        config=config,
        preprocessor=preprocessor,
        csv_path=str(csv_path),
        n_samples=3,
    )

    assert refit_calls == [], (
        f"preprocessor.fit fue llamado {len(refit_calls)} vez(ces) — no debe re-fitear en inferencia"
    )


def test_load_specialist_returns_model_config_preprocessor(mock_registry):
    """load_specialist retorna tupla (model, config, preprocessor)."""
    from inference.loader import load_specialist

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)

    assert model is not None
    # Verificar que config tiene los valores del registry
    assert config.seq_len == 20
    assert config.label_len == 10
    assert config.pred_len == 4
    assert config.target_features == ["Close"]
    # Verificar que preprocessor está fitted
    assert preprocessor.fitted is True
    # Verificar que el modelo está en eval mode
    assert not model.training


def test_load_specialist_accepts_lowercase_registry_key(mock_registry):
    """load_specialist resuelve claves legacy en minúsculas."""
    from inference.loader import load_specialist

    _rewrite_registry_ticker_key(mock_registry, "nvda")

    model, config, preprocessor = load_specialist("nvda", registry_path=mock_registry)

    assert model is not None
    assert config.label_len == 10
    assert preprocessor.fitted is True


def test_load_specialist_accepts_uppercase_query_for_lowercase_registry_key(
    mock_registry,
):
    """load_specialist debe resolver queries uppercase contra claves lowercase."""
    from inference.loader import load_specialist

    _rewrite_registry_ticker_key(mock_registry, "nvda")

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)

    assert model is not None
    assert config.target_features == ["Close"]
    assert preprocessor.fitted is True


def test_load_specialist_accepts_mixed_case_registry_key(mock_registry):
    """load_specialist resuelve claves mixed case sin exigir normalización previa."""
    from inference.loader import load_specialist

    _rewrite_registry_ticker_key(mock_registry, "NvDa")

    model, config, preprocessor = load_specialist("nvda", registry_path=mock_registry)

    assert model is not None
    assert config.pred_len == 4
    assert preprocessor.fitted is True


# ---------------------------------------------------------------------------
# Task 4: Tests del CLI (inference/__main__.py)
# ---------------------------------------------------------------------------


def test_inference_cli_help():
    """CLI de inference responde a --help sin errores."""
    result = subprocess.run(
        [sys.executable, "-m", "inference", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--ticker" in result.stdout
    assert "--csv" in result.stdout
    assert "--registry" in result.stdout


def test_inference_cli_list_models(mock_registry):
    """CLI --list-models muestra tickers del registry."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inference",
            "--list-models",
            "--registry",
            str(mock_registry),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "NVDA" in result.stdout


def test_inference_cli_unknown_ticker(mock_registry):
    """CLI falla con error claro para ticker no registrado."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inference",
            "--ticker",
            "FAKE",
            "--csv",
            "x.csv",
            "--registry",
            str(mock_registry),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Error" in result.stderr or "no registrado" in result.stderr


# ---------------------------------------------------------------------------
# Task 5: Edge case tests
# ---------------------------------------------------------------------------


def test_load_specialist_unknown_ticker_raises(mock_registry):
    """load_specialist lanza ValueError para ticker no registrado."""
    from inference.loader import load_specialist

    with pytest.raises(ValueError, match="no registrado"):
        load_specialist("FAKE", registry_path=mock_registry)


def test_predict_insufficient_data(mock_registry):
    """predict retorna ForecastOutput vacío si CSV tiene pocas filas."""
    from inference.loader import load_specialist
    from inference.predictor import predict

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)

    # CSV con 5 filas — insuficiente para seq_len=20 + pred_len=4
    csv_path = mock_registry.parent / "tiny_data.csv"
    pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        }
    ).to_csv(csv_path, index=False)

    forecast = predict(
        model=model,
        config=config,
        preprocessor=preprocessor,
        csv_path=str(csv_path),
        n_samples=3,
    )

    assert forecast.preds_real.size == 0


def test_forecast_output_quantile_levels(mock_registry):
    """ForecastOutput tiene quantile_levels [0.1, 0.5, 0.9]."""
    from inference.loader import load_specialist
    from inference.predictor import predict

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)

    rng = np.random.default_rng(55)
    n_rows = config.seq_len + config.pred_len + 10
    csv_path = mock_registry.parent / "enough_data.csv"
    pd.DataFrame(
        {
            "Close": np.cumsum(rng.standard_normal(n_rows)) + 100,
            "Volume": rng.integers(1000, 10000, n_rows).astype(float),
        }
    ).to_csv(csv_path, index=False)

    forecast = predict(
        model=model,
        config=config,
        preprocessor=preprocessor,
        csv_path=str(csv_path),
        n_samples=3,
    )

    # CSV tiene suficientes filas — exigir predicciones no vacías
    assert forecast.preds_real.size > 0, (
        "predict() retornó ForecastOutput vacío con CSV de suficientes filas"
    )
    np.testing.assert_array_almost_equal(
        forecast.quantile_levels,
        [0.1, 0.5, 0.9],
    )


# ---------------------------------------------------------------------------
# Bug fixes: P1 (short CSV), P2 (multi-target export), P3 (mean vs p50)
# ---------------------------------------------------------------------------


def test_pad_csv_for_forecast_returns_none_when_sufficient(mock_registry, tmp_path):
    """_pad_csv_for_forecast retorna None cuando el CSV ya tiene filas suficientes."""
    from inference.__main__ import _pad_csv_for_forecast

    csv_path = tmp_path / "long.csv"
    pd.DataFrame({"Close": range(50), "Volume": range(50)}).to_csv(
        csv_path, index=False
    )

    result = _pad_csv_for_forecast(str(csv_path), seq_len=20, pred_len=4)
    assert result is None


def test_pad_csv_for_forecast_pads_short_csv(tmp_path):
    """_pad_csv_for_forecast padea el CSV cuando tiene seq_len <= filas < seq_len+pred_len."""
    from inference.__main__ import _pad_csv_for_forecast
    import os

    # CSV con exactamente seq_len filas — falta pred_len para la ventana
    seq_len, pred_len = 20, 4
    csv_path = tmp_path / "short.csv"
    pd.DataFrame({"Close": range(seq_len), "Volume": range(seq_len)}).to_csv(
        csv_path, index=False
    )

    padded_path = _pad_csv_for_forecast(
        str(csv_path), seq_len=seq_len, pred_len=pred_len
    )
    try:
        assert padded_path is not None
        df_padded = pd.read_csv(padded_path)
        assert len(df_padded) == seq_len + pred_len
        # Las últimas pred_len filas son copia de la última fila original
        assert df_padded["Close"].iloc[-1] == seq_len - 1
    finally:
        if padded_path:
            os.unlink(padded_path)


def test_export_predictions_includes_all_targets(mock_registry, tmp_path):
    """_export_predictions escribe columnas para todos los targets, no solo el primero."""
    from inference.__main__ import _export_predictions
    from training.forecast_output import ForecastOutput
    from training.trainer import DEFAULT_QUANTILE_LEVELS

    n_windows, pred_len, n_targets = 2, 4, 2
    n_samples = 3
    rng = np.random.default_rng(42)

    samples = rng.standard_normal((n_samples, n_windows, pred_len, n_targets)).astype(
        np.float32
    )
    quantiles = np.quantile(samples, DEFAULT_QUANTILE_LEVELS, axis=0).astype(np.float32)

    forecast = ForecastOutput(
        preds_scaled=quantiles[1],
        gt_scaled=rng.standard_normal((n_windows, pred_len, n_targets)).astype(
            np.float32
        ),
        samples_scaled=samples,
        preds_real=quantiles[1].copy(),
        gt_real=rng.standard_normal((n_windows, pred_len, n_targets)).astype(
            np.float32
        ),
        samples_real=samples.copy(),
        quantiles_scaled=quantiles,
        quantiles_real=quantiles.copy(),
        quantile_levels=DEFAULT_QUANTILE_LEVELS.copy(),
        target_names=["Close", "Volume"],
    )

    output_path = tmp_path / "out.csv"
    _export_predictions(forecast, output_path)

    df = pd.read_csv(output_path)
    # Ambos targets deben aparecer en el CSV
    assert any("Close" in c for c in df.columns), f"Columnas: {df.columns.tolist()}"
    assert any("Volume" in c for c in df.columns), f"Columnas: {df.columns.tolist()}"


def test_export_predictions_uses_sample_mean_not_p50(mock_registry, tmp_path):
    """_export_predictions usa media muestral (no p50) para la columna de punto central."""
    from inference.__main__ import _export_predictions
    from training.forecast_output import ForecastOutput
    from training.trainer import DEFAULT_QUANTILE_LEVELS

    n_windows, pred_len, n_targets = 1, 4, 1
    n_samples = 3
    rng = np.random.default_rng(7)

    samples = rng.standard_normal((n_samples, n_windows, pred_len, n_targets)).astype(
        np.float32
    )
    quantiles = np.quantile(samples, DEFAULT_QUANTILE_LEVELS, axis=0).astype(np.float32)

    forecast = ForecastOutput(
        preds_scaled=quantiles[1],
        gt_scaled=np.zeros((n_windows, pred_len, n_targets), dtype=np.float32),
        samples_scaled=samples,
        preds_real=quantiles[1].copy(),
        gt_real=np.zeros((n_windows, pred_len, n_targets), dtype=np.float32),
        samples_real=samples.copy(),
        quantiles_scaled=quantiles,
        quantiles_real=quantiles.copy(),
        quantile_levels=DEFAULT_QUANTILE_LEVELS.copy(),
        target_names=["Close"],
    )

    output_path = tmp_path / "out.csv"
    _export_predictions(forecast, output_path)

    df = pd.read_csv(output_path)
    # No debe haber columna "pred_mean" (era incorrecto)
    assert "pred_mean" not in df.columns, "pred_mean debería haberse renombrado"
    # Debe haber columna de media basada en muestras
    assert any("mean" in c for c in df.columns), f"Columnas: {df.columns.tolist()}"


# ---------------------------------------------------------------------------
# Task 3: Tests de --plot y --output-dir en CLI
# ---------------------------------------------------------------------------


def test_cli_plot_flag_accepted():
    """CLI acepta --plot y --output-dir sin error de argparse."""
    result = subprocess.run(
        [sys.executable, "-m", "inference", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--plot" in result.stdout
    assert "--output-dir" in result.stdout


def test_cli_plot_generates_pngs(mock_registry, tmp_path, monkeypatch):
    """--plot genera fan_chart y calibration PNGs en --output-dir."""
    import os
    import shutil

    os.environ["MPLBACKEND"] = "Agg"

    # Crear CSV de predicciones sintético (lo que _export_predictions generaría)
    rng = np.random.default_rng(42)
    n_windows, pred_len = 3, 4
    rows = []
    for w in range(n_windows):
        for s in range(pred_len):
            q = np.sort(rng.normal(0, 0.02, 3))
            rows.append(
                {
                    "window": w,
                    "step": s,
                    "mean_Close": rng.normal(0, 0.02),
                    "gt_Close": rng.normal(0, 0.02),
                    "p10_Close": float(q[0]),
                    "p50_Close": float(q[1]),
                    "p90_Close": float(q[2]),
                }
            )
    csv_pred = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(csv_pred, index=False)

    output_dir = tmp_path / "plots"

    # Construir un ForecastOutput dummy para que main() no falle
    from training.forecast_output import ForecastOutput
    from training.trainer import DEFAULT_QUANTILE_LEVELS

    n_targets = 1
    n_samples = 3
    samples = rng.standard_normal((n_samples, n_windows, pred_len, n_targets)).astype(
        np.float32
    )
    quantiles = np.quantile(samples, DEFAULT_QUANTILE_LEVELS, axis=0).astype(np.float32)
    dummy_forecast = ForecastOutput(
        preds_scaled=quantiles[1],
        gt_scaled=np.zeros((n_windows, pred_len, n_targets), dtype=np.float32),
        samples_scaled=samples,
        preds_real=quantiles[1].copy(),
        gt_real=np.zeros((n_windows, pred_len, n_targets), dtype=np.float32),
        samples_real=samples.copy(),
        quantiles_scaled=quantiles,
        quantiles_real=quantiles.copy(),
        quantile_levels=DEFAULT_QUANTILE_LEVELS.copy(),
        target_names=["Close"],
    )

    # Parchear predict y _export_predictions para inyectar el CSV sintético
    import inference.__main__ as cli_module

    def fake_predict(**kwargs):
        return dummy_forecast

    def fake_export(forecast, output_path):
        """Copiar CSV sintético a la ruta esperada por main()."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(csv_pred, output_path)

    monkeypatch.setattr(cli_module, "predict", fake_predict)
    monkeypatch.setattr(cli_module, "_export_predictions", fake_export)

    # Parchear load_specialist para evitar cargar modelo real
    from inference.loader import load_specialist

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)
    monkeypatch.setattr(
        cli_module, "load_specialist", lambda *a, **kw: (model, config, preprocessor)
    )

    # Parchear _pad_csv_for_forecast para que retorne None (sin padding)
    monkeypatch.setattr(cli_module, "_pad_csv_for_forecast", lambda *a, **kw: None)

    # CSV path real (existe en mock_registry)
    csv_data = mock_registry.parent.parent / "NVDA_features.csv"

    # Ejecutar main con --plot
    # --output apunta a tmp_path para no contaminar results/ real
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference",
            "--ticker",
            "NVDA",
            "--csv",
            str(csv_data),
            "--registry",
            str(mock_registry),
            "--plot",
            "--output-dir",
            str(output_dir),
            "--output",
            str(tmp_path / "inference_nvda.csv"),
        ],
    )

    from inference.__main__ import main

    ret = main()

    assert ret == 0, "main() debe retornar 0"
    assert (output_dir / "fan_chart_nvda.png").exists(), "fan_chart PNG no generado"
    assert (output_dir / "calibration_nvda.png").exists(), "calibration PNG no generado"
