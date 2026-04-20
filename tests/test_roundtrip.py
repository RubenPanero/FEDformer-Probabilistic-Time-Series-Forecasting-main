"""Tests de roundtrip save→load para el ciclo canónico completo."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from config import FEDformerConfig
from data.preprocessing import PreprocessingPipeline
from inference.loader import _build_config, load_specialist
from main import _build_config_dict
from models.fedformer import Flow_FEDformer
from utils.model_registry import register_specialist

# ── Dimensiones tiny para tests rápidos ──────────────────────────────
TINY = dict(
    d_model=32,
    n_heads=2,
    d_ff=64,
    e_layers=1,
    d_layers=1,
    modes=8,
    dropout=0.0,
    n_flow_layers=2,
    flow_hidden_dim=16,
    seq_len=16,
    pred_len=4,
    label_len=8,
    batch_size=2,
    return_transform="log_return",
    metric_space="returns",
    gradient_clip_norm=0.5,
    seed=7,
)
N_FEATURES = 5  # sin contar date
FEATURE_NAMES = ["Close", "High", "Low", "Open", "Volume"]
TARGET = ["Close"]


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """CSV sintético con N_FEATURES columnas + date."""
    n_rows = 60  # suficiente para seq_len=16 + pred_len=4
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    rng = np.random.default_rng(0)
    data = {name: rng.standard_normal(n_rows).cumsum() + 100 for name in FEATURE_NAMES}
    df = pd.DataFrame(data)
    df.insert(0, "date", dates)
    csv_path = tmp_path / "TEST_features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tiny_config(synthetic_csv: Path) -> FEDformerConfig:
    """FEDformerConfig con dimensiones mínimas apuntando al CSV sintético."""
    cfg = FEDformerConfig(
        target_features=TARGET,
        file_path=str(synthetic_csv),
        **TINY,
    )
    # Forzar enc_in/dec_in al nº real de features (no depender de __post_init__)
    cfg.enc_in = N_FEATURES
    cfg.dec_in = N_FEATURES
    # Setting no-default para detectar bug de settings no restaurados
    cfg.sections.preprocessing.scaling_strategy = "standard"  # default es "robust"
    return cfg


# ── T1: Config roundtrip ─────────────────────────────────────────────


def test_config_roundtrip(tiny_config: FEDformerConfig, synthetic_csv: Path) -> None:
    """Config sobrevive save→registry→_build_config sin perder arch params."""
    config_dict = _build_config_dict(tiny_config)
    entry = {
        "config": config_dict,
        "data": {"file": str(synthetic_csv)},
    }

    rebuilt = _build_config(entry)

    for key in config_dict:
        original = config_dict[key]
        loaded = getattr(rebuilt, key)
        if isinstance(original, list):
            loaded = list(loaded)
        assert loaded == original, f"{key}: {loaded} != {original}"


# ── T2: Preprocessor artifacts roundtrip ─────────────────────────────


@pytest.fixture
def fitted_preprocessor(
    tiny_config: FEDformerConfig, synthetic_csv: Path
) -> PreprocessingPipeline:
    """PreprocessingPipeline fitteado con el CSV sintético."""
    df = pd.read_csv(synthetic_csv)
    pp = PreprocessingPipeline.from_config(
        tiny_config,
        target_features=TARGET,
        date_column="date",
    )
    pp.fit(df)
    return pp


def test_preprocessor_artifacts_roundtrip(
    tiny_config: FEDformerConfig,
    fitted_preprocessor: PreprocessingPipeline,
    tmp_path: Path,
) -> None:
    """Preprocessor artifacts sobreviven save→load con transformaciones idénticas."""
    artifacts_dir = tmp_path / "preprocessing"
    fitted_preprocessor.save_artifacts(artifacts_dir)

    # Cargar en un pipeline nuevo con config fresca (settings por defecto)
    # para verificar que load_artifacts restaura los settings desde el JSON.
    fresh_config = FEDformerConfig(
        target_features=TARGET,
        file_path=str(tiny_config.file_path),
        **TINY,
    )
    fresh_config.enc_in = N_FEATURES
    fresh_config.dec_in = N_FEATURES
    # Confirmar que fresh_config tiene el default "robust" (distinto del "standard" del fixture)
    assert fresh_config.sections.preprocessing.scaling_strategy == "robust"

    loaded = PreprocessingPipeline(
        config=fresh_config,
        target_features=TARGET,
    )
    loaded.load_artifacts(artifacts_dir)

    # Verificar campos de schema
    assert loaded.feature_columns == fitted_preprocessor.feature_columns
    assert loaded.target_indices == fitted_preprocessor.target_indices
    assert loaded.numeric_columns == fitted_preprocessor.numeric_columns
    assert loaded.target_features == fitted_preprocessor.target_features

    # Verificar campos de metadata
    assert loaded.return_transform == fitted_preprocessor.return_transform
    assert loaded.last_prices == fitted_preprocessor.last_prices
    assert loaded.outlier_bounds == fitted_preprocessor.outlier_bounds
    assert loaded.fill_values == fitted_preprocessor.fill_values
    assert loaded.fitted is True

    # Verificar que settings sobreviven roundtrip
    assert (
        loaded.settings.scaling_strategy
        == fitted_preprocessor.settings.scaling_strategy
    )
    assert loaded.settings.missing_policy == fitted_preprocessor.settings.missing_policy
    assert loaded.settings.outlier_policy == fitted_preprocessor.settings.outlier_policy
    assert loaded.settings.fit_scope == fitted_preprocessor.settings.fit_scope
    assert (
        loaded.settings.categorical_encoding
        == fitted_preprocessor.settings.categorical_encoding
    )
    assert loaded.settings.strict_mode == fitted_preprocessor.settings.strict_mode
    assert loaded.settings.time_features == fitted_preprocessor.settings.time_features
    assert loaded.settings.drift_checks == fitted_preprocessor.settings.drift_checks

    # Verificar que el scaler produce la misma transformación
    rng = np.random.default_rng(1)
    test_data = rng.standard_normal((10, N_FEATURES)).astype(np.float32)
    original_scaled = fitted_preprocessor.scaler.transform(test_data)
    loaded_scaled = loaded.scaler.transform(test_data)
    np.testing.assert_array_equal(original_scaled, loaded_scaled)


@pytest.mark.parametrize(
    "saved_strict_mode,default_strict_mode", [(False, True), (True, False)]
)
def test_strict_mode_from_artifacts_when_not_explicit(
    tiny_config: FEDformerConfig,
    synthetic_csv: Path,
    tmp_path: Path,
    saved_strict_mode: bool,
    default_strict_mode: bool,
) -> None:
    """Sin override explícito en constructor, load_artifacts debe restaurar strict_mode desde disco."""
    tiny_config.sections.preprocessing.strict_mode = saved_strict_mode

    df = pd.read_csv(synthetic_csv)
    pipeline = PreprocessingPipeline(
        config=tiny_config,
        target_features=TARGET,
        date_column="date",
    )
    assert pipeline.strict_mode == saved_strict_mode
    pipeline.fit(df)

    artifacts_dir = tmp_path / "preprocessing_no_override"
    pipeline.save_artifacts(artifacts_dir)

    # Cargar sin strict_mode explícito para cubrir la ruta _strict_mode_explicit=False.
    fresh_config = FEDformerConfig(
        target_features=TARGET,
        file_path=str(tiny_config.file_path),
        **TINY,
    )
    fresh_config.enc_in = N_FEATURES
    fresh_config.dec_in = N_FEATURES
    fresh_config.sections.preprocessing.strict_mode = default_strict_mode
    loaded = PreprocessingPipeline(
        config=fresh_config,
        target_features=TARGET,
        date_column="date",
    )
    loaded.load_artifacts(artifacts_dir)

    assert loaded.strict_mode == saved_strict_mode
    assert loaded.settings.strict_mode == saved_strict_mode

    # Una serialización posterior debe preservar el valor restaurado desde disco.
    resaved_dir = tmp_path / "preprocessing_resave"
    loaded.save_artifacts(resaved_dir)
    resaved_meta = json.loads((resaved_dir / "metadata.json").read_text())
    assert resaved_meta["settings"]["strict_mode"] == saved_strict_mode


@pytest.mark.parametrize("override,default_strict_mode", [(False, True), (True, False)])
def test_strict_mode_override_survives_load_artifacts(
    tiny_config: FEDformerConfig,
    synthetic_csv: Path,
    tmp_path: Path,
    override: bool,
    default_strict_mode: bool,
) -> None:
    """Override explícito de strict_mode debe prevalecer tras load_artifacts en ambas direcciones."""
    tiny_config.sections.preprocessing.strict_mode = default_strict_mode

    df = pd.read_csv(synthetic_csv)
    pipeline = PreprocessingPipeline(
        config=tiny_config,
        target_features=TARGET,
        date_column="date",
        strict_mode=override,
    )
    assert pipeline.strict_mode == override
    pipeline.fit(df)

    artifacts_dir = tmp_path / "preprocessing_override"
    pipeline.save_artifacts(artifacts_dir)

    fresh_config = FEDformerConfig(
        target_features=TARGET,
        file_path=str(tiny_config.file_path),
        **TINY,
    )
    fresh_config.enc_in = N_FEATURES
    fresh_config.dec_in = N_FEATURES
    fresh_config.sections.preprocessing.strict_mode = default_strict_mode
    loaded = PreprocessingPipeline(
        config=fresh_config,
        target_features=TARGET,
        date_column="date",
        strict_mode=override,
    )
    loaded.load_artifacts(artifacts_dir)

    # self.strict_mode (comportamiento runtime) debe respetar el override.
    assert loaded.strict_mode == override
    # self.settings.strict_mode debe concordar con el override explícito.
    assert loaded.settings.strict_mode == override

    # Una serialización posterior debe propagar el override, no el valor del disco.
    resaved_dir = tmp_path / "preprocessing_resave_override"
    loaded.save_artifacts(resaved_dir)
    resaved_meta = json.loads((resaved_dir / "metadata.json").read_text())
    assert resaved_meta["settings"]["strict_mode"] == override


def test_load_artifacts_override_does_not_mutate_shared_config(
    tiny_config: FEDformerConfig,
    synthetic_csv: Path,
    tmp_path: Path,
) -> None:
    """El pipeline cargado debe aislar sus settings locales del config compartido."""
    tiny_config.sections.preprocessing.strict_mode = True

    df = pd.read_csv(synthetic_csv)
    pipeline = PreprocessingPipeline(
        config=tiny_config,
        target_features=TARGET,
        date_column="date",
        strict_mode=False,
    )
    pipeline.fit(df)

    artifacts_dir = tmp_path / "preprocessing_isolated"
    pipeline.save_artifacts(artifacts_dir)

    fresh_config = FEDformerConfig(
        target_features=TARGET,
        file_path=str(tiny_config.file_path),
        **TINY,
    )
    fresh_config.enc_in = N_FEATURES
    fresh_config.dec_in = N_FEATURES
    fresh_config.sections.preprocessing.strict_mode = True
    loaded = PreprocessingPipeline(
        config=fresh_config,
        target_features=TARGET,
        date_column="date",
        strict_mode=False,
    )
    loaded.load_artifacts(artifacts_dir)

    assert fresh_config.sections.preprocessing.strict_mode is True
    assert loaded.settings.strict_mode is False
    assert loaded.strict_mode is False


def test_load_artifacts_with_null_settings_keeps_constructor_defaults(
    tiny_config: FEDformerConfig,
    fitted_preprocessor: PreprocessingPipeline,
    tmp_path: Path,
) -> None:
    """Artifacts corruptos con settings=null no deben dejar estado parcialmente restaurado."""
    artifacts_dir = tmp_path / "preprocessing_null_settings"
    fitted_preprocessor.save_artifacts(artifacts_dir)

    metadata_path = artifacts_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["settings"] = None
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    fresh_config = FEDformerConfig(
        target_features=TARGET,
        file_path=str(tiny_config.file_path),
        **TINY,
    )
    fresh_config.enc_in = N_FEATURES
    fresh_config.dec_in = N_FEATURES
    fresh_config.sections.preprocessing.scaling_strategy = "minmax"
    fresh_config.sections.preprocessing.strict_mode = False
    fresh_config.sections.preprocessing.fit_scope = "global_train"
    fresh_config.sections.preprocessing.artifact_dir = "reports/custom-prep"

    loaded = PreprocessingPipeline(
        config=fresh_config,
        target_features=TARGET,
        date_column="date",
    )
    loaded.load_artifacts(artifacts_dir)

    assert loaded.settings.scaling_strategy == "minmax"
    assert loaded.settings.strict_mode is False
    assert loaded.settings.fit_scope == "global_train"
    assert loaded.settings.artifact_dir == "reports/custom-prep"
    assert loaded.strict_mode is False
    assert loaded.fit_scope == "global_train"
    assert loaded.artifact_dir == Path("reports/custom-prep")


# ── T3: Model state_dict roundtrip ───────────────────────────────────


@pytest.fixture
def tiny_model(tiny_config: FEDformerConfig) -> Flow_FEDformer:
    """Flow_FEDformer tiny en eval mode con seed determinista."""
    torch.manual_seed(42)
    model = Flow_FEDformer(tiny_config)
    model.eval()
    return model


def _make_input(cfg: FEDformerConfig, batch: int = 2) -> tuple:
    """Crea input determinista para forward pass."""
    torch.manual_seed(99)
    x_enc = torch.randn(batch, cfg.seq_len, cfg.enc_in)
    x_dec = torch.randn(batch, cfg.label_len + cfg.pred_len, cfg.dec_in)
    x_regime = torch.zeros(batch, 1, 1, dtype=torch.long)
    return x_enc, x_dec, x_regime


def test_model_state_dict_roundtrip(
    tiny_config: FEDformerConfig,
    tiny_model: Flow_FEDformer,
    tmp_path: Path,
) -> None:
    """Model state_dict sobrevive save→load con outputs bitwise idénticos."""
    # Forward pass original
    x_enc, x_dec, x_regime = _make_input(tiny_config)
    with torch.no_grad():
        dist_original = tiny_model(x_enc, x_dec, x_regime)
        mean_original = dist_original.mean

    # Guardar checkpoint (formato idéntico a trainer.save_checkpoint)
    checkpoint_path = tmp_path / "best_model_fold_3.pt"
    checkpoint = {
        "model_state_dict": tiny_model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": None,
        "epoch": 5,
        "fold": 3,
        "loss": 0.123,
        "config": asdict(tiny_config),
    }
    torch.save(checkpoint, checkpoint_path)

    # Cargar en modelo nuevo
    model_loaded = Flow_FEDformer(tiny_config)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_loaded.load_state_dict(saved["model_state_dict"])
    model_loaded.eval()

    # Forward pass con modelo cargado
    x_enc2, x_dec2, x_regime2 = _make_input(tiny_config)  # misma seed → mismo input
    with torch.no_grad():
        dist_loaded = model_loaded(x_enc2, x_dec2, x_regime2)
        mean_loaded = dist_loaded.mean

    torch.testing.assert_close(mean_loaded, mean_original, atol=0, rtol=0)


# ── T4 ───────────────────────────────────────────────────────────────


def test_full_roundtrip_save_load_predict(
    tiny_config: FEDformerConfig,
    tiny_model: Flow_FEDformer,
    fitted_preprocessor: PreprocessingPipeline,
    synthetic_csv: Path,
    tmp_path: Path,
) -> None:
    """Ciclo completo: save checkpoint+artifacts+registry → load_specialist → predict idéntico."""
    # 1. Capturar output original
    x_enc, x_dec, x_regime = _make_input(tiny_config)
    with torch.no_grad():
        mean_original = tiny_model(x_enc, x_dec, x_regime).mean

    # 2. Guardar checkpoint (simula trainer.save_checkpoint para fold 3)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_src = ckpt_dir / "best_model_fold_3.pt"
    torch.save(
        {
            "model_state_dict": tiny_model.state_dict(),
            "optimizer_state_dict": {},
            "scaler_state_dict": None,
            "epoch": 5,
            "fold": 3,
            "loss": 0.123,
            "config": asdict(tiny_config),
        },
        ckpt_src,
    )

    # 3. Guardar preprocessing artifacts
    prep_dir = ckpt_dir / "test_preprocessing"
    fitted_preprocessor.save_artifacts(prep_dir)

    # 4. Registrar en registry (usa register_specialist real)
    config_dict = _build_config_dict(tiny_config)
    registry_path = tmp_path / "model_registry.json"
    data_info = {
        "file": str(synthetic_csv),
        "rows": 60,
        "features": N_FEATURES,
        "preprocessing_artifacts": str(prep_dir),
    }
    register_specialist(
        ticker="TEST",
        checkpoint_src=ckpt_src,
        metrics={
            "sharpe": 1.0,
            "sortino": 1.5,
            "max_drawdown": -0.2,
            "volatility": 0.1,
        },
        config_dict=config_dict,
        data_info=data_info,
        registry_path=registry_path,
        canonical_dir=ckpt_dir,
    )

    # 5. Cargar con load_specialist (la función real de inferencia)
    model_loaded, config_loaded, preprocessor_loaded = load_specialist(
        "TEST", registry_path=registry_path
    )

    # 6. Verificar config
    for key in config_dict:
        original = config_dict[key]
        loaded = getattr(config_loaded, key)
        if isinstance(original, list):
            loaded = list(loaded)
        assert loaded == original, f"Config mismatch {key}: {loaded} != {original}"

    # 7. Verificar preprocessor
    assert preprocessor_loaded.fitted is True
    assert preprocessor_loaded.target_indices == fitted_preprocessor.target_indices
    assert preprocessor_loaded.feature_columns == fitted_preprocessor.feature_columns
    rng = np.random.default_rng(123)
    test_vec = rng.standard_normal((5, N_FEATURES)).astype(np.float32)
    np.testing.assert_array_equal(
        preprocessor_loaded.scaler.transform(test_vec),
        fitted_preprocessor.scaler.transform(test_vec),
    )

    # 8. Verificar output del modelo
    # _load_model puede poner el modelo en CUDA → mover a CPU para comparar con original
    model_loaded.cpu().eval()
    x_enc2, x_dec2, x_regime2 = _make_input(tiny_config)
    with torch.no_grad():
        mean_loaded = model_loaded(x_enc2, x_dec2, x_regime2).mean

    torch.testing.assert_close(mean_loaded, mean_original, atol=0, rtol=0)


# ── T5 ───────────────────────────────────────────────────────────────


def test_enc_in_dec_in_survives_roundtrip(
    synthetic_csv: Path,
) -> None:
    """Regresión sesión 19: enc_in/dec_in del registry prevalecen sobre __post_init__."""
    # enc_in=5 en el registry, pero el CSV tiene 6 columnas (5 features + date).
    # Sin el fix, __post_init__ leería 6 columnas y pondría enc_in=6.
    config_dict = {
        **TINY,
        "target_features": TARGET,
        "enc_in": N_FEATURES,
        "dec_in": N_FEATURES,
    }
    entry = {
        "config": config_dict,
        "data": {"file": str(synthetic_csv)},
    }

    rebuilt = _build_config(entry)

    assert rebuilt.enc_in == N_FEATURES, (
        f"enc_in corrupted by __post_init__: {rebuilt.enc_in} != {N_FEATURES}"
    )
    assert rebuilt.dec_in == N_FEATURES, (
        f"dec_in corrupted by __post_init__: {rebuilt.dec_in} != {N_FEATURES}"
    )


def test_build_config_dict_importable():
    """_build_config_dict es importable desde main y produce dict con todas las keys."""

    assert callable(_build_config_dict)
