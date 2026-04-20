"""Carga de modelos canónicos desde el model_registry."""

import logging
from pathlib import Path

import numpy as np
import torch

from config import FEDformerConfig
from data.preprocessing import PreprocessingPipeline
from models.fedformer import Flow_FEDformer
from utils.model_registry import (
    DEFAULT_REGISTRY_PATH,
    get_specialist,
    list_specialists,
    load_registry,
)

logger = logging.getLogger(__name__)


def load_specialist(
    ticker: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> tuple[Flow_FEDformer, FEDformerConfig, PreprocessingPipeline]:
    """Carga un modelo canónico con su config y preprocessor desde el registry.

    Args:
        ticker: Símbolo del activo financiero (ej. "NVDA").
        registry_path: Ruta al model_registry.json.

    Returns:
        Tupla (modelo, config, preprocessor) listos para inferencia.

    Raises:
        ValueError: Si el ticker no está registrado.
        FileNotFoundError: Si el checkpoint o artefactos no existen.
    """
    ticker, entry = _resolve_registry_entry(ticker, registry_path=registry_path)
    if entry is None:
        raise ValueError(
            f"Ticker '{ticker}' no registrado. "
            f"Disponibles: {available_tickers(registry_path)}"
        )

    # Reconstruir config — necesita file_path a un CSV real
    config = _build_config(entry)

    # Cargar modelo
    checkpoint_path = Path(entry["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    model = _load_model(config, checkpoint_path)

    # Cargar preprocessor
    artifacts_path = entry.get("data", {}).get("preprocessing_artifacts")
    if artifacts_path is None:
        raise FileNotFoundError(
            f"Artefactos de preprocessing no registrados para '{ticker}'. "
            "Re-entrena con --save-canonical para generarlos."
        )
    preprocessor = _load_preprocessor(config, Path(artifacts_path))

    return model, config, preprocessor


def _resolve_registry_entry(
    ticker: str, registry_path: Path = DEFAULT_REGISTRY_PATH
) -> tuple[str, dict | None]:
    """Resuelve el ticker contra el registry sin perder compatibilidad por casing."""
    entry = get_specialist(ticker, registry_path=registry_path)
    if entry is not None:
        return ticker, entry

    registry = load_registry(registry_path)
    specialists = registry.get("specialists", {})
    normalized = ticker.casefold()
    matches = [key for key in specialists if key.casefold() == normalized]

    if len(matches) > 1:
        raise ValueError(
            f"Ticker ambiguo '{ticker}' en registry: {matches}. "
            "Normaliza las claves del registry para evitar colisiones por casing."
        )
    if len(matches) == 1:
        resolved = matches[0]
        return resolved, specialists[resolved]
    return ticker, None


def _validated_file_path(data_info: dict) -> str:
    """Valida que file_path del registry exista antes de pasarlo a FEDformerConfig."""
    file_path = data_info.get("file", "")
    if not file_path or not Path(file_path).exists():
        raise FileNotFoundError(
            f"CSV de entrenamiento no encontrado: '{file_path}'. "
            "El registry apunta a un archivo que ya no existe."
        )
    return file_path


def _build_config(entry: dict) -> FEDformerConfig:
    """Reconstruye FEDformerConfig desde el dict del registry.

    Lee target_features del config guardado — no asume siempre 'Close'.
    file_path apunta al CSV original para que __init__ detecte enc_in/dec_in.
    """
    saved_config = entry.get("config", {})
    data_info = entry.get("data", {})
    target_features = saved_config.get("target_features", ["Close"])

    config = FEDformerConfig(
        target_features=target_features,
        file_path=_validated_file_path(data_info),
        seq_len=saved_config.get("seq_len", 96),
        label_len=saved_config.get("label_len", 48),
        pred_len=saved_config.get("pred_len", 20),
        batch_size=saved_config.get("batch_size", 64),
        gradient_clip_norm=saved_config.get("gradient_clip_norm", 0.5),
        return_transform=saved_config.get("return_transform", "log_return"),
        metric_space=saved_config.get("metric_space", "returns"),
        seed=saved_config.get("seed", 7),
        # Parámetros de arquitectura — restauran el modelo exacto usado en entrenamiento
        d_model=saved_config.get("d_model", 512),
        n_heads=saved_config.get("n_heads", 8),
        d_ff=saved_config.get("d_ff", 2048),
        e_layers=saved_config.get("e_layers", 2),
        d_layers=saved_config.get("d_layers", 1),
        modes=saved_config.get("modes", 64),
        dropout=saved_config.get("dropout", 0.1),
        n_flow_layers=saved_config.get("n_flow_layers", 4),
        flow_hidden_dim=saved_config.get("flow_hidden_dim", 64),
    )

    # __post_init__ deduce enc_in/dec_in del CSV actual, que puede diferir del
    # entrenamiento (ej. columna 'date' contada como feature si date_column no
    # coincide). Sobreescribir con los valores guardados en el registry.
    if "enc_in" in saved_config:
        config.enc_in = saved_config["enc_in"]
        config.dec_in = saved_config.get("dec_in", saved_config["enc_in"])

    return config


def _load_model(config: FEDformerConfig, checkpoint_path: Path) -> Flow_FEDformer:
    """Carga pesos del modelo desde un checkpoint canónico."""
    import numpy._core.multiarray as _npcma  # pylint: disable=import-outside-toplevel

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Flow_FEDformer(config).to(dev, non_blocking=True)

    with torch.serialization.safe_globals(
        [_npcma.scalar, np.float64, np.float32, np.int64, np.int32, np.bool_]  # pylint: disable=no-member
    ):
        checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(
        "Modelo cargado desde %s (epoch=%s, fold=%s)",
        checkpoint_path,
        checkpoint.get("epoch", "?"),
        checkpoint.get("fold", "?"),
    )
    return model


def _load_preprocessor(
    config: FEDformerConfig, artifacts_path: Path
) -> PreprocessingPipeline:
    """Reconstruye PreprocessingPipeline desde artefactos guardados."""
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Directorio de artefactos no encontrado: {artifacts_path}"
        )

    preprocessor = PreprocessingPipeline(
        config=config,
        target_features=list(config.target_features),
    )
    preprocessor.load_artifacts(artifacts_path)
    logger.info("Preprocessor cargado desde %s", artifacts_path)
    return preprocessor


def available_tickers(registry_path: Path = DEFAULT_REGISTRY_PATH) -> list[str]:
    """Lista tickers disponibles en el registry."""
    return list_specialists(registry_path=registry_path)
