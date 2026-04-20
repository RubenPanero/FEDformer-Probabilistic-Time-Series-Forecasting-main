# -*- coding: utf-8 -*-
"""Registro de modelos especialistas por ticker."""

import json
import logging
import shutil
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("checkpoints/model_registry.json")


def load_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict:
    """Carga el registry desde disco. Retorna dict vacío si no existe.

    Args:
        registry_path: Ruta al archivo JSON del registry.

    Returns:
        Diccionario con la estructura del registry, o dict vacío si no existe.
    """
    if not registry_path.exists():
        return {}
    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict, registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """Guarda el registry en disco con indentación legible (indent=2).

    Args:
        registry: Diccionario con la estructura del registry.
        registry_path: Ruta destino del archivo JSON.
    """
    # Crear directorio padre si no existe
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def register_specialist(
    ticker: str,
    checkpoint_src: Path,
    metrics: dict,
    config_dict: dict,
    data_info: dict,
    training_command: str = "",
    notes: str = "",
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    canonical_dir: Path | None = None,
) -> Path:
    """Registra un especialista en el registry y copia su checkpoint al nombre canónico.

    El checkpoint se copia a canonical_dir/{ticker.lower()}_canonical.pt.
    Si canonical_dir es None, se usa Path("checkpoints").
    Si el ticker ya existe en el registry, actualiza su entrada.

    Args:
        ticker: Símbolo del activo financiero (ej. "NVDA").
        checkpoint_src: Ruta al checkpoint fuente del modelo entrenado.
        metrics: Diccionario con métricas de rendimiento (sharpe, sortino, etc.).
        config_dict: Diccionario con los parámetros de configuración del entrenamiento.
        data_info: Diccionario con información sobre el dataset utilizado.
        training_command: Comando CLI completo usado para reproducir el entrenamiento.
        notes: Notas adicionales sobre el especialista.
        registry_path: Ruta al archivo JSON del registry.
        canonical_dir: Directorio donde guardar el checkpoint canónico.
                       Si es None, usa Path("checkpoints").

    Returns:
        Ruta al checkpoint canónico guardado.
    """
    # 1. Cargar registry existente
    registry = load_registry(registry_path)

    # Inicializar estructura base si no existe
    if "specialists" not in registry:
        registry["specialists"] = {}
    if "version" not in registry:
        registry["version"] = "1.0"

    # 2. Determinar directorio canónico y copiar checkpoint
    dest_dir = canonical_dir if canonical_dir is not None else Path("checkpoints")
    dest_dir.mkdir(parents=True, exist_ok=True)
    canonical_filename = f"{ticker.lower()}_canonical.pt"
    canonical_path = dest_dir / canonical_filename
    shutil.copy2(checkpoint_src, canonical_path)
    logger.info("Checkpoint copiado a: %s", canonical_path)

    # 3. Construir entrada del especialista siguiendo la estructura del registry existente
    specialist_entry = {
        "checkpoint": str(canonical_path),
        "trained_at": date.today().isoformat(),
        "config": config_dict,
        "metrics": metrics,
        "data": data_info,
        "training_command": training_command,
        "notes": notes,
    }

    # 4. Actualizar registry["specialists"][ticker]
    registry["specialists"][ticker] = specialist_entry

    # 5. Actualizar registry["last_updated"] con la fecha de hoy
    registry["last_updated"] = date.today().isoformat()

    # 6. Guardar registry
    save_registry(registry, registry_path)

    # 7. Log del registro
    logger.info("Especialista '%s' registrado en %s", ticker, registry_path)

    # 8. Retornar Path del canonical checkpoint
    return canonical_path


def get_specialist(
    ticker: str, registry_path: Path = DEFAULT_REGISTRY_PATH
) -> dict | None:
    """Retorna la entrada del ticker en el registry, o None si no existe.

    Args:
        ticker: Símbolo del activo financiero (ej. "NVDA").
        registry_path: Ruta al archivo JSON del registry.

    Returns:
        Diccionario con la entrada del especialista, o None si no está registrado.
    """
    registry = load_registry(registry_path)
    return registry.get("specialists", {}).get(ticker)


def list_specialists(registry_path: Path = DEFAULT_REGISTRY_PATH) -> list[str]:
    """Lista los tickers registrados en el registry.

    Args:
        registry_path: Ruta al archivo JSON del registry.

    Returns:
        Lista de tickers registrados (puede estar vacía).
    """
    registry = load_registry(registry_path)
    return list(registry.get("specialists", {}).keys())
