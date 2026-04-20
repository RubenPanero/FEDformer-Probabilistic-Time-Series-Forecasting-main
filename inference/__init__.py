"""API de inferencia para modelos canónicos FEDformer."""


def load_specialist(*args, **kwargs):
    """Carga un especialista canónico registrado en el model registry.

    Este wrapper difiere la importación del loader real para evitar imports
    circulares en tiempo de carga del paquete `inference`.

    Args:
        *args: Argumentos posicionales reenviados a `inference.loader.load_specialist`.
        **kwargs: Argumentos con nombre reenviados al loader real.

    Returns:
        El mismo valor retornado por `inference.loader.load_specialist`.
    """
    from inference.loader import load_specialist as _load

    return _load(*args, **kwargs)


def predict(*args, **kwargs):
    """Ejecuta la API pública de predicción probabilística del paquete.

    Este wrapper mantiene una superficie de importación estable y retrasa la
    carga de `inference.predictor` hasta el momento de uso.

    Args:
        *args: Argumentos posicionales reenviados a `inference.predictor.predict`.
        **kwargs: Argumentos con nombre reenviados a la función real.

    Returns:
        El mismo valor retornado por `inference.predictor.predict`.
    """
    from inference.predictor import predict as _predict  # pylint: disable=import-error,no-name-in-module

    return _predict(*args, **kwargs)


__all__ = ["load_specialist", "predict"]
