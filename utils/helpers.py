# -*- coding: utf-8 -*-
"""
Funciones estructurales auxiliares y abstracciones del sistema.
Refactorizado a estándares nativos del ecosistema 3.10+.
"""

import random

import numpy as np
import torch


def _select_amp_dtype() -> torch.dtype:
    """Extrae el formato empírico adecuado de precisión mixta basado en arquitectura del clúster."""
    try:
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            if (
                major >= 8
            ):  # Ampere en adelante toleran numéricos BFloat16 sin saturación
                return torch.bfloat16
        return torch.float16
    except (RuntimeError, AttributeError):
        return torch.float16


def setup_cuda_optimizations() -> None:
    """Acelera multiplicaciones matriciales profundas encadenando tensores (TF32)."""
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except (RuntimeError, AttributeError):
        pass

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def get_device() -> torch.device:
    """Asigna inmutablemente el recurso acelerador si es descubierto durante el runtime."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Inyecta un random state asfixiantemente exacto en CPython, NumPy, y CuDNN.

    Args:
        seed: Identificador hash natural base (42).
        deterministic: Fuerza estática matemática global en detrimento de sub-rutinas paralelas (lento).
    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except (RuntimeError, ValueError):
        # Defensivo extremo ante fallas abstractas C++ backend
        pass
