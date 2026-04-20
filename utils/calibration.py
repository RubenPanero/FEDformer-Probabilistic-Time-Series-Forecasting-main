# -*- coding: utf-8 -*-
"""
Asistentes de calibración conforme (Conformal Calibration) para predicciones estocásticas.
Refactorizado a Python 3.10+ para garantizar typing nativo purificado, eficiencia y PEP 8.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1
) -> float:
    """Computa el cuantil residual dual para particiones analíticas."""
    if not 0 < alpha < 1:
        raise ValueError(
            f"El parámetro Alpha debe encontrarse en (0, 1), recibido {alpha}"
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Las matrices y_true e y_pred deben estructurarse asimétricamente idénticas"
        )

    residuals = np.abs(y_true - y_pred).reshape(-1)
    if residuals.size == 0:
        raise ValueError(
            "Las matrices de calibración estocásticas carecen de vectores empíricos"
        )

    n = residuals.size
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, q_level, method="higher"))


def apply_conformal_interval(
    y_pred: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construye un canal paramétrico [Low, High] envolviendo predicciones directas."""
    if q_hat < 0:
        raise ValueError(
            f"El estimador hat de Cuantil debe ser positivo, pero se computó: {q_hat}"
        )

    lower = y_pred - q_hat
    upper = y_pred + q_hat
    return lower, upper


def conformal_calibration_walkforward(
    residuals_by_fold: dict[int, np.ndarray],
    alpha: float = 0.2,
) -> dict[int, float | None]:
    """Calibración conformal walk-forward fold-aware.

    Para cada fold k, calcula q_hat usando residuos de folds 0..k-1.
    Fold 0 retorna None (sin datos de calibración previos).

    Args:
        residuals_by_fold: Diccionario {fold_id: array de residuos del fold}.
        alpha: Nivel de error (0.2 → cobertura nominal 80%).

    Returns:
        Diccionario {fold_id: q_hat calculado, o None si no hay datos previos}.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"El parámetro alpha debe estar en (0, 1), recibido {alpha}")

    sorted_folds = sorted(residuals_by_fold.keys())
    result: dict[int, float | None] = {}

    for idx, fold_k in enumerate(sorted_folds):
        if idx == 0:
            result[fold_k] = None
            continue

        prev_folds = sorted_folds[:idx]
        calibration_residuals = np.concatenate(
            [residuals_by_fold[f].reshape(-1) for f in prev_folds]
        )

        if calibration_residuals.size == 0:
            result[fold_k] = None
            continue

        n = calibration_residuals.size
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        result[fold_k] = float(
            np.quantile(calibration_residuals, q_level, method="higher")
        )

    return result
