# -*- coding: utf-8 -*-
"""
Métricas probabilísticas puras para evaluación de distribuciones de predicción.

Funciones sin estado, sin imports de torch, compatibles con numpy.

Shapes esperados del pipeline FEDformer:
  y_true    : (n_windows, pred_len, n_targets)
  quantiles : (n_q, n_windows, pred_len, n_targets)
  samples   : (n_samples, n_windows, pred_len, n_targets)
"""

from __future__ import annotations

import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball loss (quantile loss) para el cuantil q.

    Args:
        y_true: Valores reales, shape arbitrario.
        y_pred: Predicción del cuantil q, mismo shape que y_true.
        q: Nivel del cuantil en (0, 1).

    Returns:
        Media de la pérdida sobre todos los elementos (escalar).
    """
    diff = y_true - y_pred
    loss = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def multi_quantile_pinball_loss(
    y_true: np.ndarray,
    quantiles: np.ndarray,
    levels: np.ndarray,
) -> dict[str, float]:
    """Pinball loss para múltiples cuantiles.

    Args:
        y_true: Ground truth, shape (n_windows, pred_len, n_targets).
        quantiles: Cuantiles predichos, shape (n_q, n_windows, pred_len, n_targets).
        levels: Niveles de cuantil, shape (n_q,).

    Returns:
        Dict {"pinball_pXX": valor} donde XX es el percentil entero (10, 50, 90, etc.).
    """
    return {
        f"pinball_p{int(round(float(q) * 100))}": pinball_loss(
            y_true, quantiles[i], float(q)
        )
        for i, q in enumerate(levels)
    }


def empirical_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fracción de observaciones que caen en el intervalo [lower, upper].

    Args:
        y_true: Valores reales, shape (n_windows, pred_len, n_targets).
        lower: Límite inferior del intervalo, mismo shape que y_true.
        upper: Límite superior del intervalo, mismo shape que y_true.

    Returns:
        Cobertura empírica en [0, 1].
    """
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Anchura media del intervalo de predicción (sharpness inversa).

    Args:
        lower: Límite inferior, shape arbitrario.
        upper: Límite superior, mismo shape que lower.

    Returns:
        Anchura media (escalar).
    """
    return float(np.mean(upper - lower))


def calibration_gap(
    y_true: np.ndarray,
    quantiles: np.ndarray,
    levels: np.ndarray,
) -> dict[str, float]:
    """Gap de calibración: |fracción empírica por debajo del cuantil - q|.

    Un modelo bien calibrado tiene gap_q → 0 para todos los niveles.

    Args:
        y_true: Ground truth, shape (n_windows, pred_len, n_targets).
        quantiles: Cuantiles predichos, shape (n_q, n_windows, pred_len, n_targets).
        levels: Niveles de cuantil, shape (n_q,).

    Returns:
        Dict {"calibration_gap_pXX": valor} para cada nivel.
    """
    result: dict[str, float] = {}
    for i, q in enumerate(levels):
        fraction_below = float(np.mean(y_true <= quantiles[i]))
        key = f"calibration_gap_p{int(round(float(q) * 100))}"
        result[key] = abs(fraction_below - float(q))
    return result


def coverage_by_quantile_pair(
    y_true: np.ndarray,
    quantiles: np.ndarray,
    levels: np.ndarray,
    lower_q: float,
    upper_q: float,
) -> float:
    """Cobertura empírica del intervalo [Q_{lower_q}, Q_{upper_q}].

    Args:
        y_true: Ground truth, shape (n_windows, pred_len, n_targets).
        quantiles: Cuantiles predichos, shape (n_q, n_windows, pred_len, n_targets).
        levels: Niveles de cuantil, shape (n_q,).
        lower_q: Nivel del cuantil inferior (p.ej. 0.1).
        upper_q: Nivel del cuantil superior (p.ej. 0.9).

    Returns:
        Cobertura empírica en [0, 1].

    Raises:
        ValueError: Si lower_q o upper_q no están en levels.
    """

    def _find_idx(q: float) -> int:
        idx = int(np.argmin(np.abs(levels.astype(float) - q)))
        if abs(float(levels[idx]) - q) > 1e-6:
            raise ValueError(
                f"Cuantil {q} no encontrado en los niveles disponibles: {list(levels)}"
            )
        return idx

    return empirical_coverage(
        y_true, quantiles[_find_idx(lower_q)], quantiles[_find_idx(upper_q)]
    )


def crps_from_samples(y_true: np.ndarray, samples: np.ndarray) -> float:
    """Continuous Ranked Probability Score via aproximación Monte Carlo.

    CRPS(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]
    donde X, X' son muestras independientes de la distribución F.

    Para samples tipo Dirac (todas iguales a y_pred): CRPS = MAE(y_pred, y_true).

    Args:
        y_true: Ground truth, shape (n_windows, pred_len, n_targets).
        samples: Muestras del modelo, shape (n_samples, n_windows, pred_len, n_targets).

    Returns:
        CRPS medio (escalar).
    """
    # Término 1: E_F[|X - y|]
    term1 = float(np.mean(np.abs(samples - y_true[np.newaxis])))

    n_samples = samples.shape[0]
    if n_samples < 2:
        return term1

    # Término 2: 0.5 * E[|X - X'|] vía estimador Monte Carlo
    # Limitamos a n_sub muestras para eficiencia computacional
    n_sub = min(n_samples, 50)
    sub = samples[:n_sub]

    # E[|X - X'|] ≈ mean_i mean_j |sub[i] - sub[j]| (i=j contribuye 0)
    energy = 0.0
    for i in range(n_sub):
        energy += float(np.mean(np.abs(sub - sub[i : i + 1])))
    energy /= n_sub

    return float(term1 - 0.5 * energy)


def sharpness_from_quantiles(lower: np.ndarray, upper: np.ndarray) -> float:
    """Sharpness del intervalo de predicción.

    Alias semántico de interval_width. Un valor menor indica mayor sharpness
    (intervalos más estrechos).

    Args:
        lower: Límite inferior, shape arbitrario.
        upper: Límite superior, mismo shape que lower.

    Returns:
        Anchura media del intervalo (escalar).
    """
    return interval_width(lower, upper)
