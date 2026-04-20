# -*- coding: utf-8 -*-
"""Visualización probabilística para predicciones FEDformer."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Colores del diseño aprobado
_COLOR_PRED = "#4fc3f7"  # Azul — predicciones / intervalos
_COLOR_GT = "#ff7043"  # Naranja — ground truth


def _detectar_target(df: pd.DataFrame) -> str:
    """Detecta el nombre del target a partir de los nombres de columna.

    Args:
        df: DataFrame con columnas del tipo mean_*, gt_*, p10_*, etc.

    Returns:
        Nombre del target (p.ej. "Close").

    Raises:
        ValueError: Si no existe ninguna columna con prefijo "mean_".
    """
    targets = [c.replace("mean_", "") for c in df.columns if c.startswith("mean_")]
    if not targets:
        raise ValueError(
            "El DataFrame no contiene ninguna columna con prefijo 'mean_'. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    return targets[0]


def _a_pct_cambio(serie: np.ndarray) -> np.ndarray:
    """Convierte log-returns a variación porcentual (%).

    Fórmula: pct = (e^log_return - 1) * 100

    Args:
        serie: Array de valores en espacio log-return.

    Returns:
        Array de valores en porcentaje.
    """
    return np.expm1(serie) * 100.0


def plot_fan_chart(df: pd.DataFrame, ticker: str) -> Figure:
    """Fan chart con 2 subplots: rolling overview + zoom última ventana.

    Args:
        df: DataFrame con columnas window, step, mean_*, gt_*, p10_*, p50_*, p90_*
            Valores en espacio log-return.
        ticker: Nombre del ticker para títulos y etiquetas.

    Returns:
        Figure de matplotlib lista para guardar.
    """
    target = _detectar_target(df)

    # Columnas del target detectado
    col_p10 = f"p10_{target}"
    col_p50 = f"p50_{target}"
    col_p90 = f"p90_{target}"
    col_gt = f"gt_{target}"

    # ── Subplot superior: rolling overview (step=0 de cada ventana) ──────────
    df_step0 = df[df["step"] == 0].copy().sort_values("window").reset_index(drop=True)

    x_roll = np.arange(len(df_step0))
    p10_roll = _a_pct_cambio(df_step0[col_p10].to_numpy())
    p50_roll = _a_pct_cambio(df_step0[col_p50].to_numpy())
    p90_roll = _a_pct_cambio(df_step0[col_p90].to_numpy())
    gt_roll = _a_pct_cambio(df_step0[col_gt].to_numpy())

    # ── Subplot inferior: zoom última ventana (todos los steps) ──────────────
    last_window = int(df["window"].max())
    df_last = (
        df[df["window"] == last_window]
        .copy()
        .sort_values("step")
        .reset_index(drop=True)
    )

    x_last = df_last["step"].to_numpy()
    p10_last = _a_pct_cambio(df_last[col_p10].to_numpy())
    p50_last = _a_pct_cambio(df_last[col_p50].to_numpy())
    p90_last = _a_pct_cambio(df_last[col_p90].to_numpy())
    gt_last = _a_pct_cambio(df_last[col_gt].to_numpy())

    # ── Figura ────────────────────────────────────────────────────────────────
    with plt.style.context("dark_background"):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8))

        # — Subplot superior —
        ax_top.fill_between(
            x_roll,
            p10_roll,
            p90_roll,
            alpha=0.3,
            color=_COLOR_PRED,
            label="Banda p10–p90",
        )
        ax_top.plot(
            x_roll, p50_roll, color=_COLOR_PRED, linewidth=1.2, label="p50 (mediana)"
        )
        ax_top.plot(
            x_roll,
            gt_roll,
            color=_COLOR_GT,
            linewidth=1.0,
            linestyle="dotted",
            label="Ground truth",
        )
        ax_top.set_title(
            f"Fan Chart — {ticker} | Rolling Overview (step=0)", fontsize=11
        )
        ax_top.set_xlabel("Ventana")
        ax_top.set_ylabel("Variación % (log-return)")
        ax_top.legend(fontsize=8)

        # — Subplot inferior —
        ax_bot.fill_between(
            x_last,
            p10_last,
            p90_last,
            alpha=0.3,
            color=_COLOR_PRED,
            label="Banda p10–p90",
        )
        ax_bot.plot(
            x_last, p50_last, color=_COLOR_PRED, linewidth=1.5, label="p50 (mediana)"
        )
        ax_bot.plot(
            x_last,
            gt_last,
            color=_COLOR_GT,
            linewidth=1.5,
            linestyle="dotted",
            label="Ground truth",
        )
        ax_bot.set_title(f"Última Ventana (window {last_window})", fontsize=11)
        ax_bot.set_xlabel("Step")
        ax_bot.set_ylabel("Variación % (log-return)")
        ax_bot.legend(fontsize=8)

        fig.tight_layout()
    return fig


def plot_calibration(df: pd.DataFrame, ticker: str) -> Figure:
    """Calibration con 2 subplots: reliability diagram + PIT histogram.

    Args:
        df: DataFrame con columnas window, step, mean_*, gt_*, p10_*, p50_*, p90_*
            Valores en espacio log-return.
        ticker: Nombre del ticker para títulos y etiquetas.

    Returns:
        Figure de matplotlib lista para guardar.
    """
    target = _detectar_target(df)

    # Columnas del target detectado
    col_p10 = f"p10_{target}"
    col_p50 = f"p50_{target}"
    col_p90 = f"p90_{target}"
    col_gt = f"gt_{target}"

    p10 = df[col_p10].to_numpy()
    p50 = df[col_p50].to_numpy()
    p90 = df[col_p90].to_numpy()
    gt = df[col_gt].to_numpy()

    # ── Reliability diagram ────────────────────────────────────────────────
    # Niveles nominales y cobertura empírica observada
    niveles_nominales = np.array([0.1, 0.5, 0.9])
    cobertura_empirica = np.array(
        [
            float(np.mean(gt <= p10)),
            float(np.mean(gt <= p50)),
            float(np.mean(gt <= p90)),
        ]
    )

    # ── PIT scores ────────────────────────────────────────────────────────
    # Cálculo por fila: interpola el GT sobre los cuantiles de esa observación.
    # Usamos np.sort para garantizar xp no-decreciente (requisito de np.interp).
    # left=0.0 / right=1.0: GT por debajo de p10 → 0.0, GT por encima de p90 → 1.0,
    # lo que permite que el histograma muestre masa en las colas [0, 0.1) y (0.9, 1].
    pit_scores = np.array(
        [
            float(
                np.interp(
                    gt[i],
                    np.sort([p10[i], p50[i], p90[i]]),
                    [0.1, 0.5, 0.9],
                    left=0.0,
                    right=1.0,
                )
            )
            for i in range(len(gt))
        ]
    )

    # ── Figura ────────────────────────────────────────────────────────────────
    with plt.style.context("dark_background"):
        fig, (ax_rel, ax_pit) = plt.subplots(1, 2, figsize=(12, 5))

        # — Reliability diagram —
        # Diagonal de calibración perfecta
        ax_rel.plot(
            [0, 1],
            [0, 1],
            linestyle="dashed",
            color="gray",
            linewidth=1.0,
            label="Calibración perfecta",
        )
        ax_rel.scatter(
            niveles_nominales,
            cobertura_empirica,
            color=_COLOR_PRED,
            s=80,
            zorder=5,
            label="Cobertura empírica",
        )
        ax_rel.plot(
            niveles_nominales, cobertura_empirica, color=_COLOR_PRED, linewidth=1.2
        )
        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1)
        ax_rel.set_title(f"Reliability Diagram — {ticker}", fontsize=11)
        ax_rel.set_xlabel("Nivel nominal")
        ax_rel.set_ylabel("Cobertura empírica")
        ax_rel.legend(fontsize=8)

        # — PIT histogram —
        ax_pit.hist(
            pit_scores,
            bins=10,
            range=(0.0, 1.0),
            color=_COLOR_PRED,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        # Línea de referencia uniforme
        n_total = len(pit_scores)
        altura_uniforme = n_total / 10.0
        ax_pit.axhline(
            altura_uniforme,
            color=_COLOR_GT,
            linestyle="dashed",
            linewidth=1.2,
            label="Uniforme (ideal)",
        )
        ax_pit.set_xlim(0, 1)
        ax_pit.set_title(f"PIT Histogram (approx) — {ticker}", fontsize=11)
        ax_pit.set_xlabel("PIT score")
        ax_pit.set_ylabel("Frecuencia")
        ax_pit.legend(fontsize=8)

        fig.tight_layout()
    return fig
