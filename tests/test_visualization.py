"""Tests para utils/visualization.py — fan charts y calibración probabilística."""

import os

# Configurar backend headless ANTES de cualquier import de matplotlib o visualization
os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.use("Agg")

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers de fixtures
# ---------------------------------------------------------------------------


def make_synthetic_df(n_windows: int = 5, pred_len: int = 20) -> pd.DataFrame:
    """Genera un DataFrame sintético con estructura de inferencia CSV.

    Args:
        n_windows: Número de ventanas de predicción.
        pred_len: Pasos por ventana (horizonte de predicción).

    Returns:
        DataFrame con columnas window, step, mean_Close, gt_Close,
        p10_Close, p50_Close, p90_Close.
    """
    rng = np.random.default_rng(seed=42)
    rows = []
    for w in range(n_windows):
        for s in range(pred_len):
            # Cuantiles ordenados: muestrear 3 valores y ordenar para garantizar p10 ≤ p50 ≤ p90
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
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixture compartida
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """DataFrame sintético reutilizable en todos los tests del módulo."""
    return make_synthetic_df(n_windows=5, pred_len=20)


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    """Cierra todas las figuras matplotlib después de cada test para evitar leak."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Tests: plot_fan_chart
# ---------------------------------------------------------------------------


def test_plot_fan_chart_returns_figure(synthetic_df: pd.DataFrame) -> None:
    """plot_fan_chart debe retornar un objeto matplotlib.figure.Figure."""
    from matplotlib.figure import Figure

    from utils.visualization import plot_fan_chart

    fig = plot_fan_chart(synthetic_df, ticker="TEST")
    assert isinstance(fig, Figure), (
        f"plot_fan_chart debe retornar Figure, obtuvo {type(fig)}"
    )


def test_plot_fan_chart_has_two_subplots(synthetic_df: pd.DataFrame) -> None:
    """El fan chart debe tener exactamente 2 ejes (rolling overview + zoom)."""
    from utils.visualization import plot_fan_chart

    fig = plot_fan_chart(synthetic_df, ticker="TEST")
    axes = fig.get_axes()
    assert len(axes) == 2, f"Fan chart debe tener 2 subplots, encontró {len(axes)}"


def test_plot_fan_chart_pct_scale(synthetic_df: pd.DataFrame) -> None:
    """Los datos graficados deben estar en escala % change, no en log-return crudo.

    Los valores en % change son ~100× mayores que los log-returns crudos
    (ej. 5-10% vs 0.05-0.10). Verificamos que al menos un dato visible
    supera 1.0 en valor absoluto (imposible en log-return normal de mercado).
    """
    from utils.visualization import plot_fan_chart

    # gt_Close tiene media 0, std ~0.02 → en % change: np.expm1(0.02)*100 ≈ 2.0%
    # Un rango de ±2% debe estar en el orden de magnitud de % (>0.5), no de log-returns
    fig = plot_fan_chart(synthetic_df, ticker="TEST")
    ax_top = fig.get_axes()[0]

    # Obtener datos de las líneas graficadas en el subplot superior
    lines = ax_top.get_lines()
    assert len(lines) > 0, "El subplot superior debe contener al menos una línea"

    # Recolectar todos los valores Y de las líneas
    all_y_values = np.concatenate(
        [line.get_ydata() for line in lines if len(line.get_ydata()) > 0]
    )

    # En log-return crudo, max(|y|) ≈ 0.1. En % change, max(|y|) > 1.0
    max_abs_y = np.max(np.abs(all_y_values))
    assert max_abs_y > 0.5, (
        f"Los valores del eje Y parecen estar en log-return crudo (max={max_abs_y:.4f}). "
        "Se esperan valores en % change (>0.5%)."
    )


# ---------------------------------------------------------------------------
# Tests: plot_calibration
# ---------------------------------------------------------------------------


def test_plot_calibration_returns_figure(synthetic_df: pd.DataFrame) -> None:
    """plot_calibration debe retornar un objeto matplotlib.figure.Figure."""
    from matplotlib.figure import Figure

    from utils.visualization import plot_calibration

    fig = plot_calibration(synthetic_df, ticker="TEST")
    assert isinstance(fig, Figure), (
        f"plot_calibration debe retornar Figure, obtuvo {type(fig)}"
    )


def test_plot_calibration_has_two_subplots(synthetic_df: pd.DataFrame) -> None:
    """El calibration plot debe tener exactamente 2 ejes (1×2 layout)."""
    from utils.visualization import plot_calibration

    fig = plot_calibration(synthetic_df, ticker="TEST")
    axes = fig.get_axes()
    assert len(axes) == 2, (
        f"Calibration plot debe tener 2 subplots, encontró {len(axes)}"
    )


def test_plot_calibration_reliability_diagram(synthetic_df: pd.DataFrame) -> None:
    """El reliability diagram debe tener exactamente 3 puntos (p10, p50, p90)."""
    from utils.visualization import plot_calibration

    fig = plot_calibration(synthetic_df, ticker="TEST")
    ax_reliability = fig.get_axes()[0]

    # El reliability diagram dibuja los 3 niveles nominales como puntos/línea
    lines = ax_reliability.get_lines()
    # Buscar la línea de calibración observada (excluyendo la diagonal de referencia)
    # La línea de datos observados tendrá exactamente 3 puntos
    data_lines = [line for line in lines if len(line.get_xdata()) == 3]
    assert len(data_lines) >= 1, (
        f"El reliability diagram debe tener una línea con 3 puntos (p10/p50/p90). "
        f"Líneas encontradas: {[len(ln.get_xdata()) for ln in lines]}"
    )


def test_plot_calibration_pit_histogram(synthetic_df: pd.DataFrame) -> None:
    """El PIT histogram debe tener exactamente 10 barras (bins=10)."""
    from utils.visualization import plot_calibration

    fig = plot_calibration(synthetic_df, ticker="TEST")
    ax_pit = fig.get_axes()[1]

    patches = ax_pit.patches
    assert len(patches) == 10, (
        f"El PIT histogram debe tener exactamente 10 barras (bins=10), encontró {len(patches)}"
    )


def test_plot_calibration_pit_tails() -> None:
    """PIT debe asignar 0.0 a GT << p10 y 1.0 a GT >> p90, no clampar a 0.1/0.9.

    Verifica que el histograma tiene masa en el primer bin [0, 0.1) y en el
    último bin (0.9, 1] cuando todas las observaciones caen fuera del intervalo.
    """
    from utils.visualization import plot_calibration

    # Construir DataFrame donde GT siempre cae fuera del rango [p10, p90]
    rng = np.random.default_rng(seed=0)
    n = 50
    center = rng.normal(0, 0.005, n)  # cuantiles apretados cerca de 0

    # Mitad de GT muy por debajo de p10, mitad muy por encima de p90
    gt_bajo = center - 0.1  # GT << p10 → PIT debe ser 0.0
    gt_alto = center + 0.1  # GT >> p90 → PIT debe ser 1.0
    gt_all = np.concatenate([gt_bajo, gt_alto])

    rows = []
    for i in range(len(gt_all)):
        q = np.sort(center[i % n] + rng.normal(0, 0.001, 3))
        rows.append(
            {
                "window": i,
                "step": 0,
                "mean_Close": float(q[1]),
                "gt_Close": float(gt_all[i]),
                "p10_Close": float(q[0]),
                "p50_Close": float(q[1]),
                "p90_Close": float(q[2]),
            }
        )
    df_tails = pd.DataFrame(rows)

    fig = plot_calibration(df_tails, ticker="TAILS")
    ax_pit = fig.get_axes()[1]

    # Extraer alturas de los bins
    heights = np.array([p.get_height() for p in ax_pit.patches])

    # Con left=0.0/right=1.0: masa en primer bin (0–0.1) y último bin (0.9–1.0)
    assert heights[0] > 0, (
        "PIT bin [0, 0.1) debe tener masa cuando GT < p10 — "
        "si es 0, np.interp está clampando a 0.1 en vez de retornar 0.0"
    )
    assert heights[-1] > 0, (
        "PIT bin (0.9, 1.0] debe tener masa cuando GT > p90 — "
        "si es 0, np.interp está clampando a 0.9 en vez de retornar 1.0"
    )


# ---------------------------------------------------------------------------
# Tests: guardado a PNG
# ---------------------------------------------------------------------------


def test_save_fan_chart_png(synthetic_df: pd.DataFrame) -> None:
    """plot_fan_chart debe poder guardarse como .png en un directorio temporal."""
    from utils.visualization import plot_fan_chart

    fig = plot_fan_chart(synthetic_df, ticker="SAVETEST")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "fan_chart_SAVETEST.png"
        fig.savefig(str(output_path))
        assert output_path.exists(), f"El archivo PNG no fue creado en {output_path}"
        assert output_path.stat().st_size > 0, "El archivo PNG está vacío"


def test_save_calibration_png(synthetic_df: pd.DataFrame) -> None:
    """plot_calibration debe poder guardarse como .png en un directorio temporal."""
    from utils.visualization import plot_calibration

    fig = plot_calibration(synthetic_df, ticker="SAVETEST")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration_SAVETEST.png"
        fig.savefig(str(output_path))
        assert output_path.exists(), f"El archivo PNG no fue creado en {output_path}"
        assert output_path.stat().st_size > 0, "El archivo PNG está vacío"
