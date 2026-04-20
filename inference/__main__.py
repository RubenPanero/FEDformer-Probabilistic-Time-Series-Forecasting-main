"""CLI de inferencia para modelos canónicos FEDformer.

Uso:
    python3 -m inference --ticker NVDA --csv data/NVDA_features.csv
    python3 -m inference --ticker NVDA --csv data/NVDA_features.csv --output predictions.csv
    python3 -m inference --list-models
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from inference.loader import available_tickers, load_specialist
from inference.predictor import predict
from utils.model_registry import DEFAULT_REGISTRY_PATH

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia probabilística con modelos canónicos FEDformer",
        prog="python3 -m inference",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Símbolo del ticker (ej. NVDA, GOOGL)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Ruta al CSV con datos para predecir",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta del CSV de salida (default: results/inference_{ticker}.csv)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Número de muestras MC Dropout (default: 50)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Ruta al model_registry.json (default: checkpoints/model_registry.json)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Lista modelos canónicos disponibles y sale",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Genera fan chart y calibration plot tras inferencia",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directorio de salida para plots (default: results/)",
    )
    return parser.parse_args()


def _pad_csv_for_forecast(csv_path: str, seq_len: int, pred_len: int) -> str | None:
    """Padea el CSV si tiene seq_len <= filas < seq_len+pred_len (inferencia online).

    Duplica la última fila pred_len veces para que TimeSeriesDataset pueda
    crear al menos una ventana. La predicción sobre esas filas paddeadas no
    tiene sentido como ground truth, pero el modelo genera predicciones futuras
    válidas sobre el contexto real.

    Retorna ruta a un fichero temporal, o None si no hace falta padding.
    El caller es responsable de borrar el fichero con os.unlink().
    """
    df = pd.read_csv(csv_path)
    n = len(df)
    needed = seq_len + pred_len

    if n >= needed or n < seq_len:
        return None

    n_pad = needed - n
    padding = pd.concat([df.tail(1)] * n_pad, ignore_index=True)
    padded = pd.concat([df, padding], ignore_index=True)

    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    padded.to_csv(tmp_path, index=False)
    return tmp_path


def _export_predictions(forecast, output_path: Path) -> None:
    """Exporta predicciones a CSV con cuantiles para todos los targets."""
    n_windows = forecast.preds_real.shape[0]
    pred_len = forecast.preds_real.shape[1]
    n_targets = forecast.preds_real.shape[2]
    target_names = (
        list(forecast.target_names)
        if forecast.target_names
        else [f"target_{i}" for i in range(n_targets)]
    )

    # Media muestral real — estadísticamente correcta aunque la distribución sea asimétrica
    has_samples = forecast.samples_real is not None and forecast.samples_real.size > 0
    mean_real = forecast.samples_real.mean(axis=0) if has_samples else None

    rows = []
    for w in range(n_windows):
        for t in range(pred_len):
            row: dict = {"window": w, "step": t}
            for i, name in enumerate(target_names):
                if mean_real is not None:
                    row[f"mean_{name}"] = float(mean_real[w, t, i])
                row[f"gt_{name}"] = float(forecast.gt_real[w, t, i])
                if forecast.quantiles_real is not None:
                    row[f"p10_{name}"] = float(forecast.p10_real[w, t, i])
                    row[f"p50_{name}"] = float(forecast.p50_real[w, t, i])
                    row[f"p90_{name}"] = float(forecast.p90_real[w, t, i])
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Predicciones exportadas a %s (%d filas)", output_path, len(df))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = _parse_args()
    registry_path = Path(args.registry) if args.registry else DEFAULT_REGISTRY_PATH

    if args.list_models:
        tickers = available_tickers(registry_path)
        if tickers:
            print("Modelos canónicos disponibles:")
            for t in tickers:
                print(f"  - {t}")
        else:
            print("No hay modelos canónicos registrados.")
        return 0

    if not args.ticker or not args.csv:
        print("Error: --ticker y --csv son requeridos.", file=sys.stderr)
        print("Uso: python3 -m inference --ticker NVDA --csv data/NVDA_features.csv")
        return 1

    ticker = args.ticker.upper()
    csv_path = args.csv

    if not Path(csv_path).exists():
        print(f"Error: CSV no encontrado: {csv_path}", file=sys.stderr)
        return 1

    try:
        model, config, preprocessor = load_specialist(ticker, registry_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error cargando modelo: {exc}", file=sys.stderr)
        return 1

    # Padding automático para predicción online (CSV solo con contexto histórico)
    padded_path = _pad_csv_for_forecast(csv_path, config.seq_len, config.pred_len)
    if padded_path:
        logger.info(
            "CSV corto detectado — padding automático para predicción online "
            "(seq_len=%d, pred_len=%d)",
            config.seq_len,
            config.pred_len,
        )
    effective_csv = padded_path if padded_path else csv_path

    try:
        forecast = predict(
            model=model,
            config=config,
            preprocessor=preprocessor,
            csv_path=effective_csv,
            n_samples=args.n_samples,
        )
    finally:
        if padded_path:
            os.unlink(padded_path)

    if forecast.preds_real.size == 0:
        print("Error: no se generaron predicciones.", file=sys.stderr)
        return 1

    output_path = Path(args.output or f"results/inference_{ticker.lower()}.csv")
    _export_predictions(forecast, output_path)

    # Visualización probabilística (solo si --plot activado)
    fan_path = None
    cal_path = None
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")  # Headless — antes de importar pyplot
        import matplotlib.pyplot as plt

        from utils.visualization import plot_calibration, plot_fan_chart

        df_viz = pd.read_csv(output_path)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig_fan = plot_fan_chart(df_viz, ticker=ticker)
        fan_path = output_dir / f"fan_chart_{ticker.lower()}.png"
        fig_fan.savefig(str(fan_path), dpi=150)
        plt.close(fig_fan)
        logger.info("Fan chart guardado en %s", fan_path)

        fig_cal = plot_calibration(df_viz, ticker=ticker)
        cal_path = output_dir / f"calibration_{ticker.lower()}.png"
        fig_cal.savefig(str(cal_path), dpi=150)
        plt.close(fig_cal)
        logger.info("Calibration plot guardado en %s", cal_path)

    # Resumen en stdout
    print(f"\n{'=' * 50}")
    print(f"Inferencia {ticker} completada")
    print(f"{'=' * 50}")
    print(f"  Ventanas evaluadas: {forecast.preds_real.shape[0]}")
    print(f"  Horizonte (pred_len): {forecast.preds_real.shape[1]}")
    print(f"  Muestras MC: {args.n_samples}")
    print(f"  Output: {output_path}")
    if forecast.quantiles_real is not None:
        p10_mean = float(np.mean(forecast.p10_real))
        p50_mean = float(np.mean(forecast.p50_real))
        p90_mean = float(np.mean(forecast.p90_real))
        print(
            f"  Media cuantiles — p10: {p10_mean:.4f}  p50: {p50_mean:.4f}  p90: {p90_mean:.4f}"
        )
    if args.plot:
        print(f"  Fan chart: {fan_path}")
        print(f"  Calibration: {cal_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
