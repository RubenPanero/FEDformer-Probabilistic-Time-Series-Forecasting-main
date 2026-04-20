"""Builder de dataset financiero mínimo para FEDformer.

Features de salida (11):
    OHLCV (5) + VIX_Close (1) + RSI_14 (1) + ATRr_14 (1) + MACD×3 (3)
"""

import argparse
import os
import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pandas_ta as ta  # noqa: F401

from data.alpha_vantage_client import AlphaVantageClient
from data.vix_data import VixDataFetcher

logger = logging.getLogger(__name__)


def build_financial_dataset(
    symbol: str, output_dir: str, use_mock: bool = False
) -> str:
    """Construye y guarda el dataset financiero con 11 features para FEDformer.

    Args:
        symbol:     Ticker, e.g. "NVDA"
        output_dir: Directorio de salida para el CSV
        use_mock:   Si True, usa yfinance en lugar de Alpha Vantage

    Returns:
        Ruta absoluta al CSV generado.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Obtener OHLCV (7 años)
    df = _fetch_ohlcv(symbol, use_mock)

    # Filtrar a los últimos 7 años — calculado en tiempo de llamada para evitar drift por import
    seven_years_ago = (datetime.now() - timedelta(days=7 * 365)).strftime("%Y-%m-%d")
    cutoff = pd.Timestamp(seven_years_ago)
    df = df[df.index >= cutoff]

    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {symbol}.")

    # 2. VIX
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    vix_fetcher = VixDataFetcher()
    vix_df = vix_fetcher.get_vix_data(start_date=start_date, end_date=end_date)
    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df["VIX_Close"] = df["VIX_Close"].ffill()
    else:
        df["VIX_Close"] = 0.0

    # 3. Indicadores técnicos (mínimo seleccionado)
    logger.info("Calculando indicadores técnicos para %s...", symbol)
    df.ta.rsi(length=14, append=True)  # RSI_14
    df.ta.atr(length=14, append=True)  # ATRr_14
    df.ta.macd(fast=12, slow=26, signal=9, append=True)  # MACD_12_26_9/h/s

    # Limpiar NaN del warm-up de indicadores (~26 filas)
    df.dropna(inplace=True)

    # 4. Guardar
    output_path = os.path.join(output_dir, f"{symbol}_features.csv")
    df.index.name = "date"
    df.to_csv(output_path)
    logger.info("Dataset guardado: %s — forma %s", output_path, df.shape)

    return output_path


def validate_dataset(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """Comprueba la integridad del dataset antes del entrenamiento.

    Args:
        df:     DataFrame con índice DatetimeIndex y columnas OHLCV + features.
        symbol: Ticker del activo para mensajes de log.

    Returns:
        dict con: shape, date_range, max_date_gap_days, null_counts,
                  price_inconsistencies, y out_of_range por indicador.
    """
    report: dict[str, Any] = {}

    # Continuidad temporal
    date_gaps = df.index.to_series().diff().dt.days
    max_gap = int(date_gaps.max()) if not date_gaps.empty else 0
    report["max_date_gap_days"] = max_gap
    if max_gap > 7:
        logger.warning("GAP temporal detectado: %d días en %s", max_gap, symbol)

    # Valores nulos
    null_counts = df.isnull().sum()
    report["null_counts"] = null_counts[null_counts > 0].to_dict()
    if null_counts.sum() > 0:
        logger.warning("Nulos en %s:\n%s", symbol, null_counts[null_counts > 0])

    # Rangos de indicadores conocidos
    range_checks = {
        "RSI_14": (0.0, 100.0),
        "VIX_Close": (0.0, 200.0),
    }
    for col, (lo, hi) in range_checks.items():
        if col in df.columns:
            out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if out > 0:
                logger.warning("%s: %d valores fuera de [%s, %s]", col, out, lo, hi)
                report[f"{col}_out_of_range"] = out

    # Consistencia OHLCV
    price_inconsistent = int((df["High"] < df["Low"]).sum())
    report["price_inconsistencies"] = price_inconsistent
    if price_inconsistent > 0:
        logger.error("High < Low en %d filas de %s", price_inconsistent, symbol)

    # Resumen
    report["shape"] = df.shape
    report["date_range"] = (str(df.index.min()), str(df.index.max()))
    report["n_features"] = df.shape[1]

    logger.info(
        "validate_dataset %s: %d filas × %d cols | %s → %s",
        symbol,
        df.shape[0],
        df.shape[1],
        df.index.min().date(),
        df.index.max().date(),
    )
    return report


def _fetch_ohlcv(symbol: str, use_mock: bool) -> pd.DataFrame:
    """Descarga OHLCV desde yfinance (mock) o Alpha Vantage (real).

    Args:
        symbol:   Ticker a descargar.
        use_mock: Si True, usa yfinance.

    Returns:
        DataFrame con índice DatetimeIndex y columnas OHLCV.
    """
    if use_mock:
        import yfinance as yf

        logger.info("Usando yfinance (mock) para %s", symbol)
        df = yf.download(symbol, period="7y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.index.tz is not None:  # guard: yfinance moderno retorna índice tz-naive
            df.index = df.index.tz_localize(None)
    else:
        av_client = AlphaVantageClient()
        df = av_client.get_daily_data(symbol, outputsize="full")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Financial Dataset Builder — FEDformer"
    )
    parser.add_argument("--symbol", type=str, default="NVDA", help="Ticker a descargar")
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Directorio de salida"
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Usa yfinance en lugar de Alpha Vantage (no requiere API key)",
    )
    args = parser.parse_args()
    build_financial_dataset(args.symbol, args.output_dir, args.use_mock)
