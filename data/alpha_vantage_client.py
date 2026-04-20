import os
import time
import pandas as pd
import requests
import logging
from requests import RequestException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaVantageClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
        self.base_url = "https://www.alphavantage.co/query"

    def get_daily_data(self, symbol, outputsize="full"):
        """
        Descarga datos OHLCV diarios desde Alpha Vantage.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "datatype": "json",
        }
        logger.info(f"Descargando datos para {symbol} desde Alpha Vantage...")

        # Considerar retry logic para APIs gratuitas
        for attempt in range(3):
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
            except RequestException as exc:
                logger.warning(
                    "Error de red en intento %s/3 para %s: %s",
                    attempt + 1,
                    symbol,
                    exc,
                )
                if attempt < 2:
                    time.sleep(10)
                    continue
                raise

            if response.status_code != 200:
                logger.error(f"Error de red: {response.status_code}")
                time.sleep(10)
                continue

            data = response.json()

            if "Time Series (Daily)" in data:
                df: pd.DataFrame = pd.DataFrame.from_dict(
                    data["Time Series (Daily)"], orient="index"
                )
                df.index = pd.to_datetime(df.index)
                df = df.rename(
                    columns={
                        "1. open": "Open",
                        "2. high": "High",
                        "3. low": "Low",
                        "4. close": "Close",
                        "5. volume": "Volume",
                    }
                )
                # Volver numéricos
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])  # pylint: disable=unsupported-assignment-operation,unsubscriptable-object

                df.sort_index(inplace=True)
                return df

            elif "Information" in data and "rate limit" in data["Information"].lower():
                logger.warning(
                    f"Límite de API alcanzado. Reintentando en 60s (Intento {attempt + 1}/3)"
                )
                time.sleep(60)
            else:
                logger.error(
                    f"Error inesperado o símbolo no encontrado en Alpha Vantage para {symbol}: {data}"
                )
                break

        raise Exception(
            f"Fallo al descargar datos para {symbol} desde Alpha Vantage. Revisa tu API key o los limites diarios."
        )


if __name__ == "__main__":
    client = AlphaVantageClient()
    try:
        df = client.get_daily_data("GOOGL", outputsize="compact")
        print(df.head())
    except Exception as e:
        print(e)
