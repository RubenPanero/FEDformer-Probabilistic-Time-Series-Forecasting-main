import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class VixDataFetcher:
    def __init__(self):
        self.ticker = "^VIX"

    def get_vix_data(self, start_date=None, end_date=None):
        """
        Descarga datos del VIX usando yfinance.
        Retorna la columna Close del VIX.
        """
        logger.info("Descargando historial del VIX...")
        try:
            vix = yf.download(
                self.ticker, start=start_date, end=end_date, progress=False
            )
            if vix.empty:
                logger.warning(
                    "No se obtuvieron datos del VIX en las fechas solicitadas."
                )
                return pd.DataFrame()

            # yfinance a veces retorna MultiIndex si se piden varios, pero con uno solo es simple o MultiIndex (en versiones nuevas de yf)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.droplevel(1)

            vix_close = vix[["Close"]].rename(columns={"Close": "VIX_Close"})
            vix_close.index = pd.to_datetime(
                vix_close.index
            ).normalize()  # Ensure index is date only timezone naive

            # Si yfinance retorna un indice con tz (timezone-aware), se le quita
            if vix_close.index.tz is not None:
                vix_close.index = vix_close.index.tz_localize(None)

            return vix_close
        except Exception as e:
            logger.error(f"Error descargando el VIX: {e}")
            raise
