import os
import logging
from config import FEDformerConfig
from data.dataset import TimeSeriesDataset
from training.trainer import WalkForwardTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_base_training():
    data_path = "data/GOOGL_features.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No se encontró el dataset base en {data_path}. Ejecuta el preprocesamiento primero."
        )

    # Configuramos para predecir 'Close' usando el dataset financiero
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=data_path,
        seq_len=96,
        label_len=48,
        pred_len=24,
        date_column="date",
        n_epochs_per_fold=5,
        batch_size=32,
        learning_rate=0.0001,
        n_regimes=3,  # Detectar automáticamente 3 regímenes de volatilidad
        scaling_strategy="robust",  # Mejor para datos financieros con outliers
    )

    logger.info("Iniciando carga del dataset...")
    # Cargar el dataset en modo 'all' para que el WalkForwardTrainer haga los cortes temporales
    dataset = TimeSeriesDataset(config, flag="all")

    logger.info("Instanciando el WalkForwardTrainer...")
    trainer = WalkForwardTrainer(config, full_dataset=dataset)

    logger.info("Ejecutando Backtest (Entrenamiento Walk-Forward)...")
    # Utilizamos n_splits=5 para simular un ambiente realista en finanzas
    try:
        forecast = trainer.run_backtest(n_splits=5)
        logger.info(
            f"Entrenamiento base finalizado. Predicciones forma: {forecast.preds_scaled.shape}"
        )
    except Exception as e:
        logger.error(f"Fallo durante el entrenamiento base: {e}")


if __name__ == "__main__":
    run_base_training()
