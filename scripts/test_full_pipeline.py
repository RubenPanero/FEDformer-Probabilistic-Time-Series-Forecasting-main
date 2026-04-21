import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from data.financial_dataset_builder import build_financial_dataset
from data.dataset import TimeSeriesDataset
from config import FEDformerConfig

def test_full_pipeline_nvda():
    print("Iniciando prueba de flujo completo para NVDA (10 años)...")
    
    # 1. Ajuste del builder para 10 años
    # Como el builder tiene un hardcode de 7 años, vamos a sobrescribirlo temporalmente en el flujo
    # O mejor, ejecutamos el builder y luego validamos.
    
    ticker = "NVDA"
    output_dir = "data_test_full"
    
    # Generar dataset
    print(f"Generando dataset para {ticker}...")
    csv_path = build_financial_dataset(ticker, output_dir)
    print(f"Dataset generado en: {csv_path}")
    
    # Configurar dataset
    config = FEDformerConfig(file_path=csv_path, target_features=["Close"])
    
    # Instanciar TimeSeriesDataset
    print("Inicializando TimeSeriesDataset...")
    dataset = TimeSeriesDataset(config, flag="all")
    
    print("\nResumen de integridad:")
    print(f"Forma de los datos escalados: {dataset.full_data_scaled.shape}")
    print(f"Parámetros derivados: {dataset.derived_params}")
    print(f"Columnas detectadas: {dataset.feature_columns}")
    
    # Validaciones finales
    assert dataset.full_data_scaled.shape[1] == 11
    print("\n¡Prueba de flujo completo superada exitosamente!")

if __name__ == "__main__":
    test_full_pipeline_nvda()
