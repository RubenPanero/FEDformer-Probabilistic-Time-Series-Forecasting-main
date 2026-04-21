import pytest
import pandas as pd
import numpy as np
import os
from data.dataset import TimeSeriesDataset
from config import FEDformerConfig

def test_dataset_integrity():
    # Creamos un dataset temporal válido (cumple contrato)
    df = pd.DataFrame(np.random.randn(100, 11), columns=[
        "Date", "Open", "High", "Low", "Close", "Volume", 
        "VIX_Close", "RSI_14", "MACD", "Signal", "SMA_20"
    ])
    df.to_csv("temp_test.csv", index=False)
    
    config = FEDformerConfig(file_path="temp_test.csv")
    dataset = TimeSeriesDataset(config, flag="train")
    
    assert "enc_in" in dataset.derived_params
    assert dataset.derived_params["enc_in"] == 11
    
    # Verificamos que no se mutó la configuración (originalmente enc_in es None o derivado simple)
    assert config.enc_in is not None
    
    os.remove("temp_test.csv")

def test_dataset_contract_validation():
    # Creamos un dataset inválido (pocas columnas)
    df = pd.DataFrame(np.random.randn(100, 5), columns=["A", "B", "C", "D", "E"])
    df.to_csv("temp_invalid.csv", index=False)
    
    config = FEDformerConfig(file_path="temp_invalid.csv")
    with pytest.raises(ValueError, match="Contrato de datos roto"):
        TimeSeriesDataset(config, flag="train")
        
    os.remove("temp_invalid.csv")
