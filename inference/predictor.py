# -*- coding: utf-8 -*-
"""Predicción probabilística sobre datos nuevos usando modelos canónicos."""

import logging

import numpy as np
import torch

from config import FEDformerConfig
from data.dataset import TimeSeriesDataset
from data.preprocessing import PreprocessingPipeline
from models.fedformer import Flow_FEDformer
from training.forecast_output import ForecastOutput
from training.trainer import DEFAULT_QUANTILE_LEVELS
from training.utils import mc_dropout_inference

logger = logging.getLogger(__name__)


def predict(
    model: Flow_FEDformer,
    config: FEDformerConfig,
    preprocessor: PreprocessingPipeline,
    csv_path: str,
    n_samples: int = 50,
) -> ForecastOutput:
    """Genera predicciones probabilisticas sobre datos nuevos.

    Crea un dataset con el preprocessor pre-ajustado (sin re-fit),
    evalúa todas las ventanas disponibles con MC Dropout, y retorna
    un ForecastOutput con predicciones en espacio real.

    Args:
        model: Modelo Flow_FEDformer con pesos cargados.
        config: Configuración del modelo.
        preprocessor: PreprocessingPipeline ya ajustado (fitted).
        csv_path: Ruta al CSV con datos para predecir.
        n_samples: Número de muestras MC Dropout por ventana.

    Returns:
        ForecastOutput con predicciones, cuantiles y muestras.

    Raises:
        RuntimeError: Si el pipeline de inferencia no puede construir el
            dataset a partir del preprocesador restaurado.
    """
    # Crear dataset reutilizando el preprocessor pre-ajustado (no re-fit).
    # fit_scope="fold_train_only" activa refit incondicionalmente en _fit_and_transform,
    # incluso si el preprocessor ya está fitted. Se sobreescribe temporalmente para
    # garantizar que los artefactos de entrenamiento no sean contaminados con datos nuevos.
    inference_config = _make_inference_config(config, csv_path)
    original_fit_scope = preprocessor.fit_scope
    preprocessor.fit_scope = "inference"
    try:
        dataset = TimeSeriesDataset(
            config=inference_config,
            flag="all",
            preprocessor=preprocessor,
        )
    finally:
        preprocessor.fit_scope = original_fit_scope

    if len(dataset) == 0:
        logger.warning("CSV no contiene ventanas suficientes para predicción.")
        return _empty_forecast(config)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False
    )

    all_preds, all_gt, all_samples, all_quantiles = _evaluate(model, loader, n_samples)

    if not all_preds:
        return _empty_forecast(config)

    preds_scaled = np.concatenate(all_preds, axis=0)
    gt_scaled = np.concatenate(all_gt, axis=0)
    samples_scaled = np.concatenate(all_samples, axis=1)
    quantiles_scaled = np.concatenate(all_quantiles, axis=1)

    # Invertir escala a espacio real — por slices para manejar dimensiones extra
    targets = list(config.target_features)
    preds_real = preprocessor.inverse_transform_targets(preds_scaled, targets)
    gt_real = preprocessor.inverse_transform_targets(gt_scaled, targets)

    n_q = quantiles_scaled.shape[0]
    quantiles_real = np.stack(
        [
            preprocessor.inverse_transform_targets(quantiles_scaled[i], targets)
            for i in range(n_q)
        ]
    )

    n_s = samples_scaled.shape[0]
    samples_real = np.stack(
        [
            preprocessor.inverse_transform_targets(samples_scaled[i], targets)
            for i in range(n_s)
        ]
    )

    return ForecastOutput(
        preds_scaled=preds_scaled,
        gt_scaled=gt_scaled,
        samples_scaled=samples_scaled,
        preds_real=preds_real,
        gt_real=gt_real,
        samples_real=samples_real,
        quantiles_scaled=quantiles_scaled,
        quantiles_real=quantiles_real,
        quantile_levels=DEFAULT_QUANTILE_LEVELS.copy(),
        metric_space=config.metric_space,
        return_transform=config.return_transform,
        target_names=targets,
    )


def _make_inference_config(config: FEDformerConfig, csv_path: str) -> FEDformerConfig:
    """Crea config apuntando al CSV de inferencia.

    Preserva label_len del modelo entrenado — un valor distinto al default
    causaría tensores x_dec incompatibles y fallos silenciosos en mc_dropout.
    """
    return FEDformerConfig(
        target_features=list(config.target_features),
        file_path=csv_path,
        seq_len=config.seq_len,
        label_len=config.label_len,
        pred_len=config.pred_len,
        batch_size=config.batch_size,
        return_transform=config.return_transform,
        metric_space=config.metric_space,
    )


def _evaluate(
    model: Flow_FEDformer,
    loader: torch.utils.data.DataLoader,
    n_samples: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Evaluación probabilística con MC Dropout sobre un DataLoader.

    NO envuelve en torch.inference_mode() — mc_dropout_inference ya usa
    torch.no_grad() internamente y necesita modo train en capas dropout.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    all_samples: list[np.ndarray] = []
    all_quantiles: list[np.ndarray] = []

    for batch in loader:
        try:
            # samples: (n_samples, batch_size, pred_len, n_features)
            samples = mc_dropout_inference(
                model, batch, n_samples=n_samples, use_flow_sampling=True
            )
            samples_cpu = samples.detach().to("cpu", dtype=torch.float32)
            quantiles_cpu = torch.quantile(
                samples_cpu,
                q=torch.tensor(DEFAULT_QUANTILE_LEVELS.tolist(), dtype=torch.float32),
                dim=0,
            )
            all_samples.append(samples_cpu.numpy())
            all_quantiles.append(quantiles_cpu.numpy())
            all_preds.append(quantiles_cpu[1].numpy())  # p50
            all_gt.append(batch["y_true"].cpu().numpy())
        except (RuntimeError, ValueError) as exc:
            logger.warning("Error en evaluación de batch: %s", exc)
            continue

    return all_preds, all_gt, all_samples, all_quantiles


def _empty_forecast(config: FEDformerConfig) -> ForecastOutput:
    """Retorna un ForecastOutput vacío con shapes coherentes."""
    n_targets = len(config.target_features)
    empty_2d = np.empty((0, config.pred_len, n_targets), dtype=np.float32)
    empty_q = np.empty(
        (len(DEFAULT_QUANTILE_LEVELS), 0, config.pred_len, n_targets),
        dtype=np.float32,
    )
    empty_s = np.empty((0, 0, config.pred_len, n_targets), dtype=np.float32)
    return ForecastOutput(
        preds_scaled=empty_2d,
        gt_scaled=empty_2d.copy(),
        samples_scaled=empty_s,
        preds_real=empty_2d.copy(),
        gt_real=empty_2d.copy(),
        samples_real=empty_s.copy(),
        quantiles_scaled=empty_q,
        quantiles_real=empty_q.copy(),
        quantile_levels=DEFAULT_QUANTILE_LEVELS.copy(),
        metric_space=config.metric_space,
        return_transform=config.return_transform,
        target_names=list(config.target_features),
    )
