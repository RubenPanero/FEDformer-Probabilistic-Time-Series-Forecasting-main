# -*- coding: utf-8 -*-
"""
Utilidades para entrenamiento del modelo FEDformer.
Refactorizado a Python 3.10+ para garantizar typing nativo purificado y eficiencia PEP 8.
"""

import logging

import torch
from torch import nn

from utils import get_device

logger = logging.getLogger(__name__)
device = get_device()


def mc_dropout_inference(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    n_samples: int = 100,
    use_flow_sampling: bool = True,
    mc_batch_size: int = 1,
) -> torch.Tensor:
    """Ejecuta muestreo MC Dropout y acumula las trayectorias resultantes.

    Si ``use_flow_sampling`` es True y el modelo expone una distribución capaz
    de muestrear (``.sample()``), extraemos simulaciones finitas del ruido.
    De otro modo, el sistema recae predictivamente en la media esperada.

    ``mc_batch_size`` solo trocea el bucle Python que acumula ``n_samples``.
    La implementación actual sigue ejecutando un forward del modelo por muestra,
    por lo que no realiza vectorización real ni batching adicional en GPU.
    """

    def enable_dropout(m: nn.Module) -> None:
        if isinstance(m, nn.Dropout):
            m.train()

    prev_mode = model.training
    model.apply(enable_dropout)

    x_enc = batch["x_enc"].to(device, non_blocking=True)
    x_dec = batch["x_dec"].to(device, non_blocking=True)
    x_regime = batch["x_regime"].to(device, non_blocking=True)

    if mc_batch_size <= 0:
        raise ValueError(f"mc_batch_size must be positive, got {mc_batch_size}")

    samples: list[torch.Tensor] = []

    with torch.no_grad():
        remaining = n_samples
        while remaining > 0:
            current_batch = min(mc_batch_size, remaining)
            for _ in range(current_batch):
                try:
                    dist = model(x_enc, x_dec, x_regime)
                    if use_flow_sampling and hasattr(dist, "sample"):
                        s = dist.sample(1)  # [1, B, T, F] or [1, B, T]
                        samples.append(s[0])
                    else:
                        samples.append(dist.mean)
                except (RuntimeError, ValueError) as exc:
                    logger.warning(
                        "Fallo en el muestreo de distribución MC Dropout: %s", exc
                    )
                    if samples:
                        samples.append(torch.zeros_like(samples[0]))
                    else:
                        # Acceso seguro al shape nativo asumiendo inicializado el config
                        pred_len = getattr(model, "config", None)
                        if pred_len is not None:
                            dummy_shape = (
                                int(x_enc.size(0)),
                                int(model.config.pred_len),  # type: ignore
                                int(model.config.c_out),  # type: ignore
                            )
                        else:
                            dummy_shape = (
                                int(x_enc.size(0)),
                                1,
                                1,
                            )  # Caída ciega defensiva
                        samples.append(torch.zeros(*dummy_shape, device=device))
            remaining -= current_batch

    if not samples:
        logger.error("Se abortaron todos los muestreos de MonteCarlo.")
        pred_len = getattr(model, "config", None)
        if pred_len is not None:
            dummy_shape = (
                int(x_enc.size(0)),
                int(model.config.pred_len),  # type: ignore
                int(model.config.c_out),  # type: ignore
            )
        else:
            dummy_shape = (int(x_enc.size(0)), 1, 1)
        out = torch.zeros(1, *dummy_shape, device=device)
    else:
        out = torch.stack(samples)

    # Restauración inmutable del modo nativo del modelo
    if not prev_mode:
        model.eval()

    return out
