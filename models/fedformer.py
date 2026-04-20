# -*- coding: utf-8 -*-
"""
Modelo principal FEDformer con Normalizing Flows.
Refactorizado para cumplir con PEP 8, tipeo estricto y Python 3.10+.
"""

import torch
import torch.distributions
from torch import nn

from config import FEDformerConfig
from .encoder_decoder import DecoderLayer, EncoderLayer, LayerConfig
from .flows import NormalizingFlow
from .layers import OptimizedSeriesDecomp


class Flow_FEDformer(nn.Module):
    """Modelo FEDformer probabilistico condicionado por regimes y flows.

    Combina encoder/decoder FEDformer con un bloque de normalizing flows para
    producir una distribucion predictiva en lugar de una salida puntual.
    """

    # pylint: disable=invalid-name

    def __init__(self, config: FEDformerConfig) -> None:
        super().__init__()
        self.config = config

        self.components = nn.ModuleDict(
            {
                "decomp": OptimizedSeriesDecomp(config.moving_avg),
                "regime_embedding": nn.Embedding(
                    config.n_regimes, config.regime_embedding_dim
                ),
                "trend_proj": nn.Linear(config.dec_in, config.d_model),
                "enc_embedding": nn.Linear(
                    config.enc_in + config.regime_embedding_dim, config.d_model
                ),
                "dec_embedding": nn.Linear(
                    config.dec_in + config.regime_embedding_dim, config.d_model
                ),
                "dropout": nn.Dropout(config.dropout),
                "flow_conditioner_proj": nn.Linear(
                    config.d_model, config.c_out * config.flow_hidden_dim
                ),
            }
        )

        encoder_config = LayerConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            seq_len=config.seq_len,
            d_ff=config.d_ff,
            modes=config.modes,
            dropout=config.dropout,
            activation=config.activation,
            moving_avg=config.moving_avg,
        )
        decoder_config = LayerConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            seq_len=config.label_len + config.pred_len,
            d_ff=config.d_ff,
            modes=config.modes,
            dropout=config.dropout,
            activation=config.activation,
            moving_avg=config.moving_avg,
        )

        self.sequence_layers = nn.ModuleDict(
            {
                "encoders": nn.ModuleList(
                    [EncoderLayer(encoder_config) for _ in range(config.e_layers)]
                ),
                "decoders": nn.ModuleList(
                    [DecoderLayer(decoder_config) for _ in range(config.d_layers)]
                ),
            }
        )

        self.flows = nn.ModuleList(
            [
                NormalizingFlow(
                    n_layers=config.n_flow_layers,
                    d_model=config.pred_len,
                    hidden_dim=config.flow_hidden_dim,
                    context_dim=config.flow_hidden_dim,
                )
                for _ in range(config.c_out)
            ]
        )

    def _prepare_decoder_input(
        self, x_dec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare decoder seasonal and trend initial components."""
        trend_proj = self.components["trend_proj"]
        decomp = self.components["decomp"]

        mean = torch.mean(x_dec[:, : self.config.label_len, :], dim=1, keepdim=True)
        seasonal_init = torch.zeros_like(x_dec[:, -self.config.pred_len :, :])

        trend_init_in = mean.expand(-1, self.config.pred_len, -1)
        # Using type: ignore for dynamic nn.ModuleDict resolution if preferred,
        # but PyTorch normally accepts this smoothly.
        trend_init = trend_proj(trend_init_in)

        seasonal_dec_hist, trend_dec_hist = decomp(x_dec[:, : self.config.label_len, :])
        seasonal_out = torch.cat([seasonal_dec_hist, seasonal_init], dim=1)

        trend_dec_hist_proj = trend_proj(trend_dec_hist)
        trend_out = torch.cat([trend_dec_hist_proj, trend_init], dim=1)
        return seasonal_out, trend_out

    def _attach_regime_vectors(
        self,
        x_enc: torch.Tensor,
        seasonal_init: torch.Tensor,
        x_regime: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate regime embeddings with encoder/decoder tensors."""
        batch_size = x_enc.size(0)
        regime_idx = x_regime.long().reshape(-1)

        if regime_idx.numel() == 1 and batch_size > 1:
            regime_idx = regime_idx.expand(batch_size)

        if regime_idx.numel() != batch_size:
            raise RuntimeError(
                f"Secuencia de tamaño incorrecto. Esperado {batch_size}, recibido {regime_idx.numel()}"
            )

        regime_embedding_layer = self.components["regime_embedding"]
        regime_vec = regime_embedding_layer(regime_idx)

        regime_vec_enc = regime_vec.unsqueeze(1).expand(
            batch_size, self.config.seq_len, regime_vec.size(-1)
        )
        regime_vec_dec = regime_vec.unsqueeze(1).expand(
            batch_size,
            self.config.label_len + self.config.pred_len,
            regime_vec.size(-1),
        )
        return (
            torch.cat([x_enc, regime_vec_enc], dim=-1),
            torch.cat([seasonal_init, regime_vec_dec], dim=-1),
        )

    def _embed_with_dropout(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Apply the requested embedding followed by shared dropout."""
        module = self.components[key]
        dropout = self.components["dropout"]
        return dropout(module(tensor))

    def _run_sequence_layers(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        trend_init: torch.Tensor,
        use_checkpointing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Executes the encoding and decoding sequences with optional checkpointing."""
        trend_part = torch.zeros_like(trend_init)

        if use_checkpointing and self.training:
            for encoder_layer in self.sequence_layers["encoders"]:
                enc_out = torch.utils.checkpoint.checkpoint(
                    encoder_layer, enc_out, use_reentrant=False
                )
            for decoder_layer in self.sequence_layers["decoders"]:
                dec_out, trend_delta = torch.utils.checkpoint.checkpoint(
                    decoder_layer, dec_out, enc_out, use_reentrant=False
                )
                trend_part = trend_part + trend_delta
        else:
            for encoder_layer in self.sequence_layers["encoders"]:
                enc_out = encoder_layer(enc_out)
            for decoder_layer in self.sequence_layers["decoders"]:
                dec_out, trend_delta = decoder_layer(dec_out, enc_out)
                trend_part = trend_part + trend_delta

        return dec_out, trend_part

    def forward(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, x_regime: torch.Tensor
    ) -> "NormalizingFlowDistribution":
        """Ejecuta el forward y devuelve una distribucion condicionada.

        Args:
            x_enc: Tensor del encoder con shape `(batch, seq_len, enc_in)`.
            x_dec: Tensor del decoder con shape
                `(batch, label_len + pred_len, dec_in)`.
            x_regime: Tensor de indices de regimen por batch.

        Returns:
            `NormalizingFlowDistribution` con media y flows condicionados para
            calcular `log_prob()` o muestrear trayectorias futuras.

        Raises:
            RuntimeError: Si hay incompatibilidades de shape o falla la
                propagacion interna del modelo.
        """
        seasonal_init, trend_init = self._prepare_decoder_input(x_dec)

        x_enc_with_regime, seasonal_init_with_regime = self._attach_regime_vectors(
            x_enc, seasonal_init, x_regime
        )

        enc_out = self._embed_with_dropout(x_enc_with_regime, "enc_embedding")
        dec_out = self._embed_with_dropout(seasonal_init_with_regime, "dec_embedding")

        try:
            dec_out, trend_part = self._run_sequence_layers(
                enc_out, dec_out, trend_init, self.config.use_gradient_checkpointing
            )
        except Exception as e:
            raise RuntimeError(
                f"Error durante la propagación (forward pass): {e}"
            ) from e

        if trend_init.shape[-1] != trend_part.shape[-1]:
            raise RuntimeError(
                f"Discrepancia vectorial: trend_init={trend_init.shape[-1]} "
                f"vs trend_part={trend_part.shape[-1]}."
            )

        final_trend = trend_init + trend_part

        dec_ctx = dec_out[:, -self.config.pred_len :, :]
        conditioner = self.components["flow_conditioner_proj"]
        flow_conditioned = conditioner(dec_ctx)

        batch_size, time_steps = flow_conditioned.shape[:2]
        flow_conditioned = flow_conditioned.view(
            batch_size,
            time_steps,
            self.config.c_out,
            self.config.flow_hidden_dim,
        )

        feature_context = flow_conditioned.mean(dim=1)
        mean_pred = final_trend[:, -self.config.pred_len :, : self.config.c_out]

        return NormalizingFlowDistribution(
            mean_pred,
            self.flows,
            feature_context,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            training=self.training,
        )


class NormalizingFlowDistribution:
    """Distribucion predictiva respaldada por normalizing flows por feature.

    Expone una interfaz compatible con entrenamiento probabilistico mediante
    `mean`, `log_prob()` y `sample()`, manteniendo un flow independiente por
    variable objetivo y su contexto asociado.
    """

    def __init__(
        self,
        means: torch.Tensor,
        flows: nn.ModuleList,
        contexts: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        training: bool = False,
    ) -> None:
        """Store flow distribution parameters."""
        self.means = means
        self.flows = flows
        self.contexts = contexts
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training = training

    @property
    def mean(self) -> torch.Tensor:
        """Return the predictive mean."""
        return self.means

    def log_prob(
        self, y_true: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Calcula log-prob medio por batch.

        Args:
            y_true: Targets reales con shape `(batch, pred_len, c_out)`.
            mask: Mascara opcional sobre la dimension temporal.

        Returns:
            Tensor con log-probabilidad media por elemento del batch.
        """
        batch_size, time_steps, num_features = y_true.shape
        total_lp = torch.zeros(batch_size, device=y_true.device, dtype=y_true.dtype)

        for feature_idx in range(num_features):
            y_feature = y_true[..., feature_idx]
            mu_feature = self.means[..., feature_idx]
            ctx_feature = self.contexts[:, feature_idx, :]

            flow = self.flows[feature_idx]
            if self.use_gradient_checkpointing and self.training:
                lp_feature = torch.utils.checkpoint.checkpoint(
                    lambda y, mu, ctx: flow.log_prob(  # noqa: B023
                        y, base_mean=mu, context=ctx
                    ),
                    y_feature,
                    mu_feature,
                    ctx_feature,
                    use_reentrant=False,
                )
            else:
                lp_feature = flow.log_prob(
                    y_feature, base_mean=mu_feature, context=ctx_feature
                )
            total_lp = total_lp + lp_feature

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            valid_counts = mask.sum(dim=1).clamp(min=1).to(total_lp.dtype)
            return total_lp / valid_counts

        return total_lp / float(time_steps)

    def sample(self, n_samples: int) -> torch.Tensor:
        """Genera trayectorias muestreadas desde la distribucion aprendida.

        Args:
            n_samples: Numero de muestras Monte Carlo a generar.

        Returns:
            Tensor con shape `(n_samples, batch, pred_len, c_out)`.
        """
        batch_size, time_steps, num_features = self.means.shape
        feature_samples: list[torch.Tensor] = []

        for feature_idx in range(num_features):
            ctx_feature = self.contexts[:, feature_idx, :]
            expanded_ctx = ctx_feature.unsqueeze(0).expand(n_samples, -1, -1)
            z = torch.randn(
                n_samples,
                batch_size,
                time_steps,
                device=self.means.device,
                dtype=self.means.dtype,
            )
            flow = self.flows[feature_idx]
            x0 = flow.inverse(z, context=expanded_ctx)
            feature_samples.append(x0.unsqueeze(-1))

        stacked = torch.cat(feature_samples, dim=-1)
        return stacked + self.means.unsqueeze(0)
