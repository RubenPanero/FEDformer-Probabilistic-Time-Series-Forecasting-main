# -*- coding: utf-8 -*-
"""
Componentes Encoder y Decoder del modelo FEDformer.
Refactorizado con estandarización en PyTorch >2.0 y compatibilidad completa con Python 3.10.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .layers import AttentionConfig, AttentionLayer, OptimizedSeriesDecomp


@dataclass(frozen=True)
class LayerConfig:
    """Configuration container for encoder and decoder layers."""

    # pylint: disable=too-many-instance-attributes

    d_model: int
    n_heads: int
    seq_len: int
    d_ff: int
    modes: int
    dropout: float
    activation: str
    moving_avg: list[int]


class EncoderLayer(nn.Module):
    """Optimized encoder layer with pure Python typing and secure explicit execution."""

    def __init__(self, config: LayerConfig) -> None:
        super().__init__()
        attention_cfg = AttentionConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            seq_len=config.seq_len,
            modes=config.modes,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleDict(
            {
                "attention": AttentionLayer(attention_cfg),
                "decomp": nn.ModuleList(
                    [
                        OptimizedSeriesDecomp(config.moving_avg),
                        OptimizedSeriesDecomp(config.moving_avg),
                    ]
                ),
                "conv": nn.ModuleList(
                    [
                        nn.Conv1d(config.d_model, config.d_ff, 1),
                        nn.Conv1d(config.d_ff, config.d_model, 1),
                    ]
                ),
                "norm": nn.ModuleList(
                    [
                        nn.LayerNorm(config.d_model),
                        nn.LayerNorm(config.d_model),
                    ]
                ),
                "dropout": nn.Dropout(config.dropout),
                "activation": nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply encoder self-attention with seasonal-trend decomposition."""
        # Type ignored visually on dynamic Dict fetching since PyTorch knows module flow.
        attention = self.layers["attention"]
        decomp_1 = self.layers["decomp"][0]  # type: ignore
        decomp_2 = self.layers["decomp"][1]  # type: ignore
        conv_1 = self.layers["conv"][0]  # type: ignore
        conv_2 = self.layers["conv"][1]  # type: ignore
        norm_1 = self.layers["norm"][0]  # type: ignore
        norm_2 = self.layers["norm"][1]  # type: ignore

        dropout = self.layers["dropout"]
        activation = self.layers["activation"]

        x_norm = norm_1(x)
        attn_out = attention(x_norm, x_norm, x_norm)

        # Decomp returns (residual, trend), solo usamos residual
        x, _ = decomp_1(x + attn_out)

        x_norm2 = norm_2(x)

        y = conv_1(x_norm2.transpose(1, 2))
        y = activation(y)
        y = dropout(y)

        y = conv_2(y)
        y = dropout(y).transpose(1, 2)

        res, _ = decomp_2(x + y)
        return res


class DecoderLayer(nn.Module):
    """Optimized decoder layer, pure typing validation."""

    def __init__(self, config: LayerConfig) -> None:
        super().__init__()
        attention_cfg = AttentionConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            seq_len=config.seq_len,
            modes=config.modes,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleDict(
            {
                "self_attention": AttentionLayer(attention_cfg),
                "cross_attention": AttentionLayer(attention_cfg),
                "decomp": nn.ModuleList(
                    [
                        OptimizedSeriesDecomp(config.moving_avg),
                        OptimizedSeriesDecomp(config.moving_avg),
                        OptimizedSeriesDecomp(config.moving_avg),
                    ]
                ),
                "conv": nn.ModuleList(
                    [
                        nn.Conv1d(config.d_model, config.d_ff, 1),
                        nn.Conv1d(config.d_ff, config.d_model, 1),
                    ]
                ),
                "norm": nn.ModuleList(
                    [
                        nn.LayerNorm(config.d_model),
                        nn.LayerNorm(config.d_model),
                        nn.LayerNorm(config.d_model),
                    ]
                ),
                "dropout": nn.Dropout(config.dropout),
                "activation": nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            }
        )

    def forward(
        self, x: torch.Tensor, cross: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply decoder self/cross-attention and return residual/trend."""
        self_attn = self.layers["self_attention"]
        cross_attn = self.layers["cross_attention"]

        decomp_1 = self.layers["decomp"][0]  # type: ignore
        decomp_2 = self.layers["decomp"][1]  # type: ignore
        decomp_3 = self.layers["decomp"][2]  # type: ignore

        conv_1 = self.layers["conv"][0]  # type: ignore
        conv_2 = self.layers["conv"][1]  # type: ignore

        norm_1 = self.layers["norm"][0]  # type: ignore
        norm_2 = self.layers["norm"][1]  # type: ignore
        norm_3 = self.layers["norm"][2]  # type: ignore

        dropout = self.layers["dropout"]
        activation = self.layers["activation"]

        x_norm = norm_1(x)
        x_res, trend1 = decomp_1(x + self_attn(x_norm, x_norm, x_norm))

        x_norm2 = norm_2(x_res)
        cross_norm = norm_3(cross)
        x_res, trend2 = decomp_2(x_res + cross_attn(x_norm2, cross_norm, cross_norm))

        y = conv_1(x_res.transpose(1, 2))
        y = activation(y)
        y = dropout(y)
        y = conv_2(y)
        y = dropout(y).transpose(1, 2)

        x_res, trend3 = decomp_3(x_res + y)

        return x_res, trend1 + trend2 + trend3
