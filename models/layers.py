# -*- coding: utf-8 -*-
"""
Componentes básicos del modelo FEDformer: capas de atención y descomposición.
Optimizados y tipados restrictivamente para estándar productivo (Python 3.10+).
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.fft import irfft, rfft
from torch.nn.functional import avg_pool1d, interpolate, pad


def _apply_rfft(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Aplica rfft casteando a float32 si el dtype no está soportado (e.g. bfloat16).

    torch.fft.rfft no soporta bfloat16/float16. Con AMP en GPUs Ada Lovelace
    los tensores llegan en bfloat16, por lo que se castea temporalmente a float32.
    """
    return rfft(x.float(), dim=dim)  # pylint: disable=not-callable


def _apply_irfft(x: torch.Tensor, *, n: int, dim: int) -> torch.Tensor:
    """Wrapper around torch.fft.irfft for static analysis."""
    return irfft(x, n=n, dim=dim)  # pylint: disable=not-callable


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for AttentionLayer construction."""

    d_model: int
    n_heads: int
    seq_len: int
    modes: int
    dropout: float


class OptimizedSeriesDecomp(nn.Module):
    """Optimized decomposition with reduced memory footprint and pure Python typing."""

    def __init__(self, kernel_sizes: list[int]) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split input into seasonal and trend signals via moving averages."""
        x_t = x.transpose(1, 2)  # [batch, len, channels] -> [batch, channels, len]
        trends: list[torch.Tensor] = []

        for kernel_size in self.kernel_sizes:
            if kernel_size <= 1:
                trends.append(x_t)
                continue

            left = (kernel_size - 1) // 2
            right = kernel_size - 1 - left
            # Pad estricto
            x_padded = pad(
                x_t, (left, right), mode="replicate"
            )  # [batch, channels, len + left + right]

            trend_val = avg_pool1d(x_padded, kernel_size=kernel_size, stride=1)  # pylint: disable=not-callable
            trends.append(trend_val)

        trend = (
            torch.stack(trends).mean(0).transpose(1, 2)
        )  # Back to [batch, len, channels]

        return x - trend, trend


class FourierAttention(nn.Module):
    """Memory-optimized Fourier attention with better initialization."""

    def __init__(self, d_keys: int, seq_len: int, modes: int = 64) -> None:
        super().__init__()
        self.modes = min(modes, max(1, seq_len // 2))

        # Ensures reproducibility across runs with same seq_len and modes
        generator = torch.Generator()
        seed = (seq_len * 1009 + self.modes * 1013) % (2**31 - 1)
        generator.manual_seed(seed)

        indices = torch.randperm(max(1, seq_len // 2), generator=generator)[
            : self.modes
        ].sort()[0]
        self.register_buffer("index", indices)

        # Stable weight initialization (separate real/imag)
        std = 0.02 / max(1.0, math.sqrt(d_keys))
        self.weights_real = nn.Parameter(torch.randn(d_keys, d_keys, self.modes) * std)
        self.weights_imag = nn.Parameter(torch.randn(d_keys, d_keys, self.modes) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier attention over the provided multi-head representations."""
        _b, _h, seq_len, _d = x.shape
        orig_dtype = x.dtype  # preservar dtype original para restaurar tras irfft
        x = x.transpose(-1, -2)

        x_ft = _apply_rfft(x, dim=-1)

        weights_c = torch.complex(self.weights_real, self.weights_imag)
        # Type ignore for static analyzer regarding buffer properties
        idx_buffer: torch.Tensor = self.index  # type: ignore
        selected = x_ft[..., idx_buffer]

        processed_modes = torch.einsum("bhei,eoi->bhoi", selected, weights_c)

        out_ft = torch.zeros_like(x_ft)
        out_ft[..., idx_buffer] = processed_modes

        # .to(orig_dtype) restaura bfloat16/float16 que _apply_rfft convirtió a float32
        return _apply_irfft(out_ft, n=seq_len, dim=-1).to(orig_dtype).transpose(-1, -2)


class AttentionLayer(nn.Module):
    """Enhanced attention layer with better memory management."""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_keys = config.d_model // config.n_heads

        self.fourier_attention = FourierAttention(
            self.d_keys, config.seq_len, config.modes
        )
        self.query_proj = nn.Linear(config.d_model, config.d_model)
        self.key_proj = nn.Linear(config.d_model, config.d_model)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Project queries/keys/values and apply Fourier attention."""
        # pylint: disable=too-many-locals
        batch_q, len_q, _ = q.shape
        batch_k, len_k, _ = k.shape

        q_heads = (
            self.query_proj(q)
            .view(batch_q, len_q, self.n_heads, self.d_keys)
            .transpose(1, 2)
        )
        k_heads = (
            self.key_proj(k)
            .view(batch_k, len_k, self.n_heads, self.d_keys)
            .transpose(1, 2)
        )
        v_heads = (
            self.value_proj(v)
            .view(batch_k, len_k, self.n_heads, self.d_keys)
            .transpose(1, 2)
        )

        if len_k != len_q:
            head_batch = batch_k * self.n_heads
            k_reshaped = (
                k_heads.transpose(1, 2)
                .contiguous()
                .view(head_batch, self.d_keys, len_k)
            )
            v_reshaped = (
                v_heads.transpose(1, 2)
                .contiguous()
                .view(head_batch, self.d_keys, len_k)
            )
            k_resampled = interpolate(
                k_reshaped, size=len_q, mode="linear", align_corners=False
            )
            v_resampled = interpolate(
                v_reshaped, size=len_q, mode="linear", align_corners=False
            )
            k_heads = k_resampled.view(
                batch_k, self.n_heads, self.d_keys, len_q
            ).transpose(2, 3)
            v_heads = v_resampled.view(
                batch_k, self.n_heads, self.d_keys, len_q
            ).transpose(2, 3)

        # Usando la fourier attention
        attn_out = self.fourier_attention(q_heads * k_heads) * v_heads

        batch_out, len_out = q.shape[:2]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_out, len_out, -1)

        return self.dropout(self.out_proj(attn_out))
