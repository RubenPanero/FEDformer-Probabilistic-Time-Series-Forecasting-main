# -*- coding: utf-8 -*-
"""
Tests unitarios para EncoderLayer y DecoderLayer del modelo FEDformer.
Cubre: forward pass, shapes de salida, descomposición estacional,
cross-attention con dimensiones distintas y flujo de gradientes.
"""

import torch

from models.encoder_decoder import DecoderLayer, EncoderLayer, LayerConfig


# ---------------------------------------------------------------------------
# Configuración de test con parámetros pequeños para mantener rapidez
# ---------------------------------------------------------------------------

D_MODEL = 32
N_HEADS = 2
D_FF = 64
SEQ_LEN = 16  # longitud de secuencia para el encoder
DEC_LEN = 12  # longitud de secuencia para el decoder (label_len + pred_len)
ENC_LEN_CROSS = 20  # longitud distinta para probar cross-attention
MODES = 4  # número de modos de Fourier (≤ SEQ_LEN // 2)
DROPOUT = 0.0  # sin dropout para reproducibilidad
MOVING_AVG = [3]  # kernel pequeño para los tests


def _make_encoder_config(seq_len: int = SEQ_LEN) -> LayerConfig:
    """Crea un LayerConfig mínimo para instanciar EncoderLayer en tests."""
    return LayerConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        seq_len=seq_len,
        d_ff=D_FF,
        modes=MODES,
        dropout=DROPOUT,
        activation="gelu",
        moving_avg=MOVING_AVG,
    )


def _make_decoder_config(seq_len: int = DEC_LEN) -> LayerConfig:
    """Crea un LayerConfig mínimo para instanciar DecoderLayer en tests."""
    return LayerConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        seq_len=seq_len,
        d_ff=D_FF,
        modes=MODES,
        dropout=DROPOUT,
        activation="relu",
        moving_avg=MOVING_AVG,
    )


# ---------------------------------------------------------------------------
# Tests de EncoderLayer
# ---------------------------------------------------------------------------


class TestEncoderLayer:
    """Tests del forward pass y propiedades de EncoderLayer."""

    def test_forward_shape_basico(self) -> None:
        """forward() debe devolver tensor con shape (batch, seq_len, d_model)."""
        torch.manual_seed(0)
        layer = EncoderLayer(_make_encoder_config()).eval()
        batch_size = 4
        x = torch.randn(batch_size, SEQ_LEN, D_MODEL)

        out = layer(x)

        assert isinstance(out, torch.Tensor), "La salida debe ser un Tensor"
        assert out.shape == (batch_size, SEQ_LEN, D_MODEL), (
            f"Shape esperado ({batch_size}, {SEQ_LEN}, {D_MODEL}), obtenido {out.shape}"
        )

    def test_forward_dtype_preservado(self) -> None:
        """El dtype float32 de la entrada debe preservarse en la salida."""
        torch.manual_seed(1)
        layer = EncoderLayer(_make_encoder_config()).eval()
        x = torch.randn(2, SEQ_LEN, D_MODEL, dtype=torch.float32)

        out = layer(x)

        assert out.dtype == torch.float32, "El dtype de salida debe ser float32"

    def test_descomposicion_estacional_shapes(self) -> None:
        """OptimizedSeriesDecomp interno devuelve (seasonal, trend) con shapes correctos."""
        torch.manual_seed(2)
        from models.layers import OptimizedSeriesDecomp

        decomp = OptimizedSeriesDecomp(MOVING_AVG)
        batch_size = 3
        x = torch.randn(batch_size, SEQ_LEN, D_MODEL)

        seasonal, trend = decomp(x)

        # Ambos componentes deben tener el mismo shape que la entrada
        assert seasonal.shape == x.shape, (
            f"seasonal shape {seasonal.shape} != input shape {x.shape}"
        )
        assert trend.shape == x.shape, (
            f"trend shape {trend.shape} != input shape {x.shape}"
        )

    def test_descomposicion_estacional_suma_original(self) -> None:
        """seasonal + trend debe reconstruir la entrada original."""
        torch.manual_seed(3)
        from models.layers import OptimizedSeriesDecomp

        decomp = OptimizedSeriesDecomp(MOVING_AVG)
        x = torch.randn(2, SEQ_LEN, D_MODEL)

        seasonal, trend = decomp(x)

        # La suma debe recuperar x
        assert torch.allclose(seasonal + trend, x, atol=1e-5), (
            "seasonal + trend no reconstruye la entrada original"
        )

    def test_forward_batch_size_1(self) -> None:
        """EncoderLayer debe funcionar correctamente con batch_size=1."""
        torch.manual_seed(4)
        layer = EncoderLayer(_make_encoder_config()).eval()
        x = torch.randn(1, SEQ_LEN, D_MODEL)

        out = layer(x)

        assert out.shape == (1, SEQ_LEN, D_MODEL), (
            f"Shape esperado (1, {SEQ_LEN}, {D_MODEL}), obtenido {out.shape}"
        )

    def test_forward_output_finito(self) -> None:
        """La salida no debe contener NaN ni Inf."""
        torch.manual_seed(5)
        layer = EncoderLayer(_make_encoder_config()).eval()
        x = torch.randn(3, SEQ_LEN, D_MODEL)

        out = layer(x)

        assert torch.isfinite(out).all(), "La salida contiene NaN o Inf"

    def test_gradient_flow(self) -> None:
        """loss.backward() no debe lanzar error y los pesos deben tener gradientes."""
        torch.manual_seed(6)
        layer = EncoderLayer(_make_encoder_config()).train()
        x = torch.randn(2, SEQ_LEN, D_MODEL, requires_grad=True)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        # Verificar que al menos un parámetro recibió gradiente
        tiene_grads = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.parameters()
        )
        assert tiene_grads, "Ningún parámetro del EncoderLayer recibió gradiente"

    def test_activation_relu(self) -> None:
        """EncoderLayer con activation='relu' debe instanciarse y ejecutarse sin error."""
        torch.manual_seed(7)
        cfg = LayerConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            seq_len=SEQ_LEN,
            d_ff=D_FF,
            modes=MODES,
            dropout=DROPOUT,
            activation="relu",
            moving_avg=MOVING_AVG,
        )
        layer = EncoderLayer(cfg).eval()
        x = torch.randn(2, SEQ_LEN, D_MODEL)

        out = layer(x)

        assert out.shape == (2, SEQ_LEN, D_MODEL)

    def test_forward_seq_len_diferente(self) -> None:
        """EncoderLayer acepta distintas longitudes de secuencia en la config."""
        torch.manual_seed(8)
        seq_len_alt = 24
        layer = EncoderLayer(_make_encoder_config(seq_len=seq_len_alt)).eval()
        x = torch.randn(2, seq_len_alt, D_MODEL)

        out = layer(x)

        assert out.shape == (2, seq_len_alt, D_MODEL)


# ---------------------------------------------------------------------------
# Tests de DecoderLayer
# ---------------------------------------------------------------------------


class TestDecoderLayer:
    """Tests del forward pass y propiedades de DecoderLayer."""

    def test_forward_devuelve_tupla(self) -> None:
        """forward() debe devolver una tupla de dos tensores."""
        torch.manual_seed(10)
        layer = DecoderLayer(_make_decoder_config()).eval()
        x = torch.randn(4, DEC_LEN, D_MODEL)
        cross = torch.randn(4, SEQ_LEN, D_MODEL)

        resultado = layer(x, cross)

        assert isinstance(resultado, tuple), "DecoderLayer.forward debe devolver tuple"
        assert len(resultado) == 2, "La tupla debe contener exactamente 2 elementos"

    def test_forward_shape_output(self) -> None:
        """El tensor residual de salida debe tener shape (batch, dec_len, d_model)."""
        torch.manual_seed(11)
        layer = DecoderLayer(_make_decoder_config()).eval()
        batch_size = 4
        x = torch.randn(batch_size, DEC_LEN, D_MODEL)
        cross = torch.randn(batch_size, SEQ_LEN, D_MODEL)

        out, trend = layer(x, cross)

        assert out.shape == (batch_size, DEC_LEN, D_MODEL), (
            f"out shape esperado ({batch_size}, {DEC_LEN}, {D_MODEL}), obtenido {out.shape}"
        )

    def test_forward_shape_trend(self) -> None:
        """El tensor trend de salida debe tener shape (batch, dec_len, d_model)."""
        torch.manual_seed(12)
        layer = DecoderLayer(_make_decoder_config()).eval()
        batch_size = 4
        x = torch.randn(batch_size, DEC_LEN, D_MODEL)
        cross = torch.randn(batch_size, SEQ_LEN, D_MODEL)

        out, trend = layer(x, cross)

        assert trend.shape == (batch_size, DEC_LEN, D_MODEL), (
            f"trend shape esperado ({batch_size}, {DEC_LEN}, {D_MODEL}), obtenido {trend.shape}"
        )

    def test_cross_attention_enc_diferente_len(self) -> None:
        """cross-attention debe aceptar enc_out con seq_len distinta a dec_in."""
        torch.manual_seed(13)
        layer = DecoderLayer(_make_decoder_config()).eval()
        batch_size = 3
        x = torch.randn(batch_size, DEC_LEN, D_MODEL)
        # Encoder con longitud diferente a DEC_LEN
        cross = torch.randn(batch_size, ENC_LEN_CROSS, D_MODEL)

        out, trend = layer(x, cross)

        assert out.shape == (batch_size, DEC_LEN, D_MODEL), (
            "cross-attention con enc_len diferente no produce shape correcto en out"
        )
        assert trend.shape == (batch_size, DEC_LEN, D_MODEL), (
            "cross-attention con enc_len diferente no produce shape correcto en trend"
        )

    def test_forward_batch_size_1(self) -> None:
        """DecoderLayer debe funcionar correctamente con batch_size=1."""
        torch.manual_seed(14)
        layer = DecoderLayer(_make_decoder_config()).eval()
        x = torch.randn(1, DEC_LEN, D_MODEL)
        cross = torch.randn(1, SEQ_LEN, D_MODEL)

        out, trend = layer(x, cross)

        assert out.shape == (1, DEC_LEN, D_MODEL)
        assert trend.shape == (1, DEC_LEN, D_MODEL)

    def test_forward_output_finito(self) -> None:
        """La salida no debe contener NaN ni Inf."""
        torch.manual_seed(15)
        layer = DecoderLayer(_make_decoder_config()).eval()
        x = torch.randn(3, DEC_LEN, D_MODEL)
        cross = torch.randn(3, SEQ_LEN, D_MODEL)

        out, trend = layer(x, cross)

        assert torch.isfinite(out).all(), "out contiene NaN o Inf"
        assert torch.isfinite(trend).all(), "trend contiene NaN o Inf"

    def test_gradient_flow(self) -> None:
        """loss.backward() no debe lanzar error y los pesos deben tener gradientes."""
        torch.manual_seed(16)
        layer = DecoderLayer(_make_decoder_config()).train()
        x = torch.randn(2, DEC_LEN, D_MODEL, requires_grad=True)
        cross = torch.randn(2, SEQ_LEN, D_MODEL)

        out, trend = layer(x, cross)
        loss = out.sum() + trend.sum()
        loss.backward()

        # Verificar que al menos un parámetro recibió gradiente
        tiene_grads = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.parameters()
        )
        assert tiene_grads, "Ningún parámetro del DecoderLayer recibió gradiente"

    def test_gradient_fluye_desde_trend(self) -> None:
        """El componente trend también debe propagar gradientes correctamente."""
        torch.manual_seed(17)
        layer = DecoderLayer(_make_decoder_config()).train()
        x = torch.randn(2, DEC_LEN, D_MODEL, requires_grad=True)
        cross = torch.randn(2, SEQ_LEN, D_MODEL)

        _out, trend = layer(x, cross)
        # Solo se hace backward sobre trend
        loss = trend.sum()
        loss.backward()

        assert x.grad is not None, (
            "El gradiente no fluyó hasta la entrada x a través de trend"
        )

    def test_activation_gelu(self) -> None:
        """DecoderLayer con activation='gelu' debe instanciarse y ejecutarse sin error."""
        torch.manual_seed(18)
        cfg = LayerConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            seq_len=DEC_LEN,
            d_ff=D_FF,
            modes=MODES,
            dropout=DROPOUT,
            activation="gelu",
            moving_avg=MOVING_AVG,
        )
        layer = DecoderLayer(cfg).eval()
        x = torch.randn(2, DEC_LEN, D_MODEL)
        cross = torch.randn(2, SEQ_LEN, D_MODEL)

        out, trend = layer(x, cross)

        assert out.shape == (2, DEC_LEN, D_MODEL)
        assert trend.shape == (2, DEC_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# Tests de integración encoder → decoder
# ---------------------------------------------------------------------------


class TestEncoderDecoderIntegracion:
    """Tests que verifican el pipeline completo encoder → decoder."""

    def test_pipeline_encoder_a_decoder(self) -> None:
        """La salida del encoder debe poder usarse como cross en el decoder."""
        torch.manual_seed(20)
        encoder = EncoderLayer(_make_encoder_config()).eval()
        decoder = DecoderLayer(_make_decoder_config()).eval()

        batch_size = 3
        x_enc = torch.randn(batch_size, SEQ_LEN, D_MODEL)
        x_dec = torch.randn(batch_size, DEC_LEN, D_MODEL)

        # Pasar por encoder
        enc_out = encoder(x_enc)
        assert enc_out.shape == (batch_size, SEQ_LEN, D_MODEL)

        # Usar salida del encoder como cross para el decoder
        dec_out, trend = decoder(x_dec, enc_out)

        assert dec_out.shape == (batch_size, DEC_LEN, D_MODEL)
        assert trend.shape == (batch_size, DEC_LEN, D_MODEL)

    def test_pipeline_backward_conjunto(self) -> None:
        """El backward a través del pipeline encoder→decoder no debe dar error."""
        torch.manual_seed(21)
        encoder = EncoderLayer(_make_encoder_config()).train()
        decoder = DecoderLayer(_make_decoder_config()).train()

        batch_size = 2
        x_enc = torch.randn(batch_size, SEQ_LEN, D_MODEL, requires_grad=True)
        x_dec = torch.randn(batch_size, DEC_LEN, D_MODEL, requires_grad=True)

        enc_out = encoder(x_enc)
        dec_out, trend = decoder(x_dec, enc_out)

        loss = dec_out.sum() + trend.sum()
        loss.backward()

        assert x_enc.grad is not None, "Gradiente no fluye hasta x_enc"
        assert x_dec.grad is not None, "Gradiente no fluye hasta x_dec"
        assert torch.isfinite(x_enc.grad).all(), "Gradiente de x_enc contiene NaN/Inf"
        assert torch.isfinite(x_dec.grad).all(), "Gradiente de x_dec contiene NaN/Inf"
