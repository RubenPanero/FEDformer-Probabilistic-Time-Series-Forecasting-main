import torch
from models.flows import NormalizingFlow


def test_flow_sample_with_context() -> None:
    """sample() debe pasar el contexto a inverse() sin errores."""
    torch.manual_seed(0)
    d_model = 4
    flow = NormalizingFlow(n_layers=2, d_model=d_model, hidden_dim=16)
    context = torch.randn(10, d_model)
    samples = flow.sample(n_samples=10, context=context)
    assert samples.shape == (10, d_model)


def test_flow_sample_shape() -> None:
    """sample() sin contexto produce la forma esperada (n_samples, d_model)."""
    torch.manual_seed(0)
    d_model = 6
    flow = NormalizingFlow(n_layers=2, d_model=d_model, hidden_dim=16)
    samples = flow.sample(n_samples=5)
    assert samples.shape == (5, d_model)


def test_flow_roundtrip_even() -> None:
    torch.manual_seed(0)
    B, T = 2, 4
    flow = NormalizingFlow(n_layers=2, d_model=T, hidden_dim=16)
    x = torch.randn(B, T)
    z, _ = flow(x)
    x_recon = flow.inverse(z)
    assert torch.allclose(x, x_recon, atol=1e-5)


def test_flow_roundtrip_odd() -> None:
    torch.manual_seed(0)
    B, T = 2, 5
    flow = NormalizingFlow(n_layers=2, d_model=T, hidden_dim=16)
    x = torch.randn(B, T)
    z, _ = flow(x)
    x_recon = flow.inverse(z)
    assert torch.allclose(x, x_recon, atol=1e-5)
