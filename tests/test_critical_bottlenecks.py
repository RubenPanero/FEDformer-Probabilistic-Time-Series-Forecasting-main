from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from config import FEDformerConfig, apply_preset
from models.fedformer import Flow_FEDformer, NormalizingFlowDistribution
from training.trainer import WalkForwardTrainer
from training.utils import mc_dropout_inference
from utils import get_device

FIXTURE_CSV = "tests/fixtures/NVDA_features.csv"
DEVICE = get_device()


def _make_config(**kwargs: object) -> FEDformerConfig:
    return FEDformerConfig(
        target_features=["Close"],
        file_path=FIXTURE_CSV,
        **kwargs,
    )


def _make_batch(
    config: FEDformerConfig, batch_size: int = 2
) -> dict[str, torch.Tensor]:
    return {
        "x_enc": torch.randn(batch_size, config.seq_len, config.enc_in),
        "x_dec": torch.randn(
            batch_size, config.label_len + config.pred_len, config.dec_in
        ),
        "y_true": torch.randn(batch_size, config.pred_len, config.c_out),
        "x_regime": torch.randint(
            0, config.n_regimes, (batch_size, 1), dtype=torch.long
        ),
    }


def test_config_accepts_mc_dropout_eval_samples() -> None:
    cfg = _make_config(mc_dropout_eval_samples=12)
    assert cfg.mc_dropout_eval_samples == 12


def test_fourier_optimized_preset_sets_modes_to_48() -> None:
    cfg = _make_config()
    apply_preset(cfg, "fourier_optimized")
    assert cfg.modes == 48


def test_mc_dropout_inference_batched_matches_unbatched() -> None:
    config = _make_config(
        seq_len=24,
        label_len=8,
        pred_len=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=1,
        d_layers=1,
        modes=4,
        dropout=0.2,
        n_flow_layers=2,
        flow_hidden_dim=16,
    )
    model = Flow_FEDformer(config)
    batch = _make_batch(config, batch_size=2)

    torch.manual_seed(2026)
    samples_unbatched = mc_dropout_inference(
        model, batch, n_samples=4, use_flow_sampling=True, mc_batch_size=1
    )

    torch.manual_seed(2026)
    samples_batched = mc_dropout_inference(
        model, batch, n_samples=4, use_flow_sampling=True, mc_batch_size=2
    )

    assert samples_batched.shape == samples_unbatched.shape
    assert torch.allclose(samples_batched, samples_unbatched)


def test_mc_dropout_inference_still_runs_one_forward_per_sample() -> None:
    class CountingDistribution:
        def __init__(self, sample: torch.Tensor) -> None:
            self._sample = sample

        def sample(self, n_samples: int) -> torch.Tensor:
            return self._sample.repeat(n_samples, 1, 1, 1)

    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.config = SimpleNamespace(pred_len=4, c_out=1)

        def forward(
            self, x_enc: torch.Tensor, _x_dec: torch.Tensor, _x_regime: torch.Tensor
        ) -> CountingDistribution:
            self.calls += 1
            sample = torch.full(
                (1, x_enc.size(0), self.config.pred_len, self.config.c_out),
                fill_value=float(self.calls),
                device=x_enc.device,
            )
            return CountingDistribution(sample)

    batch = {
        "x_enc": torch.randn(3, 10, 2),
        "x_dec": torch.randn(3, 9, 2),
        "x_regime": torch.zeros(3, 1, dtype=torch.long),
    }
    model = CountingModel()

    samples = mc_dropout_inference(
        model,
        batch,
        n_samples=5,
        use_flow_sampling=True,
        mc_batch_size=3,
    )

    assert model.calls == 5
    assert samples.shape == (5, 3, 4, 1)
    assert torch.equal(
        samples[:, 0, 0, 0], torch.arange(1.0, 6.0, device=samples.device)
    )


def test_mc_dropout_inference_restores_model_mode() -> None:
    config = _make_config(pred_len=4)
    model = Flow_FEDformer(config)
    batch = _make_batch(config, batch_size=2)

    model.eval()
    _ = mc_dropout_inference(
        model, batch, n_samples=2, use_flow_sampling=False, mc_batch_size=2
    )
    assert model.training is False

    model.train()
    _ = mc_dropout_inference(
        model, batch, n_samples=2, use_flow_sampling=False, mc_batch_size=2
    )
    assert model.training is True


def test_mc_dropout_inference_keeps_fallback_shape_when_sampling_fails() -> None:
    class ExplodingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(pred_len=4, c_out=1)

        def forward(self, *_args, **_kwargs) -> torch.Tensor:
            raise RuntimeError("boom")

    batch = {
        "x_enc": torch.randn(3, 10, 2),
        "x_dec": torch.randn(3, 9, 2),
        "x_regime": torch.zeros(3, 1, dtype=torch.long),
    }

    samples = mc_dropout_inference(
        ExplodingModel(),
        batch,
        n_samples=5,
        use_flow_sampling=True,
        mc_batch_size=3,
    )

    assert samples.shape == (5, 3, 4, 1)
    assert torch.count_nonzero(samples) == 0


def test_trainer_evaluate_model_uses_configured_mc_dropout_eval_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(mc_dropout_eval_samples=7, pred_len=4)
    trainer = WalkForwardTrainer(config, full_dataset=SimpleNamespace())
    batch = _make_batch(config, batch_size=2)
    captured: list[tuple[int, int]] = []

    def fake_mc_dropout_inference(
        model, batch, n_samples, use_flow_sampling, mc_batch_size
    ):
        del model, batch, use_flow_sampling
        captured.append((n_samples, mc_batch_size))
        return torch.ones(n_samples, 2, config.pred_len, config.c_out)

    monkeypatch.setattr(
        "training.trainer.mc_dropout_inference", fake_mc_dropout_inference
    )

    preds, gt, samples, quantiles = trainer._evaluate_model(
        model=Flow_FEDformer(config), test_loader=[batch]
    )

    assert captured == [(7, 10)]
    assert preds.shape == (2, config.pred_len, config.c_out)
    assert gt.shape == (2, config.pred_len, config.c_out)
    assert samples.shape == (7, 2, config.pred_len, config.c_out)
    assert quantiles.shape == (3, 2, config.pred_len, config.c_out)


def test_flow_log_prob_checkpointing_matches_non_checkpointed() -> None:
    base_cfg = _make_config(
        seq_len=24,
        label_len=8,
        pred_len=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=1,
        d_layers=1,
        modes=4,
        dropout=0.0,
        n_flow_layers=2,
        flow_hidden_dim=16,
    )
    ckpt_cfg = _make_config(
        seq_len=24,
        label_len=8,
        pred_len=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=1,
        d_layers=1,
        modes=4,
        dropout=0.0,
        n_flow_layers=2,
        flow_hidden_dim=16,
        use_gradient_checkpointing=True,
    )

    model_plain = Flow_FEDformer(base_cfg)
    model_ckpt = Flow_FEDformer(ckpt_cfg)
    model_ckpt.load_state_dict(model_plain.state_dict())
    model_plain.train()
    model_ckpt.train()
    batch = _make_batch(base_cfg, batch_size=2)

    lp_plain = model_plain(batch["x_enc"], batch["x_dec"], batch["x_regime"]).log_prob(
        batch["y_true"]
    )
    lp_ckpt = model_ckpt(batch["x_enc"], batch["x_dec"], batch["x_regime"]).log_prob(
        batch["y_true"]
    )

    assert torch.allclose(lp_plain, lp_ckpt, atol=1e-6)


def test_flow_log_prob_checkpointing_preserves_gradients() -> None:
    config = _make_config(
        seq_len=24,
        label_len=8,
        pred_len=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=1,
        d_layers=1,
        modes=4,
        dropout=0.0,
        n_flow_layers=2,
        flow_hidden_dim=16,
        use_gradient_checkpointing=True,
    )
    model = Flow_FEDformer(config)
    model.train()
    batch = _make_batch(config, batch_size=2)

    loss = (
        -model(batch["x_enc"], batch["x_dec"], batch["x_regime"])
        .log_prob(batch["y_true"])
        .mean()
    )
    loss.backward()

    flow_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("flows.")
    ]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in flow_grads)


def test_trainer_backward_path_preserves_flow_gradients_with_checkpointing() -> None:
    config = _make_config(
        seq_len=24,
        label_len=8,
        pred_len=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=1,
        d_layers=1,
        modes=4,
        dropout=0.0,
        n_flow_layers=2,
        flow_hidden_dim=16,
        use_gradient_checkpointing=True,
    )
    trainer = WalkForwardTrainer(config, full_dataset=SimpleNamespace())
    model = Flow_FEDformer(config).to(DEVICE)
    model.train()
    batch = _make_batch(config, batch_size=2)

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        loss = trainer._forward_and_compute_loss(
            model,
            trainer._prepare_batch(batch),
            scaler=None,
            accumulation_steps=1,
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    assert loss is not None
    assert torch.isfinite(loss)
    flow_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("flows.")
    ]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in flow_grads)


def test_distribution_log_prob_uses_checkpoint_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow = torch.nn.ModuleList([torch.nn.Identity()])
    means = torch.zeros(2, 4, 1)
    contexts = torch.zeros(2, 1, 1)
    distribution = NormalizingFlowDistribution(
        means=means,
        flows=flow,
        contexts=contexts,
        use_gradient_checkpointing=True,
        training=True,
    )
    calls: list[str] = []

    def fake_checkpoint(function, *args, **kwargs):
        del kwargs
        calls.append("checkpoint")
        return function(*args)

    def fake_log_prob(y_feature, base_mean=None, context=None):
        del context
        centered = y_feature if base_mean is None else y_feature - base_mean
        return -centered.square().sum(dim=-1)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", fake_checkpoint)
    monkeypatch.setattr(flow[0], "log_prob", fake_log_prob, raising=False)

    y_true = torch.ones(2, 4, 1)
    _ = distribution.log_prob(y_true)

    assert calls == ["checkpoint"]


@pytest.mark.benchmark
def test_fourier_modes_benchmark_smoke() -> None:
    from models.layers import FourierAttention

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timings: dict[int, float] = {}

    for modes in (64, 48):
        attention = FourierAttention(d_keys=8, seq_len=96, modes=modes).to(device)
        x = torch.randn(8, 4, 96, 8, device=device)
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        if device.type == "cuda":
            assert start is not None and end is not None
            torch.cuda.synchronize()
            start.record()
            _ = attention(x)
            end.record()
            torch.cuda.synchronize()
            timings[modes] = float(start.elapsed_time(end))
        else:
            import time

            t0 = time.perf_counter()
            _ = attention(x)
            timings[modes] = time.perf_counter() - t0

    assert 48 in timings
    assert 64 in timings
