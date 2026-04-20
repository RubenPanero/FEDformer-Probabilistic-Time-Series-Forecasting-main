import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator, Tuple

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from config import FEDformerConfig
    from models.fedformer import Flow_FEDformer

# Fixture estable versionada para tests; no depende de datasets locales bajo data/
DATA_CSV = str(ROOT / "tests" / "fixtures" / "NVDA_features.csv")


@pytest.fixture(scope="session", autouse=True)
def deterministic_seed() -> Generator[None, None, None]:
    """Set a reproducible seed for tests."""
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Best-effort deterministic algorithms when available
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True)
    yield


@pytest.fixture
def config() -> "FEDformerConfig":
    from config import FEDformerConfig

    return FEDformerConfig(target_features=["Close"], file_path=DATA_CSV)


@pytest.fixture
def model_factory(
    config: "FEDformerConfig",
) -> Callable[[bool], "Flow_FEDformer"]:
    from models.fedformer import Flow_FEDformer

    def _make(train: bool = False) -> Flow_FEDformer:
        m = Flow_FEDformer(config)
        if not train:
            m.eval()
        return m

    return _make


@pytest.fixture
def synthetic_batch(
    config: "FEDformerConfig",
) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Create a synthetic batch matching config-derived shapes.

    Returns: (x_enc, x_dec, x_regime, y_true)
    """

    def _make(
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = batch_size
        enc_in = config.enc_in
        dec_in = config.dec_in
        seq_len = config.seq_len
        label_len = config.label_len
        pred_len = config.pred_len
        x_enc = torch.randn(B, seq_len, enc_in, dtype=torch.float32)
        x_dec = torch.randn(B, label_len + pred_len, dec_in, dtype=torch.float32)
        x_regime = torch.randint(0, config.n_regimes, (B, 1, 1), dtype=torch.long)
        y = torch.randn(B, pred_len, config.c_out, dtype=torch.float32)
        return x_enc, x_dec, x_regime, y

    return _make
