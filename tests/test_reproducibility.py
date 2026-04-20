"""Tests for reproducibility utilities used across the project."""

import random

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from utils.helpers import set_seed


@pytest.mark.parametrize("seed", [1, 42, 123])
def test_set_seed_yields_reproducible_sequences(seed: int) -> None:
    """set_seed should synchronise RNGs for Python, NumPy and PyTorch."""
    set_seed(seed)
    python_value = random.random()
    numpy_value = float(np.random.rand())
    torch_value = torch.rand(2, 3)

    set_seed(seed)
    assert random.random() == pytest.approx(python_value)
    assert float(np.random.rand()) == pytest.approx(numpy_value)
    assert_close(torch.rand(2, 3), torch_value)
