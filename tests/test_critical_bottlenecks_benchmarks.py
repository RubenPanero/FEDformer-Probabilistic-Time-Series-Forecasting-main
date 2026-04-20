from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "optimization"
    / "critical_bottlenecks_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location(
    "critical_bottlenecks_benchmark", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.benchmark
def test_critical_bottlenecks_benchmarks_collect_metrics() -> None:
    metrics = MODULE.run_all_benchmarks()

    assert set(metrics) == {"mc_dropout", "fourier_modes", "flow_checkpointing"}
    for benchmark_name, result in metrics.items():
        assert result["baseline_time_s"] >= 0.0, benchmark_name
        assert result["optimized_time_s"] >= 0.0, benchmark_name
        assert result["baseline_peak_bytes"] >= 0.0, benchmark_name
        assert result["optimized_peak_bytes"] >= 0.0, benchmark_name
