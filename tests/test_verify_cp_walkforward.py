# -*- coding: utf-8 -*-
"""Tests para scripts/verify_cp_walkforward.py."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import verify_cp_walkforward as cpwf


def test_build_training_cmd_includes_cp_walkforward() -> None:
    """El comando generado siempre incluye --cp-walkforward y flags base."""
    args = cpwf.parse_args.__globals__["argparse"].Namespace(
        csv="data/NVDA_features.csv",
        targets="Close",
        seq_len=96,
        pred_len=20,
        batch_size=64,
        splits=4,
        return_transform="log_return",
        metric_space="returns",
        gradient_clip_norm=0.5,
        seed=42,
        results_dir=Path("results"),
        objective_threshold=0.8,
        python_cmd="python3",
        compile_mode=None,
        report_only=False,
        skip_gpu_info=False,
    )

    cmd = cpwf.build_training_cmd(args)

    assert cmd[:2] == ["python3", "main.py"]
    assert "--cp-walkforward" in cmd
    assert "--save-results" in cmd
    assert "--no-show" in cmd


def test_build_training_cmd_includes_compile_mode_when_provided() -> None:
    """Si compile_mode no es None, se propaga al comando."""
    args = cpwf.parse_args.__globals__["argparse"].Namespace(
        csv="data/NVDA_features.csv",
        targets="Close",
        seq_len=96,
        pred_len=20,
        batch_size=64,
        splits=4,
        return_transform="log_return",
        metric_space="returns",
        gradient_clip_norm=0.5,
        seed=42,
        results_dir=Path("results"),
        objective_threshold=0.8,
        python_cmd="python3",
        compile_mode="",
        report_only=False,
        skip_gpu_info=False,
    )

    cmd = cpwf.build_training_cmd(args)

    assert "--compile-mode" in cmd
    idx = cmd.index("--compile-mode")
    assert cmd[idx + 1] == ""


def test_load_latest_manifest_returns_none_when_empty(tmp_path: Path) -> None:
    """Si no hay manifests, retorna None."""
    loaded = cpwf.load_latest_manifest(tmp_path)
    assert loaded is None


def test_load_latest_manifest_picks_most_recent(tmp_path: Path) -> None:
    """Carga el manifest con mtime más reciente."""
    old_path = tmp_path / "run_manifest_old.json"
    new_path = tmp_path / "run_manifest_new.json"

    old_path.write_text(json.dumps({"metrics": {"cp_wf_coverage_80": 0.75}}))
    new_path.write_text(json.dumps({"metrics": {"cp_wf_coverage_80": 0.82}}))

    base_time = time.time()
    os.utime(old_path, (base_time, base_time))
    os.utime(new_path, (base_time + 1.0, base_time + 1.0))
    assert new_path.stat().st_mtime > old_path.stat().st_mtime

    loaded = cpwf.load_latest_manifest(tmp_path)
    assert loaded is not None
    manifest_path, manifest = loaded
    assert manifest_path == new_path
    assert manifest["metrics"]["cp_wf_coverage_80"] == 0.82


def test_report_results_returns_zero_when_threshold_met(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """cp_wf_coverage_80 >= threshold -> exit code 0 y resumen legible."""
    manifest = {
        "metrics": {
            "cp_wf_coverage_80": 0.81,
            "cp_wf_folds_calibrated": 3,
        }
    }
    path = tmp_path / "run_manifest_20260329_120000.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    code = cpwf.report_results(tmp_path, objective_threshold=0.80)
    captured = capsys.readouterr()

    assert code == 0
    assert "cp_wf_coverage_80" in captured.out
    assert "OBJETIVO CUMPLIDO" in captured.out


def test_report_results_returns_two_when_threshold_not_met(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """cp_wf_coverage_80 < threshold -> exit code 2."""
    manifest = {
        "metrics": {
            "cp_wf_coverage_80": 0.71,
            "cp_wf_folds_calibrated": 2,
        }
    }
    path = tmp_path / "run_manifest_20260329_120000.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    code = cpwf.report_results(tmp_path, objective_threshold=0.80)
    captured = capsys.readouterr()

    assert code == 2
    assert "BAJO OBJETIVO" in captured.out


def test_report_results_returns_one_when_no_cp_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Si el manifest no contiene claves cp_wf_, retorna 1."""
    manifest = {
        "metrics": {
            "sharpe_ratio": 0.9,
            "sortino_ratio": 1.2,
        }
    }
    path = tmp_path / "run_manifest_20260329_120000.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    code = cpwf.report_results(tmp_path, objective_threshold=0.80)
    captured = capsys.readouterr()

    assert code == 1
    assert "No se encontraron métricas cp_wf_" in captured.out


def test_run_training_invokes_subprocess_with_agg_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """run_training propaga MPLBACKEND=Agg y no lanza excepción."""
    args = cpwf.parse_args.__globals__["argparse"].Namespace(
        csv="data/NVDA_features.csv",
        targets="Close",
        seq_len=96,
        pred_len=20,
        batch_size=64,
        splits=4,
        return_transform="log_return",
        metric_space="returns",
        gradient_clip_norm=0.5,
        seed=42,
        results_dir=Path("results"),
        objective_threshold=0.8,
        python_cmd="python3",
        compile_mode=None,
        report_only=False,
        skip_gpu_info=False,
    )

    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with patch(
        "scripts.verify_cp_walkforward.subprocess.run", return_value=mock_proc
    ) as run_mock:
        code = cpwf.run_training(args)

    captured = capsys.readouterr()
    assert code == 0
    assert "Comando:" in captured.out
    _, kwargs = run_mock.call_args
    assert kwargs["env"]["MPLBACKEND"] == "Agg"
    assert kwargs["cwd"] == cpwf.PROJECT_DIR


def test_main_report_only_skips_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() con --report-only no llama a run_training."""
    manifest = {
        "metrics": {
            "cp_wf_coverage_80": 0.83,
            "cp_wf_folds_calibrated": 3,
        }
    }
    manifest_path = tmp_path / "run_manifest_20260329_120000.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        cpwf.sys,
        "argv",
        [
            "verify_cp_walkforward.py",
            "--report-only",
            "--skip-gpu-info",
            "--results-dir",
            str(tmp_path),
        ],
    )

    with patch("scripts.verify_cp_walkforward.run_training") as run_training_mock:
        code = cpwf.main()

    assert code == 0
    run_training_mock.assert_not_called()
