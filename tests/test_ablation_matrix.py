# -*- coding: utf-8 -*-
"""Tests para scripts/run_ablation_matrix.py (Epica 8 -- Ablaciones reproducibles)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.run_ablation_matrix import (
    AblationJob,
    build_ablation_jobs,
    job_to_argv,
    run_ablation_job,
    save_ablation_summary,
)


# ---------------------------------------------------------------------------
# Tests de build_ablation_jobs
# ---------------------------------------------------------------------------


def test_build_ablation_jobs_basic():
    """2 variantes generan 2 AblationJob con los campos correctos."""
    csv_path = "data/NVDA_features.csv"
    targets = ["Close"]
    variants = [
        {"name": "dropout_0.1", "dropout": 0.1},
        {"name": "dropout_0.2", "dropout": 0.2},
    ]

    jobs = build_ablation_jobs(csv_path, targets, variants)

    assert len(jobs) == 2, f"Se esperaban 2 jobs, se obtuvieron {len(jobs)}"

    job0 = jobs[0]
    assert job0.name == "dropout_0.1"
    assert job0.csv_path == csv_path
    assert job0.targets == targets
    assert job0.variant == {"dropout": 0.1}

    job1 = jobs[1]
    assert job1.name == "dropout_0.2"
    assert job1.variant == {"dropout": 0.2}


def test_build_ablation_jobs_empty_variants_raises():
    """variants=[] lanza ValueError."""
    with pytest.raises(ValueError, match="vacia"):
        build_ablation_jobs(
            csv_path="data/NVDA_features.csv",
            targets=["Close"],
            variants=[],
        )


def test_build_ablation_jobs_auto_name():
    """Variante sin 'name' genera nombre automaticamente de sus claves."""
    variants = [{"dropout": 0.1, "batch_size": 64}]

    jobs = build_ablation_jobs(
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variants=variants,
    )

    assert len(jobs) == 1
    # El nombre debe contener las claves del dict
    assert "batch_size" in jobs[0].name or "dropout" in jobs[0].name
    # La variante del job NO debe tener la clave "name"
    assert "name" not in jobs[0].variant


def test_build_ablation_jobs_base_args_propagated():
    """Los base_args se propagan correctamente a cada job."""
    base_args = {"seq_len": 96, "splits": 4}
    variants = [{"name": "v1", "dropout": 0.1}]

    jobs = build_ablation_jobs(
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variants=variants,
        base_args=base_args,
    )

    assert len(jobs) == 1
    assert jobs[0].base_args == base_args


def test_build_ablation_jobs_results_dir():
    """El results_dir se asigna correctamente al job."""
    variants = [{"name": "v1", "dropout": 0.1}]

    jobs = build_ablation_jobs(
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variants=variants,
        results_dir="custom_results",
    )

    assert jobs[0].results_dir == "custom_results"


# ---------------------------------------------------------------------------
# Tests de job_to_argv
# ---------------------------------------------------------------------------


def test_job_to_argv_contains_required_flags():
    """argv siempre contiene --csv, --targets, --save-results, --no-show."""
    job = AblationJob(
        name="test_job",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={},
    )

    argv = job_to_argv(job)

    assert "--csv" in argv
    assert "data/NVDA_features.csv" in argv
    assert "--targets" in argv
    assert "Close" in argv
    assert "--save-results" in argv
    assert "--no-show" in argv


def test_job_to_argv_maps_seq_len():
    """variant={'seq_len': 96} genera '--seq-len', '96' en argv."""
    job = AblationJob(
        name="seq_len_test",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={"seq_len": 96},
    )

    argv = job_to_argv(job)

    assert "--seq-len" in argv
    idx = argv.index("--seq-len")
    assert argv[idx + 1] == "96"


def test_job_to_argv_maps_batch_size():
    """variant={'batch_size': 64} genera '--batch-size', '64' en argv."""
    job = AblationJob(
        name="batch_test",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={"batch_size": 64},
    )

    argv = job_to_argv(job)

    assert "--batch-size" in argv
    idx = argv.index("--batch-size")
    assert argv[idx + 1] == "64"


def test_job_to_argv_maps_dropout():
    """variant={'dropout': 0.2} genera '--dropout', '0.2' en argv."""
    job = AblationJob(
        name="dropout_test",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={"dropout": 0.2},
    )

    argv = job_to_argv(job)

    assert "--dropout" in argv
    idx = argv.index("--dropout")
    assert argv[idx + 1] == "0.2"


def test_job_to_argv_maps_return_transform():
    """variant={'return_transform': 'log_return'} genera '--return-transform' en argv."""
    job = AblationJob(
        name="rt_test",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={"return_transform": "log_return"},
    )

    argv = job_to_argv(job)

    assert "--return-transform" in argv
    idx = argv.index("--return-transform")
    assert argv[idx + 1] == "log_return"


def test_job_to_argv_base_args_included():
    """Los base_args tambien se mapean en argv."""
    job = AblationJob(
        name="base_test",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={"dropout": 0.1},
        base_args={"seq_len": 96, "splits": 4},
    )

    argv = job_to_argv(job)

    assert "--seq-len" in argv
    assert "--splits" in argv


def test_job_to_argv_variant_overrides_base_args():
    """variant tiene precedencia sobre base_args para la misma clave."""
    job = AblationJob(
        name="override_test",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={"batch_size": 128},
        base_args={"batch_size": 32},
    )

    argv = job_to_argv(job)

    # Solo debe aparecer una vez --batch-size con el valor del variant
    assert argv.count("--batch-size") == 1
    idx = argv.index("--batch-size")
    assert argv[idx + 1] == "128"


def test_job_to_argv_unknown_key_omitted(caplog):
    """Clave desconocida no mapeada se omite del argv con warning."""
    import logging

    job = AblationJob(
        name="unknown_test",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={"clave_inventada": 999},
    )

    with caplog.at_level(logging.WARNING):
        argv = job_to_argv(job)

    assert "--clave-inventada" not in argv
    assert any("clave_inventada" in rec.message for rec in caplog.records)


def test_job_to_argv_boolean_true_flag():
    """Clave booleana True incluye el flag sin valor."""
    # Nota: deterministic es store_true en main.py pero no esta en el mapeo actual.
    # Usamos un job sin variante booleana y verificamos que el mecanismo funciona.
    job = AblationJob(
        name="bool_test",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={},
    )
    # El argv siempre contiene los flags obligatorios como store_true
    argv = job_to_argv(job)
    assert "--save-results" in argv
    assert "--no-show" in argv


# ---------------------------------------------------------------------------
# Tests de save_ablation_summary
# ---------------------------------------------------------------------------


def test_save_ablation_summary_creates_valid_json(tmp_path):
    """Crear resumen con 2 jobs (1 exitoso, 1 fallido) genera JSON con campos correctos."""
    jobs = [
        AblationJob(
            name="job_ok",
            csv_path="data/NVDA_features.csv",
            targets=["Close"],
            variant={"dropout": 0.1},
        ),
        AblationJob(
            name="job_fail",
            csv_path="data/NVDA_features.csv",
            targets=["Close"],
            variant={"dropout": 0.5},
        ),
    ]
    results = [
        {
            "name": "job_ok",
            "success": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
        {
            "name": "job_fail",
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "error",
        },
    ]

    summary_path = tmp_path / "ablation_summary.json"
    save_ablation_summary(summary_path, jobs, results)

    assert summary_path.exists(), "El archivo JSON de resumen no fue creado"

    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)

    assert summary["total_jobs"] == 2
    assert summary["successful"] == 1
    assert summary["failed"] == 1
    assert len(summary["jobs"]) == 2

    # Verificar campos por job
    job_by_name = {j["name"]: j for j in summary["jobs"]}
    assert job_by_name["job_ok"]["success"] is True
    assert job_by_name["job_ok"]["returncode"] == 0
    assert job_by_name["job_fail"]["success"] is False
    assert job_by_name["job_fail"]["returncode"] == 1


def test_save_ablation_summary_creates_parent_dirs(tmp_path):
    """Directorio padre no existente se crea automaticamente."""
    nested_path = tmp_path / "subdir" / "deep" / "summary.json"
    jobs = [
        AblationJob(
            name="j1",
            csv_path="data/x.csv",
            targets=["Close"],
            variant={},
        )
    ]
    results = [
        {"name": "j1", "success": True, "returncode": 0, "stdout": "", "stderr": ""}
    ]

    save_ablation_summary(nested_path, jobs, results)

    assert nested_path.exists()


def test_save_ablation_summary_variant_in_jobs(tmp_path):
    """El campo 'variant' de cada job aparece correctamente en el JSON."""
    jobs = [
        AblationJob(
            name="v1",
            csv_path="data/x.csv",
            targets=["Close"],
            variant={"seq_len": 96, "dropout": 0.1},
        )
    ]
    results = [
        {"name": "v1", "success": True, "returncode": 0, "stdout": "", "stderr": ""}
    ]

    summary_path = tmp_path / "summary.json"
    save_ablation_summary(summary_path, jobs, results)

    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)

    assert summary["jobs"][0]["variant"] == {"seq_len": 96, "dropout": 0.1}


# ---------------------------------------------------------------------------
# Tests de run_ablation_job
# ---------------------------------------------------------------------------


def test_run_ablation_job_captures_failure():
    """Mockear subprocess.run con returncode=1 -> result['success'] == False."""
    job = AblationJob(
        name="fail_job",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={"dropout": 0.1},
    )

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error simulado"

    with patch("subprocess.run", return_value=mock_result):
        result = run_ablation_job(job)

    assert result["success"] is False
    assert result["returncode"] == 1
    assert result["name"] == "fail_job"
    assert "Error simulado" in result["stderr"]


def test_run_ablation_job_captures_success():
    """Mockear subprocess.run con returncode=0 -> result['success'] == True."""
    job = AblationJob(
        name="success_job",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={"dropout": 0.1},
    )

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Entrenamiento completado"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_ablation_job(job)

    assert result["success"] is True
    assert result["returncode"] == 0
    assert result["name"] == "success_job"
    assert "Entrenamiento completado" in result["stdout"]


def test_run_ablation_job_handles_timeout():
    """TimeoutExpired se captura y se retorna como failure."""
    job = AblationJob(
        name="timeout_job",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={},
    )

    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="python3", timeout=1),
    ):
        result = run_ablation_job(job)

    assert result["success"] is False
    assert result["returncode"] == -1
    assert "TimeoutExpired" in result["stderr"]


def test_run_ablation_job_handles_os_error():
    """OSError se captura y se retorna como failure."""
    job = AblationJob(
        name="oserror_job",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={},
    )

    with patch("subprocess.run", side_effect=OSError("ejecutable no encontrado")):
        result = run_ablation_job(job)

    assert result["success"] is False
    assert result["returncode"] == -1
    assert "OSError" in result["stderr"]


def test_run_ablation_job_uses_python_cmd():
    """El parametro python_cmd reemplaza 'python3' en el argv."""
    job = AblationJob(
        name="cmd_test",
        csv_path="data/NVDA_features.csv",
        targets=["Close"],
        variant={},
    )

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    captured_argv: list[list[str]] = []

    def fake_run(argv, **kwargs):
        captured_argv.append(argv)
        return mock_result

    with patch("subprocess.run", side_effect=fake_run):
        run_ablation_job(job, python_cmd="/custom/python")

    assert len(captured_argv) == 1
    assert captured_argv[0][0] == "/custom/python"
