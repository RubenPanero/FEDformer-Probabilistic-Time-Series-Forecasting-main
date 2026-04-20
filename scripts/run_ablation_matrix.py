#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para ejecutar matrices de ablaciones reproducibles sobre FEDformer.

Permite definir una lista de variantes de hiperparámetros, construir los jobs
correspondientes y ejecutarlos secuencialmente lanzando main.py como subproceso.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapeo de claves de variante a flags de main.py
# Solo se incluyen flags verificados en _parse_arguments de main.py
# ---------------------------------------------------------------------------
_VARIANT_KEY_TO_FLAG: dict[str, str] = {
    "seq_len": "--seq-len",
    "pred_len": "--pred-len",
    "batch_size": "--batch-size",
    "dropout": "--dropout",
    "return_transform": "--return-transform",
    "metric_space": "--metric-space",
    "monitor_metric": "--monitor-metric",
    "epochs": "--epochs",
    "splits": "--splits",
    "weight_decay": "--weight-decay",
    "scheduler_type": "--scheduler-type",
    "warmup_epochs": "--warmup-epochs",
    "patience": "--patience",
    "min_delta": "--min-delta",
    "gradient_clip_norm": "--gradient-clip-norm",
    "seed": "--seed",
    "preset": "--preset",
    "label_len": "--label-len",
    "grad_accum_steps": "--grad-accum-steps",
    "rehearsal_k": "--rehearsal-k",
    "rehearsal_epochs": "--rehearsal-epochs",
    "rehearsal_lr_mult": "--rehearsal-lr-mult",
}

# Nota: --lr, --n-flow-layers y --flow-hidden-dim NO existen en main.py actual
# (no expuestos en CLI). Si se anaden al CLI en el futuro, actualizar este mapeo.


@dataclass
class AblationJob:
    """Representa un job de ablacion: config base + una variante."""

    name: str  # identificador legible, ej. "dropout_0.1"
    csv_path: str  # ruta al CSV del dataset
    targets: list[str]  # columnas target, ej. ["Close"]
    variant: dict[str, Any]  # parametros a variar, ej. {"dropout": 0.1}
    base_args: dict[str, Any] = field(default_factory=dict)  # args base comunes
    results_dir: str = "results"  # directorio de salida


def _auto_name(variant: dict[str, Any]) -> str:
    """Genera nombre automatico de variante a partir de sus claves y valores.

    Args:
        variant: dict de parametros de la variante (sin clave "name").

    Returns:
        Cadena con pares clave_valor separados por doble guion bajo.
    """
    parts = []
    for key, val in sorted(variant.items()):
        # Normalizar floats para nombres de archivo legibles
        if isinstance(val, float):
            val_str = f"{val:.0e}" if abs(val) < 1e-2 else f"{val}"
        else:
            val_str = str(val)
        parts.append(f"{key}_{val_str}")
    return "__".join(parts) if parts else "default"


def build_ablation_jobs(
    csv_path: str,
    targets: list[str],
    variants: list[dict[str, Any]],
    base_args: dict[str, Any] | None = None,
    results_dir: str = "results",
) -> list[AblationJob]:
    """Construye lista de AblationJob a partir de variantes.

    Cada variante puede tener una clave "name" para identificarla.
    Si no tiene "name", se genera automaticamente de las claves del dict.

    Args:
        csv_path: ruta al CSV del dataset.
        targets: columnas target.
        variants: lista de dicts con parametros a variar.
        base_args: args base comunes a todos los jobs.
        results_dir: directorio de resultados.

    Returns:
        Lista de AblationJob.

    Raises:
        ValueError: si variants esta vacio.
    """
    if not variants:
        raise ValueError(
            "La lista de variantes no puede estar vacia. "
            "Proporciona al menos una variante con parametros a explorar."
        )

    effective_base = base_args or {}
    jobs: list[AblationJob] = []

    for variant in variants:
        # Extraer nombre explicito o generarlo automaticamente
        variant_copy = dict(variant)
        name = variant_copy.pop("name", None)
        if name is None:
            name = _auto_name(variant_copy)

        job = AblationJob(
            name=name,
            csv_path=csv_path,
            targets=list(targets),
            variant=variant_copy,
            base_args=dict(effective_base),
            results_dir=results_dir,
        )
        jobs.append(job)

    return jobs


def job_to_argv(job: AblationJob) -> list[str]:
    """Convierte un AblationJob a lista de argumentos para main.py.

    Mapea los campos conocidos de job.variant y job.base_args a flags de main.py.
    Siempre incluye: --csv, --targets, --save-results, --no-show.

    Solo se mapean flags que tienen entrada en _VARIANT_KEY_TO_FLAG.
    Claves desconocidas se omiten con un warning en el logger.

    Args:
        job: AblationJob a convertir.

    Returns:
        Lista de strings lista para pasar a subprocess.run.
    """
    argv: list[str] = ["python3", "main.py"]

    # Flags obligatorios
    argv += ["--csv", job.csv_path]
    argv += ["--targets", ",".join(job.targets)]
    argv += ["--save-results"]
    argv += ["--no-show"]

    # Combinar base_args y variant (variant tiene precedencia sobre base_args)
    combined: dict[str, Any] = {}
    combined.update(job.base_args)
    combined.update(job.variant)

    for key, val in combined.items():
        flag = _VARIANT_KEY_TO_FLAG.get(key)
        if flag is None:
            logger.warning(
                "Clave de variante '%s' no tiene mapeo a flag de main.py -- omitida.",
                key,
            )
            continue
        # Flags booleanos (store_true): solo incluir si el valor es True
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        else:
            argv += [flag, str(val)]

    return argv


def run_ablation_job(job: AblationJob, python_cmd: str = "python3") -> dict[str, Any]:
    """Ejecuta un job de ablacion lanzando main.py como subproceso.

    Args:
        job: AblationJob con la configuracion del experimento.
        python_cmd: Comando de Python a usar (default: "python3").

    Returns:
        dict con: name, success (bool), returncode (int),
        stdout (str), stderr (str).
    """
    argv = job_to_argv(job)
    # Reemplazar el python3 hardcodeado por el python_cmd recibido
    if argv and argv[0] == "python3":
        argv[0] = python_cmd

    logger.info("Ejecutando job '%s': %s", job.name, " ".join(argv))

    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=3600,  # timeout de 1 hora por job
        )
        success = result.returncode == 0
        if not success:
            logger.warning(
                "Job '%s' fallo con returncode=%d. stderr: %s",
                job.name,
                result.returncode,
                result.stderr[:500],
            )
        return {
            "name": job.name,
            "success": success,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        logger.error("Job '%s' supero el timeout: %s", job.name, exc)
        return {
            "name": job.name,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"TimeoutExpired: {exc}",
        }
    except OSError as exc:
        logger.error("Error de sistema al ejecutar job '%s': %s", job.name, exc)
        return {
            "name": job.name,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"OSError: {exc}",
        }


def save_ablation_summary(
    summary_path: Path,
    jobs: list[AblationJob],
    results: list[dict[str, Any]],
) -> None:
    """Guarda resumen de la ablacion como JSON.

    Formato del JSON:
    {
        "total_jobs": N,
        "successful": M,
        "failed": K,
        "jobs": [{"name": ..., "variant": ..., "success": ..., "returncode": ...}, ...]
    }

    Args:
        summary_path: Ruta donde escribir el JSON.
        jobs: Lista de AblationJob ejecutados.
        results: Lista de dicts retornados por run_ablation_job.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Construir mapa nombre -> resultado para lookup rapido
    results_by_name: dict[str, dict[str, Any]] = {r["name"]: r for r in results}

    job_summaries: list[dict[str, Any]] = []
    for job in jobs:
        res = results_by_name.get(job.name, {})
        job_summaries.append(
            {
                "name": job.name,
                "variant": job.variant,
                "csv_path": job.csv_path,
                "targets": job.targets,
                "success": res.get("success", False),
                "returncode": res.get("returncode", -1),
            }
        )

    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful

    summary: dict[str, Any] = {
        "total_jobs": len(jobs),
        "successful": successful,
        "failed": failed,
        "jobs": job_summaries,
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Resumen de ablacion guardado en: %s (%d/%d exitosos)",
        summary_path,
        successful,
        len(results),
    )


def main() -> None:
    """CLI para ejecutar una matriz de ablaciones desde la linea de comandos."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ejecutar matriz de ablaciones FEDformer"
    )
    parser.add_argument("--csv", required=True, help="Ruta al CSV del dataset")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["Close"],
        help="Columnas target",
    )
    parser.add_argument(
        "--variants-json",
        required=True,
        help="Archivo JSON con lista de variantes",
    )
    parser.add_argument(
        "--base-args-json",
        default=None,
        help="Archivo JSON con args base comunes",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directorio de resultados",
    )
    parser.add_argument(
        "--summary-path",
        default="results/ablation_summary.json",
        help="Ruta para guardar el resumen JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar los comandos sin ejecutarlos",
    )
    args = parser.parse_args()

    # Cargar variantes
    variants_path = Path(args.variants_json)
    if not variants_path.exists():
        raise FileNotFoundError(f"Archivo de variantes no encontrado: {variants_path}")
    with open(variants_path, encoding="utf-8") as fh:
        variants: list[dict[str, Any]] = json.load(fh)

    # Cargar base_args si se proveyeron (soporta JSON inline o ruta a archivo)
    base_args: dict[str, Any] | None = None
    if args.base_args_json is not None:
        raw = args.base_args_json.strip()
        if raw.startswith("{"):
            # Interpretar como JSON inline
            base_args = json.loads(raw)
        else:
            base_path = Path(raw)
            if not base_path.exists():
                raise FileNotFoundError(f"Archivo base-args no encontrado: {base_path}")
            with open(base_path, encoding="utf-8") as fh:
                base_args = json.load(fh)

    # Construir jobs
    jobs = build_ablation_jobs(
        csv_path=args.csv,
        targets=args.targets,
        variants=variants,
        base_args=base_args,
        results_dir=args.results_dir,
    )

    logger.info("Ablacion: %d jobs construidos", len(jobs))

    if args.dry_run:
        # Solo mostrar los comandos sin ejecutar
        for job in jobs:
            argv = job_to_argv(job)
            print(f"[DRY-RUN] {job.name}: {' '.join(argv)}")
        return

    # Ejecutar jobs secuencialmente
    results: list[dict[str, Any]] = []
    for job in jobs:
        result = run_ablation_job(job)
        results.append(result)

    # Guardar resumen
    summary_path = Path(args.summary_path)
    save_ablation_summary(summary_path, jobs, results)

    # Reporte final
    successful = sum(1 for r in results if r.get("success", False))
    logger.info(
        "Ablacion completada: %d/%d jobs exitosos",
        successful,
        len(results),
    )


if __name__ == "__main__":
    main()
