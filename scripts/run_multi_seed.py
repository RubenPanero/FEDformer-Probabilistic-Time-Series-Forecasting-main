#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para ejecutar experimentos multi-seed y evaluar robustez."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_single_seed(
    csv_path: str,
    targets: list[str],
    seed: int,
    extra_args: list[str] | None = None,
    results_dir: str = "results",
    python_cmd: str = "python3",
) -> dict[str, Any]:
    """Ejecuta main.py con un seed específico como subproceso.

    Siempre pasa: --csv, --targets, --seed, --save-results, --no-show

    Args:
        csv_path: Ruta al CSV del dataset.
        targets: Lista de columnas objetivo.
        seed: Semilla aleatoria para este run.
        extra_args: Argumentos adicionales para main.py (ej. --seq-len 96).
        results_dir: Directorio de salida de resultados (no usado directamente,
            main.py usa "results/" internamente).
        python_cmd: Comando Python a usar (por defecto "python3").

    Returns:
        Dict con: seed, success (bool), returncode (int), stdout (str), stderr (str)

    No lanza excepción si el subproceso falla.
    """
    cmd = [
        python_cmd,
        "main.py",
        "--csv",
        csv_path,
        "--targets",
        ",".join(targets),
        "--seed",
        str(seed),
        "--save-results",
        "--no-show",
    ]

    if extra_args:
        cmd.extend(extra_args)

    logger.info("Ejecutando seed %d: %s", seed, " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        success = proc.returncode == 0
        if not success:
            logger.warning(
                "Seed %d falló (returncode=%d): %s",
                seed,
                proc.returncode,
                proc.stderr[-500:] if proc.stderr else "",
            )
        return {
            "seed": seed,
            "success": success,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Error inesperado ejecutando seed %d: %s", seed, exc)
        return {
            "seed": seed,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }


def run_multi_seed_experiment(
    csv_path: str,
    targets: list[str],
    seeds: list[int],
    extra_args: list[str] | None = None,
    results_dir: str = "results",
    python_cmd: str = "python3",
) -> list[dict[str, Any]]:
    """Ejecuta el experimento para cada seed en seeds.

    Secuencial (no paralelo) para evitar contención de GPU.

    Args:
        csv_path: Ruta al CSV del dataset.
        targets: Lista de columnas objetivo.
        seeds: Lista de seeds a ejecutar.
        extra_args: Argumentos adicionales para main.py.
        results_dir: Directorio base de resultados.
        python_cmd: Comando Python a usar.

    Returns:
        Lista de resultados, uno por seed.
    """
    results: list[dict[str, Any]] = []
    n_seeds = len(seeds)

    for i, seed in enumerate(seeds):
        logger.info("Ejecutando seed %d/%d (seed=%d)...", i + 1, n_seeds, seed)
        result = run_single_seed(
            csv_path=csv_path,
            targets=targets,
            seed=seed,
            extra_args=extra_args,
            results_dir=results_dir,
            python_cmd=python_cmd,
        )
        results.append(result)

        status = "OK" if result["success"] else "FALLO"
        logger.info("Seed %d: %s (returncode=%d)", seed, status, result["returncode"])

    n_ok = sum(1 for r in results if r["success"])
    logger.info("Multi-seed completado: %d/%d seeds exitosos.", n_ok, n_seeds)
    return results


def main() -> None:
    """Punto de entrada CLI del runner multi-seed."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ejecutar experimento FEDformer con múltiples seeds"
    )
    parser.add_argument("--csv", required=True, help="Ruta al CSV del dataset")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["Close"],
        help="Columnas objetivo (default: Close)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 7],
        help="Lista de seeds a usar (default: 42 123 7)",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Args adicionales para main.py (ej: --seq-len 96 --pred-len 20)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directorio base para resultados (default: results)",
    )
    parser.add_argument(
        "--summary-path",
        default="results/multi_seed_summary.json",
        help="Ruta del JSON de resumen multi-seed (default: results/multi_seed_summary.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar los comandos sin ejecutarlos",
    )
    args = parser.parse_args()

    if args.dry_run:
        # Modo dry-run: solo imprimir los comandos que se ejecutarían
        logger.info("=== DRY RUN — comandos que se ejecutarían ===")
        for seed in args.seeds:
            cmd_parts = [
                "python3",
                "main.py",
                "--csv",
                args.csv,
                "--targets",
                ",".join(args.targets),
                "--seed",
                str(seed),
                "--save-results",
                "--no-show",
            ]
            if args.extra_args:
                cmd_parts.extend(args.extra_args)
            logger.info("  %s", " ".join(cmd_parts))
        return

    # Ejecutar todos los seeds
    results = run_multi_seed_experiment(
        csv_path=args.csv,
        targets=args.targets,
        seeds=args.seeds,
        extra_args=args.extra_args if args.extra_args else None,
        results_dir=args.results_dir,
    )

    # Construir resumen
    summary = {
        "csv": args.csv,
        "targets": args.targets,
        "seeds": args.seeds,
        "results": results,
        "n_success": sum(1 for r in results if r["success"]),
        "n_total": len(results),
    }

    # Guardar resumen JSON
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        # Serializar excluyendo stdout/stderr para mantener el JSON compacto
        compact = {k: v for k, v in summary.items() if k != "results"}
        compact["results"] = [
            {
                "seed": r["seed"],
                "success": r["success"],
                "returncode": r["returncode"],
            }
            for r in results
        ]
        json.dump(compact, fh, indent=2)

    logger.info("Resumen multi-seed guardado en: %s", summary_path)
    logger.info("Total: %d/%d seeds exitosos", summary["n_success"], summary["n_total"])


if __name__ == "__main__":
    main()
