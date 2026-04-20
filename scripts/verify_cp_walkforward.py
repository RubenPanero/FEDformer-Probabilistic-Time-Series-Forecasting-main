"""Verifica empíricamente cp_wf_coverage_80 con la ruta --cp-walkforward."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Define la CLI del verificador de Conformal Prediction walk-forward."""
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta un experimento con --cp-walkforward y resume las métricas "
            "cp_wf_* del run_manifest más reciente."
        )
    )
    parser.add_argument(
        "--csv",
        default="data/NVDA_features.csv",
        help="CSV a usar para la verificación (default: data/NVDA_features.csv)",
    )
    parser.add_argument(
        "--targets",
        default="Close",
        help="Lista separada por comas de targets para main.py (default: Close)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=96,
        help="seq_len para main.py (default: 96)",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=20,
        help="pred_len para main.py (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="batch_size para main.py (default: 64)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=4,
        help="Número de splits walk-forward (default: 4)",
    )
    parser.add_argument(
        "--return-transform",
        default="log_return",
        choices=["none", "log_return", "simple_return"],
        help="Transformación de retornos para main.py (default: log_return)",
    )
    parser.add_argument(
        "--metric-space",
        default="returns",
        choices=["returns", "prices"],
        help="Espacio de métricas para main.py (default: returns)",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=0.5,
        help="Gradient clipping para main.py (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para main.py (default: 42)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_DIR / "results",
        help="Directorio donde buscar run_manifest_*.json (default: results/)",
    )
    parser.add_argument(
        "--objective-threshold",
        type=float,
        default=0.80,
        help="Umbral objetivo para cp_wf_coverage_80 (default: 0.80)",
    )
    parser.add_argument(
        "--python-cmd",
        default=sys.executable,
        help="Intérprete Python a usar para lanzar main.py (default: actual)",
    )
    parser.add_argument(
        "--compile-mode",
        default=None,
        help=(
            "Valor opcional para --compile-mode al lanzar main.py. "
            "Si se omite, no se pasa el flag."
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="No lanza entrenamiento; solo inspecciona el último run_manifest.",
    )
    parser.add_argument(
        "--skip-gpu-info",
        action="store_true",
        help="No imprime información de GPU/CPU al inicio.",
    )
    return parser.parse_args()


def check_flag_available() -> None:
    """Falla con mensaje claro si --cp-walkforward no está disponible en main.py."""
    src = (PROJECT_DIR / "main.py").read_text(encoding="utf-8")
    if "cp_walkforward" not in src and "cp-walkforward" not in src:
        raise RuntimeError("--cp-walkforward no encontrado en main.py")


def print_gpu_info() -> None:
    """Muestra información básica del dispositivo disponible."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU : {torch.cuda.get_device_name(0)}")
            print(f"SMs : {props.multi_processor_count}")
            print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
            if props.multi_processor_count < 40:
                print(
                    "NOTA: <40 SMs -> torch.compile suele degradar a eager "
                    "(comportamiento esperado)"
                )
        else:
            print("ADVERTENCIA: CUDA no disponible -> ejecución en CPU")
    except ImportError:
        print("ADVERTENCIA: torch no importable")


def build_training_cmd(args: argparse.Namespace) -> list[str]:
    """Construye el comando de entrenamiento para main.py."""
    cmd = [
        args.python_cmd,
        "main.py",
        "--csv",
        args.csv,
        "--targets",
        args.targets,
        "--seq-len",
        str(args.seq_len),
        "--pred-len",
        str(args.pred_len),
        "--batch-size",
        str(args.batch_size),
        "--splits",
        str(args.splits),
        "--return-transform",
        args.return_transform,
        "--metric-space",
        args.metric_space,
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--seed",
        str(args.seed),
        "--cp-walkforward",
        "--save-results",
        "--no-show",
    ]
    if args.compile_mode is not None:
        cmd.extend(["--compile-mode", args.compile_mode])
    return cmd


def run_training(args: argparse.Namespace) -> int:
    """Lanza el run de verificación con --cp-walkforward."""
    cmd = build_training_cmd(args)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    print("Comando:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, check=False)
    if result.returncode != 0:
        print(f"ADVERTENCIA: main.py terminó con código {result.returncode}")
    return result.returncode


def load_latest_manifest(results_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    """Carga el run_manifest más reciente del directorio indicado."""
    manifests = [
        (path, path.stat()) for path in results_dir.glob("run_manifest_*.json")
    ]
    if not manifests:
        return None
    manifest_path, _ = max(manifests, key=lambda pair: pair[1].st_mtime_ns)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path, manifest


def report_results(results_dir: Path, objective_threshold: float) -> int:
    """Extrae y evalúa cp_wf_coverage_80 del run_manifest más reciente."""
    loaded = load_latest_manifest(results_dir)
    if loaded is None:
        print(f"No se encontró run_manifest_*.json en {results_dir}")
        return 1

    manifest_path, manifest = loaded
    metrics: dict[str, Any] = manifest.get("metrics", manifest)
    cp_keys = {k: v for k, v in metrics.items() if "cp_wf" in str(k)}

    print("\n=== Reporte ===")
    print(f"Manifest: {manifest_path}")

    if not cp_keys:
        print("No se encontraron métricas cp_wf_ en el manifiesto.")
        print("Claves raíz:", list(manifest.keys()))
        print("Claves metrics:", list(metrics.keys())[:20])
        return 1

    print("\n=== Métricas CP Enfoque 1 (walk-forward) ===")
    for key, value in cp_keys.items():
        print(f"  {key}: {value}")

    cov = cp_keys.get("cp_wf_coverage_80")
    folds_cal = cp_keys.get("cp_wf_folds_calibrated", "?")
    if cov is None:
        print("\nNo existe cp_wf_coverage_80 en el manifiesto.")
        return 1

    ok = float(cov) >= objective_threshold
    status = (
        "OBJETIVO CUMPLIDO"
        if ok
        else f"BAJO OBJETIVO (target={objective_threshold:.2f})"
    )
    print(f"\n  cp_wf_coverage_80      = {float(cov):.4f}  [{status}]")
    print(f"  cp_wf_folds_calibrated = {folds_cal}")
    return 0 if ok else 2


def main() -> int:
    """Punto de entrada CLI."""
    args = parse_args()
    check_flag_available()

    if not args.skip_gpu_info:
        print("=== GPU ===")
        print_gpu_info()

    if not args.report_only:
        print("\n=== Run --cp-walkforward ===")
        run_training(args)

    return report_results(args.results_dir, args.objective_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
