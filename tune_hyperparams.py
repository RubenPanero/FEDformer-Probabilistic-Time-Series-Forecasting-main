# -*- coding: utf-8 -*-
"""
Búsqueda de hiperparámetros con Optuna para un ticker específico.

Cada trial ejecuta main.py como subproceso independiente (seed 42 limpio,
sin contaminación cross-ticker) y extrae el Sharpe del CSV de portafolio.

El estudio es persistente (SQLite): puede interrumpirse y retomarse con
los mismos flags. Los trials fallidos/expirados retornan Sharpe=-1.0.
Los trials donde seq_len < pred_len*3 se podan sin entrenar.

Uso:
    # Búsqueda básica (20 trials in-memory)
    python3 tune_hyperparams.py --csv data/MSFT_features.csv

    # Con persistencia SQLite (reanudable)
    python3 tune_hyperparams.py --csv data/MSFT_features.csv \\
        --n-trials 20 --storage-path optuna_studies/msft.db

    # Buscar Y registrar el mejor resultado en model_registry
    python3 tune_hyperparams.py --csv data/NVDA_features.csv \\
        --n-trials 16 --best-save-canonical

    # Score compuesto (Sharpe + métricas probabilísticas)
    python3 tune_hyperparams.py --csv data/NVDA_features.csv \\
        --n-trials 20 --study-objective composite

    # Descargar 8 tickers adicionales y lanzar búsqueda para los 12
    python3 tune_hyperparams.py --download-extra-tickers
"""

import argparse
import itertools
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Espacio de búsqueda
# ---------------------------------------------------------------------------

SEQ_LENS = [48, 64, 96, 128]
PRED_LENS = [4, 6, 8, 10, 20]  # todos pares (requisito del affine coupling)
BATCH_SIZES = [32, 64]
CLIP_NORMS = [0.3, 0.5]
E_LAYERS = [1, 2, 3]
D_LAYERS = [1, 2]
N_FLOW_LAYERS = [2, 4, 6]
FLOW_HIDDEN_DIMS = [32, 64, 128]
LABEL_LENS = [24, 48, 96]
DROPOUTS = [0.05, 0.1, 0.2]


@dataclass(frozen=True)
class SearchParameter:
    """Representa un hiperparámetro tuneable del estudio de Optuna."""

    name: str
    choices: tuple[Any, ...]
    cli_flag: str


SEARCH_PARAMETERS: tuple[SearchParameter, ...] = (
    SearchParameter("seq_len", tuple(SEQ_LENS), "--seq-len"),
    SearchParameter("pred_len", tuple(PRED_LENS), "--pred-len"),
    SearchParameter("batch_size", tuple(BATCH_SIZES), "--batch-size"),
    SearchParameter("gradient_clip_norm", tuple(CLIP_NORMS), "--gradient-clip-norm"),
    SearchParameter("e_layers", tuple(E_LAYERS), "--e-layers"),
    SearchParameter("d_layers", tuple(D_LAYERS), "--d-layers"),
    SearchParameter("n_flow_layers", tuple(N_FLOW_LAYERS), "--n-flow-layers"),
    SearchParameter("flow_hidden_dim", tuple(FLOW_HIDDEN_DIMS), "--flow-hidden-dim"),
    SearchParameter("label_len", tuple(LABEL_LENS), "--label-len"),
    SearchParameter("dropout", tuple(DROPOUTS), "--dropout"),
)

CANONICAL_TRIAL_PARAMS: dict[str, Any] = {
    "seq_len": 96,
    "pred_len": 20,
    "batch_size": 64,
    "gradient_clip_norm": 0.5,
    "e_layers": 2,
    "d_layers": 1,
    "n_flow_layers": 4,
    "flow_hidden_dim": 64,
    "label_len": 48,
    "dropout": 0.1,
}

# Tickers adicionales para alcanzar 12 activos — cobertura sectorial diversa:
#   Semiconductores:  AMD, INTC  (peers directos de NVDA)
#   Big tech:         AMZN, META (peers de MSFT/GOOGL/AAPL)
#   Alta volatilidad: TSLA, NFLX (NF se beneficia de colas gruesas)
#   Financiero:       JPM        (dinámica contra-cíclica al tech)
#   Enterprise SaaS:  CRM        (correlación tech pero ciclo propio)
EXTRA_TICKERS = ["AMD", "INTC", "AMZN", "META", "TSLA", "NFLX", "JPM", "CRM"]


# ---------------------------------------------------------------------------
# Parsing de resultados
# ---------------------------------------------------------------------------


def _current_time() -> float:
    """Aísla la fuente de tiempo para evitar side effects al parchear tests."""
    return time.time()


def _find_recent_result_file(
    results_dir: Path,
    prefix: str,
    ticker_stem: str,
    after_ts: float,
) -> Path | None:
    """Retorna el CSV de resultados más reciente para el trial actual.

    Acepta tanto el formato histórico con ticker en el nombre como el formato
    actual de main.py, que solo incluye el timestamp.
    """
    patterns = [
        f"{prefix}_*_{ticker_stem}.csv",
        f"{prefix}_*.csv",
    ]
    candidates: dict[Path, Path] = {}
    for pattern in patterns:
        for path in results_dir.glob(pattern):
            candidates[path.resolve()] = path

    recent = sorted(
        (path for path in candidates.values() if path.stat().st_mtime >= after_ts),
        key=lambda path: path.stat().st_mtime,
    )
    return recent[-1] if recent else None


def _parse_portfolio_csv(results_dir: Path, ticker_stem: str, after_ts: float) -> dict:
    """Retorna métricas del CSV de portafolio más reciente generado después de after_ts.

    Args:
        results_dir: Directorio donde se guardan los CSVs de resultados.
        ticker_stem: Stem del archivo CSV del ticker (ej. "MSFT_features").
        after_ts: Timestamp Unix mínimo del archivo (inicio del trial).

    Returns:
        Dict con sharpe, sortino, max_drawdown, volatility; o valores centinela −1.0.
    """
    recent = _find_recent_result_file(
        results_dir,
        prefix="portfolio_metrics",
        ticker_stem=ticker_stem,
        after_ts=after_ts,
    )
    if recent is None:
        logger.warning(
            "No se encontró CSV de portafolio para %s tras el trial.", ticker_stem
        )
        return {"sharpe": -1.0, "sortino": -1.0, "max_drawdown": 0.0, "volatility": 0.0}

    df = pd.read_csv(recent)
    metrics = df.set_index("metric")["value"].to_dict()
    return {
        "sharpe": float(metrics.get("sharpe_ratio", -1.0)),
        "sortino": float(metrics.get("sortino_ratio", -1.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "volatility": float(metrics.get("volatility", 0.0)),
    }


def _parse_risk_csv(results_dir: Path, ticker_stem: str, after_ts: float) -> float:
    """Retorna el VaR 95% medio del CSV de riesgo más reciente."""
    recent = _find_recent_result_file(
        results_dir,
        prefix="risk_metrics",
        ticker_stem=ticker_stem,
        after_ts=after_ts,
    )
    if recent is None:
        return 1.0  # valor centinela alto = penalización
    df = pd.read_csv(recent)
    return float(df["var_95"].mean()) if "var_95" in df.columns else 1.0


def _parse_probabilistic_csv(
    results_dir: Path,
    ticker_stem: str,
    after_ts: str,
) -> dict[str, float]:
    """Lee el archivo probabilistic_metrics más reciente para el ticker dado.

    Busca archivos con nombre probabilistic_metrics_{ts}_{ticker}.csv generados
    después de after_ts. Hace pivot del formato largo (fold, metric, value,
    space, aggregation) para extraer las métricas clave promediadas por fold.

    Args:
        results_dir: Directorio donde buscar los CSVs.
        ticker_stem: Nombre del ticker sin extensión (ej. "NVDA_features").
        after_ts: Timestamp mínimo en formato "%Y%m%d_%H%M%S".

    Returns:
        Dict con métricas clave: pinball_p50, coverage_80, interval_width_80, crps.
        Si no encuentra el archivo o faltan columnas, retorna dict vacío.
    """
    # Convertir string de timestamp a float Unix para comparar con st_mtime
    try:
        ts_struct = time.strptime(after_ts, "%Y%m%d_%H%M%S")
        after_ts_float = time.mktime(ts_struct)
    except (ValueError, OverflowError):
        logger.warning(
            "Formato de after_ts inválido: '%s'. Se esperaba %%Y%%m%%d_%%H%%M%%S.",
            after_ts,
        )
        return {}

    recent = _find_recent_result_file(
        results_dir,
        prefix="probabilistic_metrics",
        ticker_stem=ticker_stem,
        after_ts=after_ts_float,
    )
    if recent is None:
        return {}

    try:
        df = pd.read_csv(recent)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning("Error al leer CSV probabilístico %s: %s", recent, exc)
        return {}

    # Verificar columnas requeridas del formato largo
    required_cols = {"metric", "value"}
    if not required_cols.issubset(df.columns):
        logger.warning(
            "CSV probabilístico %s no tiene columnas esperadas. Encontradas: %s",
            recent,
            list(df.columns),
        )
        return {}

    # Agregar por métrica (media sobre folds) — columna aggregation="mean" por diseño
    agg = df.groupby("metric")["value"].mean()
    metricas_clave = ["pinball_p50", "coverage_80", "interval_width_80", "crps"]
    result: dict[str, float] = {}
    for clave in metricas_clave:
        if clave in agg.index:
            val = float(agg[clave])
            # Nunca propagar NaN hacia el score compuesto
            result[clave] = val if math.isfinite(val) else 0.0

    return result


def _compose_trial_score(
    portfolio_metrics: dict,
    risk_metrics: dict,
    prob_metrics: dict,
    mode: str = "sharpe",
) -> float:
    """Compone el score final de un trial de Optuna.

    Modos disponibles:
    - "sharpe": usa solo Sharpe ratio (compatibilidad hacia atrás).
    - "composite": 0.5 * sharpe + 0.3 * (1 - pinball_p50_norm) + 0.2 * coverage_score
      donde pinball_p50_norm = pinball_p50 / (|mean_return| + 1e-8)  [normalizado]
      y coverage_score = max(0, 1 - |coverage_80 - 0.80| / 0.80)    [penaliza desviación]
    - "multi-objective": alias de "composite" por ahora (Optuna multi-objetivo requiere
      refactor mayor de create_study y del retorno de la función objetivo).
      TODO: implementar con optuna.create_study(directions=[...]) en una épica futura.

    Si prob_metrics está vacío y mode != "sharpe", emite warning y usa solo Sharpe.
    Nunca retorna NaN — usa 0.0 como fallback para métricas ausentes.

    Args:
        portfolio_metrics: Dict con sharpe, sortino, max_drawdown, volatility.
        risk_metrics: Dict con var_95 (float) u otras métricas de riesgo.
        prob_metrics: Dict con pinball_p50, coverage_80, interval_width_80, crps.
        mode: Modo de composición del score.

    Returns:
        Score escalar para Optuna (maximize).
    """
    sharpe = float(portfolio_metrics.get("sharpe", 0.0))
    if not math.isfinite(sharpe):
        sharpe = 0.0

    if mode == "sharpe":
        return sharpe

    # Modos composite y multi-objective — requieren métricas probabilísticas
    if mode == "multi-objective":
        # TODO: implementar con optuna multi-objetivo (directions=['maximize','maximize'])
        # cuando se refactorice create_study. Por ahora usa la misma fórmula que composite.
        pass

    if not prob_metrics:
        logger.warning(
            "mode='%s' requiere métricas probabilísticas, pero prob_metrics está vacío. "
            "Usando Sharpe puro como fallback.",
            mode,
        )
        return sharpe

    # Componente 1: Sharpe (peso 0.5)
    sharpe_component = sharpe

    # Componente 2: Pinball P50 normalizado (peso 0.3)
    # Menor pinball = mejor calibración puntual → (1 - norm) maximizable
    pinball_p50 = float(prob_metrics.get("pinball_p50", 0.0))
    if not math.isfinite(pinball_p50):
        pinball_p50 = 0.0
    # Normalizar por magnitud del retorno medio para escalar adecuadamente
    mean_return_proxy = (
        abs(sharpe) * float(portfolio_metrics.get("volatility", 0.01)) + 1e-8
    )
    pinball_norm = min(pinball_p50 / mean_return_proxy, 1.0)
    calibration_component = 1.0 - pinball_norm

    # Componente 3: Coverage score (peso 0.2)
    # Penaliza la desviación respecto al 80% de cobertura nominal
    coverage_80 = float(prob_metrics.get("coverage_80", 0.0))
    if not math.isfinite(coverage_80):
        coverage_80 = 0.0
    coverage_score = max(0.0, 1.0 - abs(coverage_80 - 0.80) / 0.80)

    score = 0.5 * sharpe_component + 0.3 * calibration_component + 0.2 * coverage_score
    return score if math.isfinite(score) else 0.0


def _suggest_search_parameters(trial: optuna.Trial) -> dict[str, Any]:
    """Sugiere el espacio de búsqueda completo para un trial."""
    params: dict[str, Any] = {}
    for parameter in SEARCH_PARAMETERS:
        params[parameter.name] = trial.suggest_categorical(
            parameter.name, list(parameter.choices)
        )
    return params


def _extend_cmd_with_search_parameters(
    cmd: list[str],
    params: dict[str, Any],
) -> None:
    """Añade al comando los flags del espacio de búsqueda centralizado."""
    params = {**CANONICAL_TRIAL_PARAMS, **params}
    for parameter in SEARCH_PARAMETERS:
        cmd.extend([parameter.cli_flag, str(params[parameter.name])])


def _count_search_space_combinations() -> int:
    """Cuenta solo combinaciones válidas tras aplicar las restricciones estructurales."""
    total = 0
    for seq_len, pred_len, label_len in itertools.product(
        SEQ_LENS, PRED_LENS, LABEL_LENS
    ):
        if seq_len < pred_len * 3 or label_len > seq_len:
            continue
        total += 1

    for parameter in SEARCH_PARAMETERS:
        if parameter.name in {"seq_len", "pred_len", "label_len"}:
            continue
        total *= len(parameter.choices)
    return total


def _build_trial_env() -> dict[str, str]:
    """Construye un entorno seguro para trials y reruns lanzados por Optuna."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(Path("/tmp") / "matplotlib")
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    return env


# ---------------------------------------------------------------------------
# Función objetivo
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.Trial,
    csv_path: str,
    n_splits: int,
    results_dir: Path,
    study_objective: str = "sharpe",
    seed: int = 7,
    compile_mode: str = "",
) -> float:
    """Entrena Flow_FEDformer con los hiperparámetros sugeridos y retorna el score.

    Penalizaciones sobre el score final:
        - Si VaR_95 > 0.08: score × 0.5 (riesgo excesivo)
        - Si Sortino < 0: score − 0.3 (sin señal asimétrica útil)

    Args:
        trial: Trial de Optuna.
        csv_path: Ruta al CSV del ticker.
        n_splits: Número de folds walk-forward.
        results_dir: Directorio de resultados para parsear CSVs.
        study_objective: Modo de score ("sharpe", "composite", "multi-objective").
        seed: Seed para reproducibilidad (debe coincidir con baselines).
        compile_mode: Modo de torch.compile para el subprocess ("" = desactivado).

    Returns:
        Score penalizado, o -1.0 si el trial falló.
    """
    params = _suggest_search_parameters(trial)
    seq_len = params["seq_len"]
    pred_len = params["pred_len"]
    batch_size = params["batch_size"]
    clip_norm = params["gradient_clip_norm"]
    e_layers = params["e_layers"]
    d_layers = params["d_layers"]
    n_flow_layers = params["n_flow_layers"]
    flow_hidden_dim = params["flow_hidden_dim"]
    label_len = params["label_len"]
    dropout = params["dropout"]

    # Restricción estructural: el contexto debe ser al menos 3× la predicción
    # para que el encoder tenga suficiente señal temporal
    if seq_len < pred_len * 3:
        raise optuna.TrialPruned()

    # Restricción estructural nueva: label_len no puede superar seq_len.
    if label_len > seq_len:
        raise optuna.TrialPruned()

    ticker_stem = Path(csv_path).stem

    cmd = [
        sys.executable,
        "main.py",
        "--csv",
        csv_path,
        "--targets",
        "Close",
        "--splits",
        str(n_splits),
        "--return-transform",
        "log_return",
        "--metric-space",
        "returns",
        "--save-results",
        "--no-show",
        "--seed",
        str(seed),
        # Sin --save-canonical: los trials no tocan el model_registry
    ]
    _extend_cmd_with_search_parameters(cmd, params)

    # Siempre pasar --compile-mode explícitamente al subprocess.
    # CRÍTICO: config.py default es "max-autotune" — omitir el flag activa compilación
    # completa en todos los trials (causó timeouts masivos en T4/Kaggle, sesión 10).
    # "" desactiva compilación vía guarda truthy en trainer.py (`if compile_mode and ...`).
    # TODO: reemplazar "" por sentinel explícito "none" cuando se refactorice
    # _effective_compile_mode para reconocer "none" → "" (evita pasar string vacío como arg CLI).
    cmd.extend(["--compile-mode", compile_mode])

    env = _build_trial_env()

    # Registrar timestamp de inicio como string para parsear CSVs probabilísticos
    ts_before = _current_time()
    ts_before_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts_before))
    logger.info(
        (
            "Trial %d | seq=%d pred=%d batch=%d clip=%.1f e_layers=%d "
            "d_layers=%d n_flow_layers=%d flow_hidden_dim=%d label_len=%d "
            "dropout=%.2f | objective=%s"
        ),
        trial.number,
        seq_len,
        pred_len,
        batch_size,
        clip_norm,
        e_layers,
        d_layers,
        n_flow_layers,
        flow_hidden_dim,
        label_len,
        dropout,
        study_objective,
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=900,  # 15 min máximo por trial
        )
        if proc.returncode != 0:
            logger.warning(
                "Trial %d falló (exit %d). stderr: %s",
                trial.number,
                proc.returncode,
                proc.stderr[-300:],
            )
            return -1.0
    except subprocess.TimeoutExpired:
        logger.warning("Trial %d expiró (>15 min).", trial.number)
        return -1.0

    portfolio = _parse_portfolio_csv(results_dir, ticker_stem, ts_before)
    var_95 = _parse_risk_csv(results_dir, ticker_stem, ts_before)

    # Parsear métricas probabilísticas si están disponibles (--save-results activo en cmd)
    prob_metrics = _parse_probabilistic_csv(results_dir, ticker_stem, ts_before_str)

    sharpe = portfolio["sharpe"]
    sortino = portfolio["sortino"]

    logger.info(
        "Trial %d → Sharpe=%.4f Sortino=%.4f VaR=%.4f pinball_p50=%s coverage_80=%s",
        trial.number,
        sharpe,
        sortino,
        var_95,
        f"{prob_metrics.get('pinball_p50', float('nan')):.4f}"
        if prob_metrics
        else "n/a",
        f"{prob_metrics.get('coverage_80', float('nan')):.4f}"
        if prob_metrics
        else "n/a",
    )

    # Registrar métricas auxiliares para análisis post-hoc en el dashboard
    trial.set_user_attr("sortino", sortino)
    trial.set_user_attr("var_95", var_95)
    trial.set_user_attr("max_drawdown", portfolio["max_drawdown"])
    trial.set_user_attr("pinball_p50", prob_metrics.get("pinball_p50", float("nan")))
    trial.set_user_attr("coverage_80", prob_metrics.get("coverage_80", float("nan")))
    trial.set_user_attr(
        "interval_width_80",
        prob_metrics.get("interval_width_80", float("nan")),
    )

    # Componer score según el objetivo seleccionado
    score = _compose_trial_score(
        portfolio_metrics=portfolio,
        risk_metrics={"var_95": var_95},
        prob_metrics=prob_metrics,
        mode=study_objective,
    )
    trial.set_user_attr("composite_score", score)

    # Penalizaciones de riesgo (aplicadas al score final)
    if var_95 > 0.08:
        score *= 0.5
    if sortino < 0:
        score -= 0.3

    return score


# ---------------------------------------------------------------------------
# Descarga de tickers adicionales
# ---------------------------------------------------------------------------


def download_extra_tickers() -> None:
    """Descarga los 8 tickers adicionales para alcanzar los 12 activos.

    Usa financial_dataset_builder.py con --use_mock (yfinance real, 7 años).
    Omite los que ya existen en data/.
    """
    for ticker in EXTRA_TICKERS:
        dest = Path("data") / f"{ticker}_features.csv"
        if dest.exists():
            logger.info(
                "%-6s ya existe (%d filas) — omitiendo.", ticker, len(pd.read_csv(dest))
            )
            continue
        logger.info("Descargando %s...", ticker)
        proc = subprocess.run(
            [
                sys.executable,
                "data/financial_dataset_builder.py",
                "--symbol",
                ticker,
                "--use_mock",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "."},
        )
        if proc.returncode == 0:
            if not dest.exists():
                logger.error("%-6s finalizó sin generar %s", ticker, dest.as_posix())
                continue
            df = pd.read_csv(dest)
            logger.info("%-6s → %d filas × %d cols", ticker, len(df), df.shape[1])
        else:
            logger.error("%-6s falló: %s", ticker, proc.stderr[-200:])


# ---------------------------------------------------------------------------
# Entrenamiento final con los mejores parámetros
# ---------------------------------------------------------------------------


def _run_best_params(
    csv_path: str,
    best_params: dict,
    n_splits: int,
    save_canonical: bool,
    seed: int = 7,
    compile_mode: str = "",
) -> None:
    """Re-entrena con los mejores hiperparámetros encontrados por Optuna.

    Args:
        csv_path: Ruta al CSV del ticker.
        best_params: Dict con seq_len, pred_len, batch_size, gradient_clip_norm.
        n_splits: Número de folds walk-forward.
        save_canonical: Si True, registra en model_registry con --save-canonical.
        seed: Seed para reproducibilidad.
        compile_mode: Modo de torch.compile ("" = desactivado).
    """
    cmd = [
        sys.executable,
        "main.py",
        "--csv",
        csv_path,
        "--targets",
        "Close",
        "--splits",
        str(n_splits),
        "--return-transform",
        "log_return",
        "--metric-space",
        "returns",
        "--save-results",
        "--no-show",
        "--seed",
        str(seed),
    ]
    _extend_cmd_with_search_parameters(cmd, best_params)
    if save_canonical:
        cmd.append("--save-canonical")
    # Ver comentario en objective(): siempre pasar explícitamente para neutralizar
    # el default "max-autotune" de config.py.
    cmd.extend(["--compile-mode", compile_mode])

    env = _build_trial_env()
    logger.info("Ejecutando run final con mejores parámetros (seed=%d)...", seed)
    subprocess.run(cmd, env=env)


def _build_completed_trials_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Construye el DataFrame de trials completados para reporting."""
    df_trials = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "user_attrs")
    )
    if "state" not in df_trials.columns:
        return pd.DataFrame(columns=df_trials.columns)
    return df_trials[df_trials["state"] == "COMPLETE"].sort_values(
        "value", ascending=False
    )


# ---------------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------------


def main() -> None:
    """Punto de entrada del optimizador de hiperparámetros."""
    parser = argparse.ArgumentParser(
        description="Búsqueda de hiperparámetros con Optuna (por ticker, sin seed cruzado)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        help="Ruta al CSV del ticker a optimizar (ej. data/MSFT_features.csv).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Número de trials Optuna a ejecutar.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="Walk-forward splits para cada trial.",
    )
    parser.add_argument(
        "--storage-path",
        default=None,
        help=(
            "Ruta SQLite para persistir el estudio y permitir reanudación. "
            "Ej: optuna_studies/msft.db. Si no se especifica, el estudio es in-memory."
        ),
    )
    parser.add_argument(
        "--best-save-canonical",
        action="store_true",
        help=(
            "Tras la búsqueda, re-entrenar con los mejores parámetros y registrar "
            "en model_registry con --save-canonical."
        ),
    )
    parser.add_argument(
        "--download-extra-tickers",
        action="store_true",
        help=f"Descarga los 8 tickers adicionales: {', '.join(EXTRA_TICKERS)}.",
    )
    parser.add_argument(
        "--study-objective",
        type=str,
        default="sharpe",
        choices=["sharpe", "composite", "multi-objective"],
        help="Objetivo de optimización del trial (default: sharpe).",
    )
    parser.add_argument(
        "--composite-score-profile",
        type=str,
        default="balanced",
        choices=["balanced"],
        help="Perfil de pesos para score compuesto (default: balanced).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed para reproducibilidad del subprocess main.py (default: 7).",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="",
        help=(
            "Modo de torch.compile para el subprocess main.py. "
            "Default '' (desactivado) — evita overhead de compilación en trials. "
            "Opciones: '', 'default', 'max-autotune'."
        ),
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=5,
        help="Trials aleatorios antes de que TPE active su modelo probabilístico.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=42,
        help=(
            "Seed del TPESampler (distinto de --seed que va al subprocess main.py). "
            "Controla la aleatoriedad interna del sampler de Optuna."
        ),
    )
    parser.add_argument(
        "--clean-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Elimina todos los CSVs stale de results/ antes de optimizar. "
            "Evita que _parse_portfolio_csv lea artefactos de runs anteriores."
        ),
    )
    parser.add_argument(
        "--enqueue-canonical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Encola la configuración canónica (seq=96, pred=20, batch=64, clip=0.5) "
            "como primer trial garantizado. Desactivar con --no-enqueue-canonical."
        ),
    )
    args = parser.parse_args()

    # Modo descarga
    if args.download_extra_tickers:
        download_extra_tickers()
        return

    if not args.csv:
        parser.error("--csv es obligatorio salvo con --download-extra-tickers.")

    csv_path = args.csv
    ticker_stem = Path(csv_path).stem
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary_dir = Path("optuna_studies")
    summary_dir.mkdir(exist_ok=True)

    # Limpiar artefactos stale si se solicita (AR #8)
    if args.clean_results:
        stale_csvs = list(results_dir.glob("*.csv"))
        logger.info("Limpiando %d artefactos stale de results/", len(stale_csvs))
        for csv_file in stale_csvs:
            csv_file.unlink()

    study_name = f"tune_{ticker_stem}"

    # Configurar storage SQLite (reanudable) o in-memory
    storage = None
    if args.storage_path:
        storage_file = Path(args.storage_path)
        storage_file.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_file}"
        logger.info("Persistiendo estudio en: %s", storage_file)

    # TPE sampler: trials exploratorios aleatorios antes de usar el modelo probabilístico
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        n_startup_trials=args.n_startup_trials,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,  # reanuda si el estudio SQLite ya existe
    )

    # Encolar configuración canónica como primer trial garantizado (AR #4)
    if args.enqueue_canonical:
        study.enqueue_trial(dict(CANONICAL_TRIAL_PARAMS))
        logger.info("Config canónica encolada como primer trial")

    logger.info(
        "Iniciando búsqueda para %s — %d trials, espacio válido: %d combinaciones "
        "tras restricciones estructurales, objetivo: %s.",
        ticker_stem,
        args.n_trials,
        _count_search_space_combinations(),
        args.study_objective,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            csv_path,
            args.n_splits,
            results_dir,
            study_objective=args.study_objective,
            seed=args.seed,
            compile_mode=args.compile_mode,
        ),
        n_trials=args.n_trials,
        show_progress_bar=False,  # interfiere con el logging estándar
    )

    # ---------------------------------------------------------------------------
    # Resumen de resultados
    # ---------------------------------------------------------------------------
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    if not completed:
        logger.error("Ningún trial completado correctamente.")
        return

    best = study.best_trial

    logger.info("=" * 60)
    logger.info("RESULTADOS — %s", ticker_stem.upper())
    logger.info("  Trials completados: %d | Podados: %d", len(completed), len(pruned))
    logger.info("  Mejor score (%s):  %.4f", args.study_objective, best.value)
    logger.info("  Mejores parámetros:")
    for k, v in best.params.items():
        logger.info("    %-25s %s", k + ":", v)
    logger.info("  Métricas auxiliares del mejor trial:")
    logger.info(
        "    Sortino:          %.4f", best.user_attrs.get("sortino", float("nan"))
    )
    logger.info(
        "    VaR_95:           %.4f", best.user_attrs.get("var_95", float("nan"))
    )
    logger.info(
        "    MaxDD:            %.4f", best.user_attrs.get("max_drawdown", float("nan"))
    )
    logger.info(
        "    Pinball P50:      %s",
        f"{best.user_attrs['pinball_p50']:.4f}"
        if not math.isnan(best.user_attrs.get("pinball_p50", float("nan")))
        else "n/a",
    )
    logger.info(
        "    Coverage 80%%:    %s",
        f"{best.user_attrs['coverage_80']:.4f}"
        if not math.isnan(best.user_attrs.get("coverage_80", float("nan")))
        else "n/a",
    )
    logger.info(
        "    Composite score:  %.4f",
        best.user_attrs.get("composite_score", float("nan")),
    )
    logger.info("=" * 60)

    # Top-5 trials en tabla
    completed_df = _build_completed_trials_dataframe(study)
    cols_show = [
        c
        for c in completed_df.columns
        if "number" in c or "value" in c or "params_" in c or "user_attrs_sortino" in c
    ]
    print("\nTop 5 trials:")
    print(completed_df[cols_show].head(5).to_string(index=False))
    print()

    # Guardar resumen completo
    summary_path = summary_dir / f"{ticker_stem}_trials.csv"
    completed_df.to_csv(summary_path, index=False)
    logger.info("Resumen de trials guardado en: %s", summary_path)

    # ---------------------------------------------------------------------------
    # Run final con los mejores parámetros (opcional)
    # ---------------------------------------------------------------------------
    if args.best_save_canonical:
        _run_best_params(
            csv_path=csv_path,
            best_params=best.params,
            n_splits=args.n_splits,
            save_canonical=True,
            seed=args.seed,
            compile_mode=args.compile_mode,
        )


if __name__ == "__main__":
    main()
