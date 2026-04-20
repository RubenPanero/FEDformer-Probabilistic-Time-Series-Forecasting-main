# -*- coding: utf-8 -*-
"""
Vanguard FEDformer: Sistema Predictivo Estocástico Listo para Producción (Refactorizado).

Orquestador principal del proyecto que enlaza todos los sub-módulos y ejecuta
el ciclo completo de entrenamiento, evaluación y visualización empírica.

Uso:
1. Asegurar entorno activado (.venv)
2. Ejecutar: python main.py --csv <filepath> --targets <targets_separados> [opcionales]
"""

import argparse
import contextlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

from config import TRAINING_PRESETS, FEDformerConfig, apply_preset
from data import TimeSeriesDataset
from simulations import PortfolioSimulator, RiskSimulator
from training import WalkForwardTrainer
from training.forecast_output import ForecastOutput
from utils import get_device, setup_cuda_optimizations
from utils.helpers import set_seed
from utils.io_experiment import (
    build_run_manifest,
    save_fold_metrics,
    save_probabilistic_metrics,
    save_run_manifest,
)
from utils.model_registry import get_specialist, register_specialist

# Consolidación inicial de determinismo e inicializaciones del clúster físico
set_seed(42, deterministic=False)
setup_cuda_optimizations()
device = get_device()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationData:
    """Contenedor puente de artefactos expuestos con la visualización matricial final."""

    forecast: ForecastOutput
    dataset: TimeSeriesDataset
    training_history: pd.DataFrame | None = None
    fold_prob_metrics: list[dict[str, float]] | None = None
    config: FEDformerConfig | None = None
    ticker: str | None = None


def _parse_arguments() -> argparse.Namespace:
    """Parses CLI arguments for the FEDformer training pipeline."""
    parser = argparse.ArgumentParser(
        description="Flow FEDformer: Probabilistic Time-Series Forecasting Engine"
    )
    parser.add_argument(
        "--csv",
        required=True,
        nargs="+",
        help="Path(s) to input CSV file(s), one per ticker.",
    )
    parser.add_argument(
        "--targets",
        required=True,
        help="Comma-separated list of target column names to predict.",
    )
    parser.add_argument(
        "--date-col",
        default=None,
        help="Name of the date/timestamp column to exclude from features.",
    )
    parser.add_argument(
        "--wandb-project",
        default="vanguard-fedformer-flow",
        help="Weights & Biases project name for experiment tracking.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases team/entity for metric logging.",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=24,
        help="Prediction horizon length; must be even (default: 24).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=96,
        help="Input sequence length for the encoder (default: 96).",
    )
    parser.add_argument(
        "--label-len",
        type=int,
        default=48,
        help="Decoder start-token overlap with encoder (default: 48).",
    )
    parser.add_argument(
        "--e-layers",
        type=int,
        default=None,
        help="Number of encoder transformer layers (default: config 2).",
    )
    parser.add_argument(
        "--d-layers",
        type=int,
        default=None,
        help="Number of decoder transformer layers (default: config 1).",
    )
    parser.add_argument(
        "--n-flow-layers",
        type=int,
        default=None,
        help="Number of normalizing flow coupling layers (default: config 4).",
    )
    parser.add_argument(
        "--flow-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension inside normalizing flow conditioner (default: config 64).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs per walk-forward fold (default: config 20).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of walk-forward cross-validation splits (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32).",
    )
    parser.add_argument(
        "--use-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce GPU memory at the cost of speed.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for effective larger batch size (default: 1).",
    )
    parser.add_argument(
        "--finetune-from",
        default=None,
        help="Path to a .pt checkpoint for warm-start fine-tuning.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze encoder/decoder backbone; train only the normalizing flow head.",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="Learning rate override for fine-tuning (default: same as --learning-rate).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable fully deterministic cuDNN mode (may reduce performance).",
    )
    parser.add_argument(
        "--save-fig",
        default=None,
        help="File path to save the portfolio performance plot (e.g., results/portfolio.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Suppress interactive plot display (use in headless environments).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Export predictions and risk/portfolio metrics as CSVs to results/.",
    )
    parser.add_argument(
        "--save-canonical",
        action="store_true",
        default=False,
        help=(
            "Save the last fold checkpoint as {ticker}_canonical.pt "
            "and register the specialist in checkpoints/model_registry.json."
        ),
    )
    parser.add_argument(
        "--return-transform",
        default="none",
        choices=["none", "log_return", "simple_return"],
        help=(
            "Return transform applied before scaling: 'none' (raw prices), "
            "'log_return' (log(P_t/P_{t-1})), 'simple_return' ((P_t-P_{t-1})/P_{t-1})."
        ),
    )
    parser.add_argument(
        "--metric-space",
        default="returns",
        choices=["returns", "prices"],
        help=(
            "Space for reporting metrics and samples: 'returns' (default) or 'prices' "
            "(reconstructs prices from cumulative returns using fold last_prices)."
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability for embeddings, conv, and attention (default: config 0.1).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="L2 weight decay for AdamW optimizer (default: config 1e-5).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="AdamW optimizer learning rate (default: config 1e-4).",
    )
    parser.add_argument(
        "--scheduler-type",
        default=None,
        choices=["none", "cosine", "cosine_warmup"],
        help="Learning rate scheduler: none, cosine, or cosine_warmup (default: config none).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Linear warmup epochs for cosine_warmup scheduler (default: config 0).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs (default: config 5).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Minimum val_loss improvement for early stopping (default: config 5e-3).",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=None,
        help="Max gradient norm for clipping (default: config 1.0). Set 0 to disable.",
    )
    parser.add_argument(
        "--rehearsal-k",
        type=int,
        default=None,
        help="Rehearsal buffer size (number of windows). Enables continual learning.",
    )
    parser.add_argument(
        "--rehearsal-epochs",
        type=int,
        default=None,
        help="Replay steps per training epoch (default: config 1).",
    )
    parser.add_argument(
        "--rehearsal-lr-mult",
        type=float,
        default=None,
        help="Learning rate multiplier for rehearsal replay steps (default: config 0.1).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(TRAINING_PRESETS.keys()),
        help=(
            "Training preset: debug, cpu_safe, gpu_research, "
            "probabilistic_eval. Explicit CLI flags take precedence."
        ),
    )
    parser.add_argument(
        "--conformal-calibration",
        action="store_true",
        default=False,
        help=(
            "Apply post-hoc Conformal Prediction on aggregated predictions "
            "(Approach 2: global calibration)."
        ),
    )
    parser.add_argument(
        "--cp-walkforward",
        action="store_true",
        default=False,
        help=(
            "Apply walk-forward fold-aware Conformal Prediction "
            "(Approach 1: no temporal data leakage)."
        ),
    )
    parser.add_argument(
        "--mc-dropout-eval-samples",
        type=int,
        default=None,
        help="MC Dropout samples used during trainer evaluation folds (default: 20).",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        help=(
            "torch.compile mode: 'max-autotune', 'default', or '' (disabled). "
            "If not specified, uses FEDformerConfig default ('max-autotune')."
        ),
    )
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace) -> List[str]:
    """Asegura la coherencia básica del mapeo interactivo de usuario evitando fallas ruidosas posteriores."""
    # Valida que cada CSV provisto sea accesible en disco
    for csv_path in args.csv:
        if not os.path.exists(csv_path):
            logger.error("Dataset inaccesible o inexistente bajo: %s", csv_path)
            raise FileNotFoundError(f"Lectura imposible sobre {csv_path}")

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        logger.error("Ninguna meta subyacente de análisis provista")
        raise ValueError("Indique al menos una variable predictiva en --targets")

    return targets


def _create_config(
    args: argparse.Namespace, targets: List[str], csv_path: str
) -> FEDformerConfig:
    """Preinstala instancias de comportamiento dictadas externamente al manifest nativo."""
    # Solo parámetros estructurales en el constructor — no tienen equivalente en presets
    config = FEDformerConfig(
        file_path=csv_path,
        target_features=targets,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        date_column=args.date_col,
        wandb_run_name=f"Modular-Flow-FEDformer_{int(time.time())}",
    )

    # Orden correcto: config_base → apply_preset() → flags_CLI
    if args.preset is not None:
        apply_preset(config, args.preset)

    # Parámetros CLI que siempre se aplican (tienen defaults no-None en argparse)
    config.pred_len = args.pred_len
    config.seq_len = args.seq_len
    config.label_len = args.label_len
    config.batch_size = args.batch_size
    config.seed = args.seed
    config.deterministic = args.deterministic
    config.return_transform = args.return_transform
    config.metric_space = args.metric_space
    config.use_gradient_checkpointing = args.use_checkpointing
    config.gradient_accumulation_steps = args.grad_accum_steps
    config.freeze_backbone = args.freeze_backbone

    # Parámetros opcionales (default=None en argparse)
    if args.finetune_from is not None:
        config.finetune_from = args.finetune_from
    if args.finetune_lr is not None:
        config.finetune_lr = args.finetune_lr
    if args.e_layers is not None:
        config.e_layers = args.e_layers
    if args.d_layers is not None:
        config.d_layers = args.d_layers
    if args.n_flow_layers is not None:
        config.n_flow_layers = args.n_flow_layers
    if args.flow_hidden_dim is not None:
        config.flow_hidden_dim = args.flow_hidden_dim
    if args.epochs is not None:
        config.n_epochs_per_fold = args.epochs
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.scheduler_type is not None:
        config.scheduler_type = args.scheduler_type
    if args.warmup_epochs is not None:
        config.warmup_epochs = args.warmup_epochs
    if args.patience is not None:
        config.patience = args.patience
    if args.min_delta is not None:
        config.min_delta = args.min_delta
    if args.gradient_clip_norm is not None:
        config.gradient_clip_norm = args.gradient_clip_norm
    if args.rehearsal_k is not None:
        config.rehearsal_enabled = True
        config.rehearsal_buffer_size = args.rehearsal_k
    if args.rehearsal_epochs is not None:
        config.sections.training.rehearsal.rehearsal_epochs = args.rehearsal_epochs
    if args.rehearsal_lr_mult is not None:
        config.rehearsal_lr_mult = args.rehearsal_lr_mult
    if args.compile_mode is not None:
        config.compile_mode = args.compile_mode
    if args.mc_dropout_eval_samples is not None:
        config.mc_dropout_eval_samples = args.mc_dropout_eval_samples

    logger.info("Transmisión paramétrica asimilada de manera segura")
    logger.info(
        "Métricas inyectadas en topología: Dimensión Modelada=%s, Cabezas(Atención)=%s",
        config.d_model,
        config.n_heads,
    )
    logger.info(
        "Carga iterativa dispuesta: Ciclos(Fold)=%s, Tamaño de Bloque=%s",
        config.n_epochs_per_fold,
        config.batch_size,
    )
    return config


def _load_dataset(config: FEDformerConfig) -> TimeSeriesDataset:
    """Pre-computa el cargador matricial general y escaladores numéricos."""
    logger.info("Apertura de lecturas y transformaciones escalares de features...")
    full_dataset = TimeSeriesDataset(config=config, flag="all")
    logger.info(
        "Tubería de datos inyectada: %s ventanas transicionales analizadas",
        len(full_dataset),
    )
    return full_dataset


def _run_backtest(
    config: FEDformerConfig, full_dataset: TimeSeriesDataset, splits: int
) -> tuple[ForecastOutput, pd.DataFrame | None, list[dict[str, float]]]:
    """Acciona el motor rotatorio de fraccionamiento evaluativo (Walk-forward).

    Devuelve el ForecastOutput, el histórico de entrenamiento por fold/época,
    y las métricas probabilísticas por fold.
    """
    logger.info("Iniciando rampa de backtesting secuencial dinámico...")
    wf_trainer = WalkForwardTrainer(config, full_dataset)
    forecast = wf_trainer.run_backtest(n_splits=splits)

    if forecast.preds_scaled.size == 0:
        logger.error(
            "Imposible procesar sin simulaciones emitidas por la heurística evaluativa. Se fuerza salida."
        )
        raise RuntimeError(
            "Defecto numérico, colapso predictivo sin outputs del Walk-Forward."
        )

    logger.info(
        "Rutina del emulador inter-folds (Walk-Forward) completada brillantemente"
    )
    logger.info(
        "Receptado tensor de previsiones fuera-de-muestra (%s items predictivos)",
        len(forecast.preds_scaled),
    )

    history = wf_trainer.metrics_tracker.to_dataframe()
    fold_prob_metrics = wf_trainer.fold_probabilistic_metrics
    return forecast, history if not history.empty else None, fold_prob_metrics


def _log_risk_summary(var: np.ndarray, cvar: np.ndarray) -> None:
    """Reporte superficial en bitácora de los tensores estadísticos caídos."""
    logger.info("Mitigador VaR (Valor en Riesgo) 95%% Medio: %.4f", float(np.mean(var)))
    logger.info(
        "Margen Severo CVaR (Riesgo Condicional Promedio) 95%%: %.4f",
        float(np.mean(cvar)),
    )


def _log_portfolio_metrics(metrics: Dict[str, Any]) -> None:
    """Transmite al estándar I/O de consola resumen técnico de capitales y ratio financiero."""
    logger.info("Bitácora de Desglose de Rendimientos Estructurales:")
    logger.info(
        "  Exceso de Beneficio por Volatilidad (Ratio Sharpe Anual): %.3f",
        float(metrics.get("sharpe_ratio", 0.0)),
    )
    logger.info(
        "  Hundimiento en el peor tramo predictivo (Max Drawdown): %.2f%%",
        float(metrics.get("max_drawdown", 0.0)) * 100,
    )
    logger.info(
        "  Flujo desviacional del portafolio (Volatilidad Anual): %.2f%%",
        float(metrics.get("volatility", 0.0)) * 100,
    )
    logger.info(
        "  Seguimiento asimétrico sin trampa bajista (Sortino Ratio): %.3f",
        float(metrics.get("sortino_ratio", 0.0)),
    )


def _create_portfolio_figure(
    metrics: Dict[str, Any],
    var: np.ndarray,
    cvar: np.ndarray,
) -> Figure:
    """Engendra una topografía vectorizada renderizando mitigadores estocástico-comerciales."""
    if plt is None:
        raise RuntimeError(
            "Requerimiento denegado, entorno de dibujado (matplotlib) fue destruido o no se encontró."
        )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(
        metrics.get("cumulative_returns", np.array([0.0])),
        label="Acumulación Patrimonial",
        color="#1f77b4",
        linewidth=2,
    )
    ax1.set_title(
        "Evolución y Desempeño Operativo de Portafolio OoS",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Desfase Temporal Evaluado (Periodos)")
    ax1.set_ylabel("Margen Financiero Retornado Acumulado")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    time_steps = range(var.shape[0])
    ax2.plot(
        time_steps,
        np.mean(var, axis=1),
        label="Zona VaR Riesgo Crítico Promediado (95%)",
        color="red",
        alpha=0.8,
    )
    ax2.plot(
        time_steps,
        np.mean(cvar, axis=1),
        label="Derrumbe CVaR Excepcional Condicional (95%)",
        color="darkred",
        alpha=0.8,
    )
    ax2.set_title(
        "Examen Longitudinal Cíclico Sobre Severidad Estocástica",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Desfase Temporal Evaluado (Periodos)")
    ax2.set_ylabel("Magnitud Pérdida Muestral Proyectada")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    return fig


def _log_metrics_to_wandb(
    fig: Figure,
    metrics: Dict[str, Any],
    var: np.ndarray,
    cvar: np.ndarray,
) -> None:
    """Envía el clúster consolidado de variables hacia el entorno colaborativo Web de ser accesible."""
    if wandb is None:
        logger.debug(
            "Módulo wandb no está disponible, abortando sincronización de métricas."
        )
        return
    assert wandb is not None  # Para satisfacer type checkers
    if not wandb.run or not hasattr(wandb.run, "log"):
        logger.debug("Sesión activa de wandb no disponible, omitiendo log de métricas.")
        return

    with contextlib.suppress(RuntimeError, ValueError, AttributeError):
        wandb.run.log(
            {
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "volatility": float(metrics.get("volatility", 0.0)),
                "sortino_ratio": float(metrics.get("sortino_ratio", 0.0)),
                "avg_var": float(np.mean(var)),
                "avg_cvar": float(np.mean(cvar)),
                "performance_chart": wandb.Image(fig),
            }
        )
        logger.info("Transmisión satisfactoria finalizada (W&B Metrics Synced)")


def _handle_visualization_output(fig: Figure, args: argparse.Namespace) -> None:
    """Discierne entre despliegue en host u opcionales exportados de render estático al disco duro."""
    if plt is None:
        return

    if args.save_fig:
        try:
            fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
            logger.info(
                "Artefacto Visual exportado a matriz estática sobre: %s", args.save_fig
            )
        except OSError as exc:
            logger.warning(
                "Caída súbita intentando forzar grabado sobre el binario visualizado: %s",
                exc,
            )

    if not args.no_show:
        with contextlib.suppress(RuntimeError):
            plt.show()

    plt.close(fig)


def _save_results_to_csv(
    forecast: ForecastOutput,
    risk_metrics: Dict[str, Any],
    portfolio_metrics: Dict[str, Any],
    results_dir: Path,
    timestamp: str,
    training_history: pd.DataFrame | None = None,
    fold_prob_metrics: list[dict[str, float]] | None = None,
    config: FEDformerConfig | None = None,
    ticker: str | None = None,
) -> None:
    """Exporta predicciones y métricas a CSVs con marca temporal en results/."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV 1: predicciones por ventana y paso temporal ---
    preds = forecast.preds_for_metrics  # (n_windows, pred_len, n_targets)
    gt = forecast.gt_for_metrics  # (n_windows, pred_len, n_targets)
    n_windows, pred_len, n_targets = preds.shape
    target_names = forecast.target_names or [
        f"target_{target_idx}" for target_idx in range(n_targets)
    ]
    if len(target_names) != n_targets:
        raise ValueError(
            "forecast.target_names debe tener la misma longitud que n_targets. "
            f"Recibido: {len(target_names)} nombres para {n_targets} targets."
        )

    fold_ids = forecast.window_fold_ids  # (n_windows,) int32 o None
    rows: list[dict[str, Any]] = []
    for w in range(n_windows):
        fold = int(fold_ids[w]) if fold_ids is not None and len(fold_ids) > w else -1
        for s in range(pred_len):
            for t_idx, t_name in enumerate(target_names):
                rows.append(
                    {
                        "fold": fold,
                        "window_idx": w,
                        "step": s,
                        "target": t_name,
                        "pred": float(preds[w, s, t_idx]),
                        "ground_truth": float(gt[w, s, t_idx]),
                    }
                )
    preds_path = results_dir / f"predictions_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(preds_path, index=False)
    logger.info("Predicciones exportadas a: %s", preds_path)

    # --- CSV 2: métricas de riesgo (VaR, CVaR por paso y target) ---
    var_arr = risk_metrics.get("var")
    cvar_arr = risk_metrics.get("cvar")
    if var_arr is not None and cvar_arr is not None:
        risk_rows: list[dict[str, Any]] = []
        for s in range(var_arr.shape[0]):
            for t_idx in range(var_arr.shape[1]):
                risk_rows.append(
                    {
                        "step": s,
                        "target_idx": t_idx,
                        "var_95": float(var_arr[s, t_idx]),
                        "cvar_95": float(cvar_arr[s, t_idx]),
                    }
                )
        risk_path = results_dir / f"risk_metrics_{timestamp}.csv"
        pd.DataFrame(risk_rows).to_csv(risk_path, index=False)
        logger.info("Métricas de riesgo exportadas a: %s", risk_path)

    # --- CSV 3: métricas de portafolio (Sharpe, drawdown, Sortino) ---
    portfolio_rows = [
        {
            "metric": "sharpe_ratio",
            "value": float(portfolio_metrics.get("sharpe_ratio", 0.0)),
        },
        {
            "metric": "max_drawdown",
            "value": float(portfolio_metrics.get("max_drawdown", 0.0)),
        },
        {
            "metric": "volatility",
            "value": float(portfolio_metrics.get("volatility", 0.0)),
        },
        {
            "metric": "sortino_ratio",
            "value": float(portfolio_metrics.get("sortino_ratio", 0.0)),
        },
    ]
    portfolio_path = results_dir / f"portfolio_metrics_{timestamp}.csv"
    pd.DataFrame(portfolio_rows).to_csv(portfolio_path, index=False)
    logger.info("Métricas de portafolio exportadas a: %s", portfolio_path)

    # --- CSV 4: histórico de entrenamiento por fold y época ---
    if training_history is not None and not training_history.empty:
        history_path = results_dir / f"training_history_{timestamp}.csv"
        training_history.to_csv(history_path, index=False)
        logger.info("Histórico de entrenamiento exportado a: %s", history_path)

    # --- Artefactos probabilísticos (PR 4): manifiesto JSON + CSVs métricas ---
    if fold_prob_metrics is not None and config is not None and ticker is not None:
        manifest = build_run_manifest(
            config=config,
            ticker=ticker,
            metrics_agg=portfolio_metrics,
            monitor_metric=config.monitor_metric,
            seed=config.seed,
            dataset_path=str(config.file_path),
            timestamp=timestamp,
        )
        save_run_manifest(manifest, results_dir, timestamp)
        save_probabilistic_metrics(fold_prob_metrics, results_dir, timestamp, ticker)
        save_fold_metrics(fold_prob_metrics, results_dir, timestamp, ticker)


def _apply_cp_calibration(
    forecast: ForecastOutput, alpha: float = 0.2
) -> dict[str, float | np.ndarray]:
    """Calibración conformal post-hoc sobre predicciones agregadas (Enfoque 2 — global).

    Usa todos los residuos como conjunto de calibración para estimar q_hat,
    luego calcula la cobertura real del intervalo simétrico [pred-q_hat, pred+q_hat].
    Garantía teórica: coverage ≈ 1 - alpha = 0.80.

    Args:
        forecast: ForecastOutput con preds_for_metrics y gt_for_metrics.
        alpha: Nivel de no-cobertura (default 0.2 → objetivo 80%).

    Returns:
        Dict con q_hat, cp_coverage_80, cp_lower y cp_upper (estos últimos como arrays).
    """
    from utils.calibration import apply_conformal_interval, conformal_quantile

    preds_flat = forecast.preds_for_metrics.reshape(-1)
    gt_flat = forecast.gt_for_metrics.reshape(-1)
    q_hat = conformal_quantile(gt_flat, preds_flat, alpha=alpha)
    cp_lower, cp_upper = apply_conformal_interval(preds_flat, q_hat)
    cp_coverage_80 = float(np.mean((gt_flat >= cp_lower) & (gt_flat <= cp_upper)))
    return {
        "q_hat": float(q_hat),
        "cp_coverage_80": cp_coverage_80,
        "cp_lower": cp_lower,
        "cp_upper": cp_upper,
    }


def _apply_cp_walkforward(
    forecast: ForecastOutput, alpha: float = 0.2
) -> dict[str, Any]:
    """Calibración conformal walk-forward fold-aware (Enfoque 1, sin data leakage).

    Para cada fold k calibra usando residuos de folds 0..k-1.
    Fold 0 se excluye del cálculo de cobertura agregada (sin datos previos).

    Args:
        forecast: ForecastOutput con preds_for_metrics, gt_for_metrics y window_fold_ids.
        alpha: Nivel de no-cobertura (default 0.2 → objetivo 80%).

    Returns:
        Dict con cp_wf_coverage_80, cp_wf_q_hat_by_fold y cp_wf_folds_calibrated.
    """
    from utils.calibration import (  # noqa: PLC0415
        apply_conformal_interval,
        conformal_calibration_walkforward,
    )

    preds = forecast.preds_for_metrics
    gt = forecast.gt_for_metrics
    fold_ids = forecast.window_fold_ids

    if fold_ids is None:
        logger.warning(
            "CP walk-forward: window_fold_ids es None. "
            "Se requiere ForecastOutput generado por WalkForwardTrainer."
        )
        return {
            "cp_wf_coverage_80": float("nan"),
            "cp_wf_q_hat_by_fold": {},
            "cp_wf_folds_calibrated": 0,
        }

    unique_folds = sorted({int(f) for f in fold_ids})
    residuals_by_fold: dict[int, np.ndarray] = {}
    for fold_k in unique_folds:
        mask = fold_ids == fold_k
        residuals_by_fold[fold_k] = np.abs(
            gt[mask].reshape(-1) - preds[mask].reshape(-1)
        )

    q_hat_by_fold = conformal_calibration_walkforward(residuals_by_fold, alpha=alpha)

    covered_all: list[bool] = []
    for fold_k, q_hat in q_hat_by_fold.items():
        if q_hat is None:
            continue
        mask = fold_ids == fold_k
        cp_lower, cp_upper = apply_conformal_interval(preds[mask].reshape(-1), q_hat)
        covered_all.extend(
            (gt[mask].reshape(-1) >= cp_lower) & (gt[mask].reshape(-1) <= cp_upper)
        )

    folds_calibrated = sum(1 for v in q_hat_by_fold.values() if v is not None)
    cp_wf_coverage_80 = float(np.mean(covered_all)) if covered_all else float("nan")

    return {
        "cp_wf_coverage_80": cp_wf_coverage_80,
        "cp_wf_q_hat_by_fold": dict(q_hat_by_fold),
        "cp_wf_folds_calibrated": folds_calibrated,
    }


def _run_portfolio_simulation(
    data: SimulationData,
    risk_stats: Tuple[np.ndarray, np.ndarray],
) -> Tuple[Dict[str, Any], Figure]:
    """Acciona el emulador lógico de trade-in uniendo lo predicho versus lo comprobado."""
    var, cvar = risk_stats

    portfolio_sim = PortfolioSimulator(data.forecast)
    strategy_returns = portfolio_sim.run_simple_strategy()

    metrics = portfolio_sim.calculate_metrics(strategy_returns)
    fig = _create_portfolio_figure(metrics, var, cvar)

    return metrics, fig


def _run_simulations_and_visualize(
    data: SimulationData,
    args: argparse.Namespace,
    timestamp: str | None = None,
) -> Dict[str, Any]:
    """Ejecuta los peritajes estadísticos marginales derivados de toda la secuencia matemática general.

    Retorna las métricas de portafolio calculadas (vacío si no aplica).
    """
    logger.info(
        "Lanzando subsistemas analíticos (Validador Riesgo & Estrategia Portafolio)..."
    )

    risk_sim = RiskSimulator(data.forecast)
    var = risk_sim.calculate_var()
    cvar = risk_sim.calculate_cvar()
    _log_risk_summary(var, cvar)

    # Calibración conformal post-hoc si se solicitó
    cp_result: dict[str, Any] = {}
    if getattr(args, "conformal_calibration", False):
        cp_result = _apply_cp_calibration(data.forecast)
        logger.info(
            "CP calibración: q_hat=%.4f | cp_coverage_80=%.4f (objetivo ≥ 0.80)",
            cp_result["q_hat"],
            cp_result["cp_coverage_80"],
        )

    # Calibración conformal walk-forward si se solicitó (Enfoque 1: fold-aware)
    cp_wf_result: dict[str, Any] = {}
    if getattr(args, "cp_walkforward", False):
        cp_wf_result = _apply_cp_walkforward(data.forecast)
        logger.info(
            "CP walk-forward: coverage_80=%.4f | folds_calibrados=%d (objetivo ≥ 0.80)",
            cp_wf_result["cp_wf_coverage_80"],
            cp_wf_result["cp_wf_folds_calibrated"],
        )

    if data.forecast.gt_for_metrics.shape[1] <= 1:
        logger.info(
            "Omitiendo cálculos de simulación por límite infranqueable predictivo (Paso Ciego de TimeStep <= 1)"
        )
        return {}

    try:
        metrics, fig = _run_portfolio_simulation(data, (var, cvar))
    except ValueError as exc:
        logger.warning(
            "Evaluador financiero corrompido, bloque ignorado preventivamente: %s", exc
        )
        return {}

    _log_portfolio_metrics(metrics)
    _log_metrics_to_wandb(fig, metrics, var, cvar)
    _handle_visualization_output(fig, args)

    # Añadir métricas de calibración conformal al dict de resultados
    if cp_result:
        metrics["cp_q_hat"] = float(cp_result["q_hat"])
        metrics["cp_coverage_80"] = float(cp_result["cp_coverage_80"])

    # Añadir métricas walk-forward al dict de resultados
    if cp_wf_result:
        metrics["cp_wf_coverage_80"] = float(cp_wf_result["cp_wf_coverage_80"])
        metrics["cp_wf_folds_calibrated"] = int(cp_wf_result["cp_wf_folds_calibrated"])

    # Exportar CSVs de resultados si el usuario lo solicitó
    if getattr(args, "save_results", False):
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        _save_results_to_csv(
            forecast=data.forecast,
            risk_metrics={"var": var, "cvar": cvar},
            portfolio_metrics=metrics,
            results_dir=results_dir,
            timestamp=ts,
            training_history=data.training_history,
            fold_prob_metrics=data.fold_prob_metrics,
            config=data.config,
            ticker=data.ticker,
        )

    return metrics


def _print_ticker_summary(
    ticker_results: List[Dict[str, Any]],
    args: argparse.Namespace,
    timestamp: str,
) -> None:
    """Imprime resumen comparativo de métricas por ticker y exporta CSV si --save-results."""
    if not ticker_results:
        return

    logger.info("=" * 60)
    logger.info("RESUMEN COMPARATIVO MULTI-TICKER")
    logger.info("=" * 60)
    logger.info("%-30s %10s %12s", "Ticker (CSV)", "Sharpe", "Max Drawdown")
    logger.info("-" * 60)

    for entry in ticker_results:
        logger.info(
            "%-30s %10.3f %11.2f%%",
            Path(entry["csv"]).stem,
            entry["sharpe_ratio"],
            entry["max_drawdown"] * 100,
        )

    logger.info("=" * 60)

    # Exportar comparativa si se solicitó
    if getattr(args, "save_results", False):
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        comparison_rows = [
            {
                "ticker": Path(e["csv"]).stem,
                "csv_path": e["csv"],
                "sharpe_ratio": e["sharpe_ratio"],
                "max_drawdown": e["max_drawdown"],
                "volatility": e["volatility"],
                "sortino_ratio": e["sortino_ratio"],
            }
            for e in ticker_results
        ]
        comparison_path = results_dir / f"ticker_comparison_{timestamp}.csv"
        pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
        logger.info("Comparativa multi-ticker exportada a: %s", comparison_path)


def _build_config_dict(config: FEDformerConfig, n_splits: int | None = None) -> dict:
    """Construye config_dict para el registry del especialista.

    Args:
        config: Configuración del modelo entrenado.
        n_splits: Número de splits walk-forward (opcional, de args.splits).

    Returns:
        Diccionario JSON-serializable con los hiperparametros minimos
        necesarios para reconstruir el modelo y la configuracion de inferencia
        del especialista canónico.
    """
    d = {
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "label_len": config.label_len,
        "return_transform": config.return_transform,
        "metric_space": config.metric_space,
        "gradient_clip_norm": config.gradient_clip_norm,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "target_features": list(config.target_features),
        # Parámetros de arquitectura — necesarios para reconstruir el modelo en inferencia
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
        "e_layers": config.e_layers,
        "d_layers": config.d_layers,
        "modes": config.modes,
        "dropout": config.dropout,
        "n_flow_layers": config.n_flow_layers,
        "flow_hidden_dim": config.flow_hidden_dim,
        "enc_in": config.enc_in,
        "dec_in": config.dec_in,
    }
    if n_splits is not None:
        d["n_splits"] = n_splits
    return d


def _save_canonical_specialist(
    csv_path: str,
    args: argparse.Namespace,
    config: FEDformerConfig,
    full_dataset: TimeSeriesDataset,
    metrics: Dict[str, Any],
) -> None:
    """Registra el especialista del ticker en el model_registry y copia su checkpoint canónico.

    Extrae el ticker del nombre del CSV, localiza el checkpoint del último fold,
    construye los metadatos y llama a register_specialist.

    Args:
        csv_path: Ruta al CSV del ticker procesado.
        args: Namespace de argparse con todos los flags CLI.
        config: Configuración FEDformerConfig usada en el entrenamiento.
        full_dataset: Dataset cargado (para obtener nº de filas y features).
        metrics: Métricas de portafolio calculadas por PortfolioSimulator.
    """
    # Derivar ticker del nombre del CSV eliminando sufijo "_features"
    ticker = Path(csv_path).stem.replace("_features", "").upper()

    # Checkpoint del último fold (índice = n_splits - 1)
    last_fold_idx = args.splits - 1
    checkpoint_src = Path("checkpoints") / f"best_model_fold_{last_fold_idx}.pt"

    if not checkpoint_src.exists():
        logger.warning(
            "Checkpoint del último fold no encontrado en %s; "
            "omitiendo registro canónico para %s.",
            checkpoint_src,
            ticker,
        )
        return

    # Construir dict de métricas desde el resultado de simulación
    metrics_dict = {
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "sortino": float(metrics.get("sortino_ratio", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "volatility": float(metrics.get("volatility", 0.0)),
    }

    # Construir dict de configuración con los parámetros del entrenamiento
    config_dict = _build_config_dict(config, n_splits=args.splits)

    # Información del dataset (nº de filas del dataset completo)
    n_rows = len(full_dataset.full_data_scaled)
    n_features = (
        full_dataset.full_data_scaled.shape[1]
        if hasattr(full_dataset.full_data_scaled, "shape")
        else 0
    )
    data_info = {
        "file": csv_path,
        "rows": n_rows,
        "features": n_features,
    }

    # Reconstruir el comando CLI de forma aproximada para reproducibilidad
    training_command = (
        f"MPLBACKEND=Agg python3 main.py --csv {csv_path} --targets {args.targets} "
        f"--seq-len {args.seq_len} --pred-len {args.pred_len} "
        f"--batch-size {args.batch_size} --splits {args.splits} "
        f"--return-transform {args.return_transform} --metric-space {args.metric_space} "
        f"--gradient-clip-norm {args.gradient_clip_norm} "
        "--save-results --no-show --save-canonical"
    )

    # Guardia 1: Sharpe mínimo absoluto
    min_sharpe = 0.5
    new_sharpe = metrics_dict["sharpe"]
    if new_sharpe < min_sharpe:
        logger.warning(
            "Especialista '%s' omitido: Sharpe %.3f < umbral mínimo %.1f.",
            ticker,
            new_sharpe,
            min_sharpe,
        )
        return

    # Guardia 2: no sobreescribir si el registry ya tiene un modelo mejor
    existing = get_specialist(ticker)
    if existing is not None:
        existing_sharpe = existing.get("metrics", {}).get("sharpe", float("-inf"))
        if new_sharpe <= existing_sharpe:
            logger.info(
                "Especialista '%s' no actualizado: Sharpe nuevo %.3f ≤ existente %.3f.",
                ticker,
                new_sharpe,
                existing_sharpe,
            )
            return

    # Guardar artefactos de preprocessing solo si el modelo pasó ambas guardias
    preprocessing_dir = Path("checkpoints") / f"{ticker.lower()}_preprocessing"
    try:
        full_dataset.preprocessor.save_artifacts(preprocessing_dir)
        logger.info("Artefactos de preprocessing guardados en %s", preprocessing_dir)
        data_info["preprocessing_artifacts"] = str(preprocessing_dir)
    except (AttributeError, OSError, RuntimeError) as exc:
        logger.error(
            "Especialista '%s' no registrado: error al guardar artefactos de preprocessing: %s",
            ticker,
            exc,
        )
        return

    try:
        canonical_path = register_specialist(
            ticker=ticker,
            checkpoint_src=checkpoint_src,
            metrics=metrics_dict,
            config_dict=config_dict,
            data_info=data_info,
            training_command=training_command,
            notes=f"Auto-registrado por --save-canonical en {date.today().isoformat()}.",
        )
        logger.info(
            "Especialista '%s' registrado con checkpoint canónico: %s",
            ticker,
            canonical_path,
        )
    except OSError as exc:
        logger.warning("Error al registrar especialista '%s': %s", ticker, exc)


def main() -> None:
    """Nodo central asimilador operando los subsistemas acoplados iterativos."""
    try:
        args = _parse_arguments()
        set_seed(args.seed, deterministic=args.deterministic)

        targets = _validate_inputs(args)

        # Marca temporal compartida para todos los CSVs de esta ejecución
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Acumula métricas por ticker para el resumen comparativo final
        ticker_results: List[Dict[str, Any]] = []

        for csv_path in args.csv:
            logger.info("Procesando ticker: %s", csv_path)

            config = _create_config(args, targets, csv_path)
            full_dataset = _load_dataset(config)
            forecast, training_history, fold_prob_metrics = _run_backtest(
                config, full_dataset, args.splits
            )

            ticker_stem = Path(csv_path).stem
            sim_data = SimulationData(
                forecast=forecast,
                dataset=full_dataset,
                training_history=training_history,
                fold_prob_metrics=fold_prob_metrics,
                config=config,
                ticker=ticker_stem,
            )

            # Sufijo por ticker para evitar colisión de nombres de archivo
            ticker_ts = f"{run_timestamp}_{ticker_stem}"

            metrics = _run_simulations_and_visualize(
                sim_data, args, timestamp=ticker_ts
            )

            # Registrar checkpoint canónico del especialista si se solicitó
            if getattr(args, "save_canonical", False):
                _save_canonical_specialist(
                    csv_path=csv_path,
                    args=args,
                    config=config,
                    full_dataset=full_dataset,
                    metrics=metrics,
                )

            ticker_results.append(
                {
                    "csv": csv_path,
                    "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                    "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                    "volatility": float(metrics.get("volatility", 0.0)),
                    "sortino_ratio": float(metrics.get("sortino_ratio", 0.0)),
                }
            )

        # Resumen comparativo sólo cuando se procesaron múltiples tickers
        if len(args.csv) > 1:
            _print_ticker_summary(ticker_results, args, run_timestamp)

        logger.info(
            "Validación, Entrenamiento e Inferencias resueltas triunfalmente. Flujo terminado."
        )

    except (FileNotFoundError, ValueError, RuntimeError):
        logger.exception(
            "Secuencia destructiva abordó el Thread Main general, paralizando arquitectura."
        )
        raise


if __name__ == "__main__":
    main()
