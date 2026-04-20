# -*- coding: utf-8 -*-
"""
Fine-tuning secuencial multi-ticker sobre un checkpoint base.

Continual Learning sin rehearsal: el checkpoint del último fold de cada ticker
se propaga como punto de partida del siguiente.
"""

import argparse
import logging
import os
from pathlib import Path

from config import FEDformerConfig
from data.dataset import TimeSeriesDataset
from training.trainer import WalkForwardTrainer

logger = logging.getLogger(__name__)


def _load_symbols_from_file(path: str) -> list[str]:
    """Lee lista de símbolos desde un archivo de texto (un símbolo por línea).

    Ignora líneas vacías y líneas que comiencen con '#'.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def finetune_sequence(
    symbols: list[str],
    base_checkpoint: str,
    output_dir: str = "checkpoints/finetuned",
    n_splits: int = 3,
    main_epochs: int = 3,
    scheduler_type: str = "none",
    warmup_epochs: int = 0,
    patience: int = 5,
    min_delta: float = 5e-3,
    return_transform: str = "none",
    metric_space: str = "returns",
    time_features: list[str] | None = None,
    rehearsal_k: int | None = None,
    rehearsal_epochs: int | None = None,
    rehearsal_lr_mult: float | None = None,
) -> str:
    """Fine-tuna secuencialmente el modelo base sobre cada ticker en ``symbols``.

    Args:
        symbols: Lista de tickers a procesar en orden.
        base_checkpoint: Ruta al checkpoint de partida.
        output_dir: Directorio raíz donde se guardan los checkpoints por ticker.
        n_splits: Splits walk-forward por ticker.
        main_epochs: Épocas de fine-tuning por fold.
        scheduler_type: Scheduler de LR ('none', 'cosine', 'cosine_warmup').
        warmup_epochs: Épocas de warmup para cosine_warmup.
        patience: Paciencia de early stopping (0 = desactivado).
        min_delta: Delta mínimo de mejora para early stopping.
        return_transform: Transformación de retornos ('none', 'log_return', 'simple_return').
        metric_space: Espacio de métricas ('returns', 'prices').
        time_features: Features temporales adicionales incluidas en el dataset.
        rehearsal_k: Tamaño del rehearsal buffer (nº ventanas). None desactiva el rehearsal.
        rehearsal_epochs: Pasos de replay por época (default config: 1).
        rehearsal_lr_mult: Multiplicador de LR para pasos de rehearsal (default config: 0.1).

    Returns:
        Ruta al checkpoint final maestro (último ticker procesado con éxito).
    """
    current_checkpoint = base_checkpoint

    if not os.path.exists(current_checkpoint):
        raise FileNotFoundError(f"El checkpoint base no existe: {current_checkpoint}")

    for symbol in symbols:
        logger.info(
            "\n%s\nIniciando Fine-Tuning Secuencial para: %s\n%s",
            "=" * 50,
            symbol,
            "=" * 50,
        )

        # 1. Preparar dataset
        data_path = f"data/{symbol}_features.csv"
        if not os.path.exists(data_path):
            logger.info("Generando dataset financiero para %s...", symbol)
            from data.financial_dataset_builder import build_financial_dataset  # noqa: PLC0415

            build_financial_dataset(symbol, "data", use_mock=True)

        # 2. Configurar entrenamiento
        config = FEDformerConfig(
            target_features=["Close"],
            file_path=data_path,
            seq_len=96,
            label_len=48,
            pred_len=24,
            date_column="date",
            n_epochs_per_fold=main_epochs,
            batch_size=32,
            finetune_from=current_checkpoint,
            finetune_lr=1e-5,
            freeze_backbone=False,
            n_regimes=3,
            scaling_strategy="robust",
            scheduler_type=scheduler_type,
            warmup_epochs=warmup_epochs,
            patience=patience,
            min_delta=min_delta,
            return_transform=return_transform,
            metric_space=metric_space,
            time_features=time_features or [],
        )

        # 2b. Configurar rehearsal buffer si se solicitó
        if rehearsal_k is not None:
            config.rehearsal_enabled = True
            config.rehearsal_buffer_size = rehearsal_k
        if rehearsal_epochs is not None:
            config.sections.training.rehearsal.rehearsal_epochs = rehearsal_epochs
        if rehearsal_lr_mult is not None:
            config.rehearsal_lr_mult = rehearsal_lr_mult

        # 3. Dataset y entrenador
        dataset = TimeSeriesDataset(config, flag="all")
        trainer = WalkForwardTrainer(config, full_dataset=dataset)

        run_ckpt_dir = Path(output_dir) / symbol
        run_ckpt_dir.mkdir(exist_ok=True, parents=True)
        trainer.checkpoint_dir = run_ckpt_dir

        # 4. Fine-tuning vía walk-forward
        try:
            forecast = trainer.run_backtest(n_splits=n_splits)
            logger.info(
                "Entrenamiento para %s finalizado. Predicciones: %s",
                symbol,
                forecast.preds_scaled.shape,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            logger.error("Fallo durante el finetuning de %s: %s", symbol, exc)
            continue

        # 5. Propagar el checkpoint del último fold al siguiente ticker
        last_fold_idx = n_splits - 1
        last_ckpt = trainer.checkpoint_dir / f"best_model_fold_{last_fold_idx}.pt"
        if last_ckpt.exists():
            current_checkpoint = str(last_ckpt)
            logger.info(
                "Siguiente checkpoint base actualizado a: %s", current_checkpoint
            )
        else:
            fallback = sorted(
                trainer.checkpoint_dir.glob("*.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            if fallback:
                current_checkpoint = str(fallback[-1])
                logger.info("Usando checkpoint alternativo: %s", current_checkpoint)
            else:
                logger.warning(
                    "No se generó ningún checkpoint en este paso. Se reusará el anterior."
                )

    logger.info(
        "Proceso de escalado secuencial completado. Checkpoint final maestro: %s",
        current_checkpoint,
    )
    return current_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning secuencial multi-ticker sobre un checkpoint base"
    )

    parser.add_argument(
        "--base_ckpt",
        type=str,
        required=True,
        help="Ruta al checkpoint base (ej. checkpoints/best_model_fold_4.pt)",
    )

    # --symbols y --symbols-file son mutuamente excluyentes
    symbols_group = parser.add_mutually_exclusive_group()
    symbols_group.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["AAPL", "MSFT", "AMZN", "NVDA"],
        help="Tickers a fine-tunear (default: AAPL MSFT AMZN NVDA)",
    )
    symbols_group.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Archivo de texto con un símbolo por línea (alternativa a --symbols)",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints/finetuned",
        help="Directorio raíz de salida para checkpoints por ticker",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=3,
        help="Splits walk-forward por ticker (default: 3)",
    )
    parser.add_argument(
        "--main-epochs",
        type=int,
        default=3,
        help="Épocas de fine-tuning por fold (default: 3)",
    )
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="none",
        choices=["none", "cosine", "cosine_warmup"],
        help="Scheduler de tasa de aprendizaje (default: none)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Épocas de warmup para cosine_warmup (default: 0)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Paciencia de early stopping; 0 = desactivado (default: 5)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=5e-3,
        help="Delta mínimo de mejora para early stopping (default: 5e-3)",
    )
    parser.add_argument(
        "--return-transform",
        type=str,
        default="none",
        choices=["none", "log_return", "simple_return"],
        help="Transformación de retornos aplicada al target (default: none)",
    )
    parser.add_argument(
        "--metric-space",
        type=str,
        default="returns",
        choices=["returns", "prices"],
        help="Espacio de cálculo de métricas financieras (default: returns)",
    )
    parser.add_argument(
        "--time-features",
        type=str,
        nargs="*",
        default=[],
        metavar="FEATURE",
        help="Features temporales adicionales (ej. --time-features month weekday)",
    )
    parser.add_argument(
        "--rehearsal-k",
        type=int,
        default=None,
        help="Tamaño del rehearsal buffer (nº ventanas). Activa el continual learning.",
    )
    parser.add_argument(
        "--rehearsal-epochs",
        type=int,
        default=None,
        help="Pasos de replay por época de entrenamiento (default: config 1).",
    )
    parser.add_argument(
        "--rehearsal-lr-mult",
        type=float,
        default=None,
        help="Multiplicador de LR para pasos de rehearsal (default: config 0.1).",
    )

    args = parser.parse_args()

    symbols = (
        _load_symbols_from_file(args.symbols_file)
        if args.symbols_file
        else args.symbols
    )

    finetune_sequence(
        symbols=symbols,
        base_checkpoint=args.base_ckpt,
        output_dir=args.out_dir,
        n_splits=args.n_splits,
        main_epochs=args.main_epochs,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        return_transform=args.return_transform,
        metric_space=args.metric_space,
        time_features=args.time_features,
        rehearsal_k=args.rehearsal_k,
        rehearsal_epochs=args.rehearsal_epochs,
        rehearsal_lr_mult=args.rehearsal_lr_mult,
    )
