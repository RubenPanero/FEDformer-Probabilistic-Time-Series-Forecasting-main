# -*- coding: utf-8 -*-
"""
Sistema de entrenamiento walk-forward para el modelo FEDformer.
Refactorizado a Python 3.10+ para garantizar typing nativo purificado, tolerancia a fallos, y eficiencia PEP 8.
"""

import gc
import logging
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import autocast
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

try:
    import wandb
    from wandb.errors import Error as WandbError
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore
    WandbError = Exception

from config import FEDformerConfig
from training.forecast_output import ForecastOutput
from training.rehearsal_buffer import RehearsalBuffer
from data import TimeSeriesDataset
from models import Flow_FEDformer
from training.utils import mc_dropout_inference
from utils import MetricsTracker, get_device
from utils.probabilistic_metrics import (
    calibration_gap,
    coverage_by_quantile_pair,
    crps_from_samples,
    interval_width,
    multi_quantile_pinball_loss,
)

logger = logging.getLogger(__name__)
device = get_device()
DEFAULT_QUANTILE_LEVELS = np.array([0.1, 0.5, 0.9], dtype=np.float32)


class _SeedWorker:
    """Callable picklable para inicializar semilla de cada worker de DataLoader.

    Clase de nivel de módulo — más robusta que functools.partial en contexto
    multiprocessing_context='spawn' de Python 3.12+ (el pickle serializa la clase
    por nombre, evitando ambigüedades de referencia).
    """

    __slots__ = ("base_seed",)

    def __init__(self, base_seed: int) -> None:
        self.base_seed = base_seed

    def __call__(self, worker_id: int) -> None:
        np.random.seed(self.base_seed + worker_id)
        torch.manual_seed(self.base_seed + worker_id)


def _cumulative_returns_to_prices(
    returns: np.ndarray, last_prices: np.ndarray, mode: str
) -> np.ndarray:
    """Reconstruye precios desde retornos acumulados. returns shape: (..., n_targets)."""
    if mode == "log_return":
        multipliers = np.exp(np.cumsum(returns, axis=-2))
    else:  # simple_return
        multipliers = np.cumprod(1 + returns, axis=-2)
    return last_prices * multipliers


class _EarlyStopping:
    """Monitor de parada anticipada basado en paciencia y delta mínimo."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, loss: float) -> bool:
        if self.patience <= 0:
            return False
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


@dataclass(frozen=True)
class TrainingComponents:
    """Consolida el modelo, optimizador, escalador y cargadores (loaders) para iterar en entrenamiento."""

    model: Flow_FEDformer
    optimizer: Optimizer
    scaler: torch.amp.GradScaler | None
    train_loader: DataLoader
    test_loader: DataLoader
    fold: int
    scheduler: Any = None
    val_loader: DataLoader | None = None


@dataclass(frozen=True)
class BatchTensors:
    """Estructura de tensores en bloque transbordados al dispositivo físico."""

    encoder: torch.Tensor
    decoder: torch.Tensor
    target: torch.Tensor
    regime: torch.Tensor


class WalkForwardTrainer:
    """Entrenador walk-forward para evaluacion leakage-safe de series temporales.

    Orquesta splits temporales, entrenamiento por fold, evaluacion
    probabilistica y persistencia de checkpoints sin mezclar informacion del
    futuro en el pipeline de entrenamiento.
    """

    def __init__(
        self, config: FEDformerConfig, full_dataset: TimeSeriesDataset
    ) -> None:
        self.config = config
        self.full_dataset = full_dataset
        self.wandb_run = None
        self.metrics_tracker = MetricsTracker()
        self.fold_probabilistic_metrics: list[dict[str, float]] = []
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        rehearsal_cfg = config.sections.training.rehearsal
        self.rehearsal_buffer: RehearsalBuffer | None = (
            RehearsalBuffer(rehearsal_cfg.buffer_size, "uniform")
            if rehearsal_cfg.enabled
            else None
        )

    # GPUs con menos de este número de SMs no soportan max-autotune fiablemente
    _MIN_SMS_FOR_MAX_AUTOTUNE = 40

    @staticmethod
    def _python_headers_available() -> bool:
        """Comprueba si Python.h está disponible para compilación JIT de triton."""
        import sysconfig  # pylint: disable=import-outside-toplevel

        include_dir = sysconfig.get_path("include")
        if not include_dir:
            return False
        return os.path.exists(os.path.join(include_dir, "Python.h"))

    @staticmethod
    def _effective_compile_mode(requested_mode: str) -> str:
        """Degrada max-autotune a '' en GPUs con SMs insuficientes.

        max-autotune requiere suficientes SMs para los kernels GEMM optimizados.
        En GPUs con pocos SMs (e.g. RTX 4050 Laptop: 20 SMs) el inductor
        genera kernels incorrectos que producen NaN en la loss.
        """
        if requested_mode != "max-autotune" or not torch.cuda.is_available():
            return requested_mode
        n_sms = torch.cuda.get_device_properties(0).multi_processor_count
        if n_sms < WalkForwardTrainer._MIN_SMS_FOR_MAX_AUTOTUNE:
            logger.warning(
                "GPU tiene %d SMs (mínimo para max-autotune: %d). "
                "Desactivando torch.compile para evitar NaN en loss.",
                n_sms,
                WalkForwardTrainer._MIN_SMS_FOR_MAX_AUTOTUNE,
            )
            return ""
        return requested_mode

    def _get_model(self) -> Flow_FEDformer:
        """Crea y preferiblemente compila el submodelo instanciado."""
        try:
            model = Flow_FEDformer(self.config).to(device, non_blocking=True)
            if self.config.finetune_from or self.config.freeze_backbone:
                return model
            compile_mode = self._effective_compile_mode(self.config.compile_mode)
            if compile_mode and device.type == "cuda" and hasattr(torch, "compile"):
                if not self._python_headers_available():
                    logger.warning(
                        "Python.h no encontrado — torch.compile desactivado. "
                        "Instala python3-dev para activar compilación JIT."
                    )
                    return model
                logger.info(
                    "Compilando el modelo con modo dinámico: %s",
                    compile_mode,
                )
                return torch.compile(model, mode=compile_mode)
            return model
        except (RuntimeError, TypeError) as exc:
            logger.warning(
                "Fallo durante invocación compilada del modelo de PyTorch (%s). Transicionando a fallback uncompiled.",
                exc,
            )
            return Flow_FEDformer(self.config).to(device, non_blocking=True)

    def _prepare_data_loaders(
        self, train_subset: Subset, test_subset: Subset
    ) -> tuple[DataLoader, DataLoader]:
        """Prepara y devuelve los cargadores en memoria DataLoaders purificados."""
        num_workers = self._num_workers()
        pin_memory = self._pin_memory_enabled()
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        # _SeedWorker es una clase de módulo picklable — evita ambigüedades de
        # functools.partial en multiprocessing_context="spawn" de Python 3.12+
        worker_init_fn = _SeedWorker(self.config.seed)

        worker_kwargs = (
            {"prefetch_factor": 2, "multiprocessing_context": "spawn"}
            if num_workers > 0
            else {}
        )

        # Evitando memory leaks sobre workers persistentes reseteando la sesión adecuadamente
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,  # False evita 0 batches cuando n_ventanas < batch_size (ej. fold 1 con seq_len grande)
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,  # Explicitely false per safe memory policy between folds
            worker_init_fn=worker_init_fn,
            generator=generator,
            **worker_kwargs,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,  # Safe parallel evaluation
            worker_init_fn=worker_init_fn,
            generator=generator,
            **worker_kwargs,
        )
        return train_loader, test_loader

    def _make_loader(self, subset: Subset, shuffle: bool = False) -> DataLoader:
        """Crea un DataLoader con la configuración estándar del trainer."""
        num_workers = self._num_workers()
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        # _SeedWorker es una clase de módulo picklable — mismo patrón que _prepare_data_loaders
        worker_init_fn = _SeedWorker(self.config.seed)
        return DataLoader(
            subset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self._pin_memory_enabled(),
            persistent_workers=False,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            generator=generator,
            **(
                {"prefetch_factor": 2, "multiprocessing_context": "spawn"}
                if num_workers > 0
                else {}
            ),
        )

    def _pin_memory_enabled(self) -> bool:
        """Activa pin_memory solo si está habilitado explícitamente y hay CUDA."""
        return bool(self.config.pin_memory and torch.cuda.is_available())

    def _num_workers(self) -> int:
        """Resuelve num_workers de runtime; por defecto mantiene auto-escalado."""
        if self.config.num_workers is not None:
            return self.config.num_workers
        return min(4, os.cpu_count() // 2) if os.cpu_count() else 0

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> BatchTensors:
        """Transborda el bloque de tensores al dispositivo hardware principal."""
        return BatchTensors(
            encoder=batch["x_enc"].to(device, non_blocking=True),
            decoder=batch["x_dec"].to(device, non_blocking=True),
            target=batch["y_true"].to(device, non_blocking=True),
            regime=batch["x_regime"].to(device, non_blocking=True),
        )

    def _forward_and_compute_loss(
        self,
        model: Flow_FEDformer,
        tensors: BatchTensors,
        scaler: torch.amp.GradScaler | None,
        accumulation_steps: int,
    ) -> torch.Tensor | None:
        """Rutina unificada para saltos (forwards) penalizados para auto-retorno acumulativo."""
        enabled = scaler.is_enabled() if scaler else False
        with autocast(device_type=device.type, enabled=enabled):
            dist = model(tensors.encoder, tensors.decoder, tensors.regime)
            loss = self._nll_loss(dist, tensors.target) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                "Caída de pérdida (loss) detectada a infinito o vacío (NaN). Omitiendo inferencia en bloque."
            )
            return None

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss

    @staticmethod
    def _should_step(
        batch_idx: int, total_batches: int, accumulation_steps: int
    ) -> bool:
        """Verifica estáticamente cuándo los acumuladores exigen salto optimizador."""
        return (batch_idx + 1) % accumulation_steps == 0 or (
            batch_idx + 1
        ) == total_batches

    @staticmethod
    def _optimizer_step(
        optimizer: Optimizer,
        scaler: torch.amp.GradScaler | None,
        model: nn.Module,
        clip_norm: float = 1.0,
    ) -> None:
        """Asienta saltos en optimizer, validando clípeos dinámicos en pesos (Scale steps)."""
        if scaler:
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer, total_epochs: int
    ) -> Any:
        """Crea el scheduler de LR según la configuración."""

        stype = self.config.scheduler_type
        if stype == "none":
            return None
        if stype == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_epochs, 1),
                eta_min=self.config.min_lr,
            )
        if stype == "cosine_warmup":
            import math  # pylint: disable=import-outside-toplevel

            warmup = self.config.warmup_epochs
            lr_min_ratio = self.config.min_lr / max(self.config.learning_rate, 1e-10)

            def lr_lambda(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / max(warmup, 1)
                progress = (epoch - warmup) / max(total_epochs - warmup, 1)
                return max(lr_min_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return None

    def _build_training_components(
        self,
        train_subset: Subset,
        test_subset: Subset,
        fold_idx: int,
        val_subset: Subset | None = None,
    ) -> TrainingComponents:
        """Instancia un bloque paramétrico ciego de modelo y loaders preajustados."""
        train_loader, test_loader = self._prepare_data_loaders(
            train_subset, test_subset
        )
        val_loader = self._make_loader(val_subset) if val_subset is not None else None
        model = self._get_model()
        self._maybe_load_finetune_checkpoint(model)

        if self.config.freeze_backbone:
            self._apply_freeze_backbone(model)

        lr = (
            self.config.finetune_lr
            if self.config.finetune_from and self.config.finetune_lr is not None
            else self.config.learning_rate
        )

        trainable_params = list(self._trainable_params(model))
        if not trainable_params:
            raise RuntimeError(
                "Excepción bloqueante: No hay parámetros en el modelo que reescribir con optimizador."
            )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.config.weight_decay,
            eps=1e-8,
        )

        scaler = (
            torch.amp.GradScaler(
                "cuda", enabled=self.config.use_amp and device.type == "cuda"
            )
            if self.config.use_amp
            else None
        )

        scheduler = self._create_scheduler(optimizer, self.config.n_epochs_per_fold)

        return TrainingComponents(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            test_loader=test_loader,
            fold=fold_idx,
            scheduler=scheduler,
            val_loader=val_loader,
        )

    def _maybe_load_finetune_checkpoint(self, model: Flow_FEDformer) -> None:
        """Rutina warm-up de subpesos extraída desde los dicts de persistencia."""
        ckpt_path = self.config.finetune_from
        if not ckpt_path:
            return

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Archivo de warm-start no encontrado en el sistema: {ckpt_path}"
            )

        import numpy._core.multiarray as _npcma  # pylint: disable=import-outside-toplevel

        with torch.serialization.safe_globals(
            [_npcma.scalar, np.float64, np.float32, np.int64, np.int32, np.bool_]  # pylint: disable=no-member
        ):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Pesos ajustados precargados mediante finetuning de %s (Perdidos=%d, Inesperados=%d)",
            ckpt_path,
            len(missing),
            len(unexpected),
        )

    def _apply_freeze_backbone(self, model: Flow_FEDformer) -> None:
        """Congela componentes estructurales preservando conectivas activas."""
        for param in model.parameters():
            param.requires_grad = False

        for flow in model.flows:
            for param in flow.parameters():
                param.requires_grad = True

        trainable_component_keys = {
            "flow_conditioner_proj",
            "enc_embedding",
            "dec_embedding",
            "regime_embedding",
        }
        for key, module in model.components.items():
            if key in trainable_component_keys:
                for param in module.parameters():
                    param.requires_grad = True

        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True

    @staticmethod
    def _trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
        """Itera parámetros que no fueron encapsulados intencionalmente estáticos."""
        for param in model.parameters():
            if param.requires_grad:
                yield param

    def save_checkpoint(
        self,
        components: TrainingComponents,
        epoch: int,
        loss: float,
        best: bool = False,
    ) -> Path:
        """Punto de auto-restauración atómica tras cada cierre heurístico favorable."""
        checkpoint = {
            "model_state_dict": components.model.state_dict(),
            "optimizer_state_dict": components.optimizer.state_dict(),
            "scaler_state_dict": components.scaler.state_dict()
            if components.scaler
            else None,
            "epoch": int(epoch),
            "fold": int(components.fold),
            "loss": float(loss),
            "config": asdict(self.config),
        }

        if best:
            path = self.checkpoint_dir / f"best_model_fold_{components.fold}.pt"
        else:
            path = self.checkpoint_dir / (
                f"checkpoint_fold_{components.fold}_epoch_{epoch}.pt"
            )

        torch.save(checkpoint, path)
        logger.info("Persistencia matemática salvada en %s", path)
        return path

    def load_checkpoint(
        self,
        model: Flow_FEDformer,
        optimizer: Optimizer,
        scaler: torch.amp.GradScaler | None,
        checkpoint_path: str,
    ) -> tuple[int, int, float]:
        """Transbordo temporal desde dict persistido devuelta a RAM GPU."""
        import numpy._core.multiarray as _npcma  # pylint: disable=import-outside-toplevel

        with torch.serialization.safe_globals(
            [_npcma.scalar, np.float64, np.float32, np.int64, np.int32, np.bool_]  # pylint: disable=no-member
        ):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scaler and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info("Modelo recuperado desde respaldo histórico en %s", checkpoint_path)
        return checkpoint["epoch"], checkpoint["fold"], checkpoint["loss"]

    def _nll_loss(self, dist: Distribution, y_true: torch.Tensor) -> torch.Tensor:
        """Log-Probabilidad marginal negativa computada para la simulación iterativa."""
        try:
            log_prob = dist.log_prob(y_true)
            log_prob = torch.clamp(log_prob, min=-1e6, max=1e2)
            return -log_prob.mean()
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "Tubería probabilística decaída: %s. Aplicando función costo residual asintótica MSE.",
                exc,
            )
            return F.mse_loss(dist.mean, y_true)

    def _initialize_wandb(self) -> None:
        """Resuelve interconexiones de monitoreo experimental W&B."""
        try:
            if wandb is None:
                logger.info(
                    "No detectado entorno nativo de Weights & Biases. Procediendo sin bitácora externa."
                )
                self.wandb_run = None
                return

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                name=self.config.wandb_run_name,
                reinit=True,
            )
            logger.info("Vínculo seguro creado junto a infraestructura de W&B")
        except (WandbError, ValueError) as exc:
            logger.warning(
                "Falló la comunicación teleoperada (W&B): %s. Bloqueando reportes asíncronos.",
                exc,
            )
            self.wandb_run = None

    def _train_epoch(self, components: TrainingComponents, epoch: int) -> float:
        """Gestiona un fragmento totalitario del fold a lo largo del subsistema."""
        components.model.train()
        epoch_losses: list[float] = []
        accumulation_steps = self.config.gradient_accumulation_steps
        total_batches = len(components.train_loader)

        for batch_idx, batch in enumerate(components.train_loader):
            try:
                tensors = self._prepare_batch(batch)
                loss = self._forward_and_compute_loss(
                    components.model,
                    tensors,
                    components.scaler,
                    accumulation_steps,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.error(
                        "Desbordamiento fatal detectado (GPU OOM). Reduzca batch_size, reinicie memoria VRAM interna."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise

                logger.warning(
                    "Operación en bloque [%s] inestable sobre fold [%s]: %s. Avanzando a recuperación cíclica.",
                    batch_idx,
                    components.fold,
                    exc,
                )
                continue

            if loss is None:
                continue

            loss_value = float(loss.item() * accumulation_steps)

            if self._should_step(batch_idx, total_batches, accumulation_steps):
                self._optimizer_step(
                    components.optimizer,
                    components.scaler,
                    components.model,
                    self.config.gradient_clip_norm,
                )

            epoch_losses.append(loss_value)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        logger.info(
            "  Ciclo Epocal %s/%s, Loss Base: %.4f",
            epoch + 1,
            self.config.n_epochs_per_fold,
            avg_loss,
        )
        self.metrics_tracker.log_metrics(
            {"train_loss": avg_loss}, epoch, fold=components.fold - 1
        )
        return avg_loss

    def _eval_epoch(self, model: Flow_FEDformer, val_loader: DataLoader) -> float:
        """Evalúa el modelo sobre el conjunto de validación sin actualizar gradientes.

        Retorna la pérdida NLL promedio sobre todos los batches. Si no hay batches
        válidos, retorna inf para que el early stopper no registre una mejora falsa.
        """
        import math  # pylint: disable=import-outside-toplevel

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                try:
                    tensors = self._prepare_batch(batch)
                    dist = model(tensors.encoder, tensors.decoder, tensors.regime)
                    loss_val = float(self._nll_loss(dist, tensors.target).item())
                    if math.isfinite(loss_val):
                        val_losses.append(loss_val)
                except (RuntimeError, ValueError) as exc:
                    logger.warning("Batch de validación descartado: %s", exc)
        return float(np.mean(val_losses)) if val_losses else float("inf")

    def _evaluate_model(
        self, model: Flow_FEDformer, test_loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gestiona el pase lógico de evaluación probabilística sin contaminación cruzada."""
        model.eval()
        fold_preds: list[np.ndarray] = []
        fold_gt: list[np.ndarray] = []
        fold_samples: list[np.ndarray] = []
        fold_quantiles: list[np.ndarray] = []

        with torch.inference_mode():
            for batch in test_loader:
                try:
                    samples = mc_dropout_inference(
                        model,
                        batch,
                        n_samples=self.config.mc_dropout_eval_samples,
                        use_flow_sampling=True,
                        mc_batch_size=10,
                    )
                    # Agregar cuantiles en CPU evita la ruta no determinista de
                    # torch.median(..., dim=0) en CUDA cuando los tests fuerzan
                    # torch.use_deterministic_algorithms(True).
                    samples_cpu = samples.detach().to("cpu", dtype=torch.float32)
                    quantiles_cpu = torch.quantile(
                        samples_cpu,
                        q=torch.tensor(
                            DEFAULT_QUANTILE_LEVELS.tolist(), dtype=torch.float32
                        ),
                        dim=0,
                    )
                    fold_samples.append(samples_cpu.numpy())
                    fold_quantiles.append(quantiles_cpu.numpy())
                    fold_preds.append(quantiles_cpu[1].numpy())  # p50
                    fold_gt.append(batch["y_true"].cpu().numpy())
                except (RuntimeError, ValueError) as exc:
                    logger.warning(
                        "Bloque estocástico de evaluador corrompido: %s", exc
                    )
                    continue

        if fold_preds and fold_gt and fold_samples and fold_quantiles:
            return (
                np.concatenate(fold_preds, axis=0),
                np.concatenate(fold_gt, axis=0),
                np.concatenate(fold_samples, axis=1),
                np.concatenate(fold_quantiles, axis=1),
            )

        logger.warning(
            "No se validaron proyecciones aptas en el subsistema post-evaluador."
        )
        return np.array([]), np.array([]), np.array([]), np.array([])

    def _populate_buffer_from_loader(self, loader: DataLoader) -> None:
        """Alimenta el buffer con las ventanas del fold recién entrenado."""
        for batch in loader:
            self.rehearsal_buffer.add_batch(batch)  # type: ignore[union-attr]

    def _rehearsal_step(self, components: TrainingComponents) -> None:
        """Un paso de replay: forward+backward con LR reducido sobre muestras históricas."""
        settings = self.config.sections.training.rehearsal
        batch = self.rehearsal_buffer.sample(k=self.config.batch_size)  # type: ignore[union-attr]
        if batch is None:
            return

        # Reducir LR temporalmente para el paso de rehearsal
        original_lrs = [pg["lr"] for pg in components.optimizer.param_groups]
        for pg in components.optimizer.param_groups:
            pg["lr"] *= settings.rehearsal_lr_mult

        try:
            tensors = self._prepare_batch(batch)
            loss = self._forward_and_compute_loss(
                components.model,
                tensors,
                components.scaler,
                accumulation_steps=1,
            )
            if loss is not None:
                self._optimizer_step(
                    components.optimizer,
                    components.scaler,
                    components.model,
                    self.config.gradient_clip_norm,
                )
        except RuntimeError as exc:
            logger.warning("Rehearsal step fallido: %s", exc)
        finally:
            # Siempre restaurar LR original
            for pg, lr in zip(components.optimizer.param_groups, original_lrs):
                pg["lr"] = lr

    def _run_single_fold(
        self,
        fold_idx: int,
        split_size: int,
        total_size: int,
        total_folds: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Ciclo acoplado de Entrenamiento/Validación sobre un salto iterativo delimitado (Leak Tolerant)."""
        train_end_idx = fold_idx * split_size
        test_end_idx = min((fold_idx + 1) * split_size, total_size)

        if test_end_idx - train_end_idx < self.config.seq_len + self.config.pred_len:
            logger.warning(
                "Insuficiente densidad de datos subyacentes operando sobre fold %s. Suspendido.",
                fold_idx,
            )
            return None

        logger.info(
            "--- Fold Analítico %s/%s: Entrenamiento precomputado [%s], Predicción empírica [%s, %s] ---",
            fold_idx,
            total_folds,
            train_end_idx,
            train_end_idx,
            test_end_idx,
        )

        self.full_dataset.refit_for_cutoff(train_end_idx)

        train_indices, test_indices = self._build_fold_indices(
            train_end_idx, test_end_idx
        )
        if not train_indices:
            logger.warning(
                "Secuencia histórica insuficiente procesando fold transitorio %s post-límites.",
                fold_idx,
            )
            return None
        if not test_indices:
            logger.warning(
                "Secuencia de ensayo colapsada o nula operando fold activo %s.",
                fold_idx,
            )
            return None

        # Reservar el final del bloque train como validación intra-fold (respeta causalidad temporal)
        val_subset: Subset | None = None
        if self.config.val_fraction > 0 and len(train_indices) > 2:
            val_size = max(1, int(len(train_indices) * self.config.val_fraction))
            val_indices = train_indices[-val_size:]
            train_indices = train_indices[:-val_size]
            val_subset = Subset(self.full_dataset, val_indices)

        train_subset = Subset(self.full_dataset, train_indices)
        test_subset = Subset(self.full_dataset, test_indices)

        # Reseed por fold: init del modelo independiente del RNG acumulado de folds previos
        torch.manual_seed(self.config.seed + fold_idx)
        np.random.seed(self.config.seed + fold_idx)

        components = self._build_training_components(
            train_subset, test_subset, fold_idx, val_subset=val_subset
        )

        loop_cfg = self.config.sections.training.loop
        # Para monitor_mode="max", negamos el valor: early stopping siempre minimiza
        monitor_sign = -1.0 if loop_cfg.monitor_mode == "max" else 1.0
        best_effective = float("inf")
        early_stopper = _EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )
        for epoch in range(self.config.n_epochs_per_fold):
            avg_loss = self._train_epoch(components, epoch)

            # Rehearsal replay tras cada época (solo si el buffer tiene datos de folds previos)
            if self.rehearsal_buffer is not None and self.rehearsal_buffer.is_ready:
                n_replay = self.config.sections.training.rehearsal.rehearsal_epochs
                for _ in range(n_replay):
                    self._rehearsal_step(components)

            # Evaluar en validación para early stopping; si no hay val_loader, usar train_loss
            train_m: dict[str, float] = {"loss": avg_loss}
            val_m: dict[str, float] | None = None
            if components.val_loader is not None:
                val_loss_raw = self._eval_epoch(components.model, components.val_loader)
                val_m = {"loss": val_loss_raw}
                logger.info(
                    "  Val Loss época %s/%s: %.4f",
                    epoch + 1,
                    self.config.n_epochs_per_fold,
                    val_loss_raw,
                )
                self.metrics_tracker.log_metrics(
                    {"val_loss": val_loss_raw}, epoch, fold=fold_idx - 1
                )

            monitor_loss = self._select_monitor_value(
                train_m, val_m, loop_cfg.monitor_metric
            )
            effective_val = monitor_sign * monitor_loss

            if effective_val < best_effective:
                best_effective = effective_val
                self.save_checkpoint(components, epoch, monitor_loss, best=True)

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(components, epoch, avg_loss, best=False)

            # Avanzar el scheduler de tasa de aprendizaje si existe
            if components.scheduler is not None:
                components.scheduler.step()

            if self.wandb_run:
                log_data = {"train_loss": avg_loss, "epoch": epoch, "fold": fold_idx}
                if val_m is not None:
                    log_data["val_loss"] = val_m["loss"]
                self.wandb_run.log(log_data)

            # Verificar parada anticipada usando el valor efectivo (normalizado por mode)
            if early_stopper.step(effective_val):
                logger.warning(
                    "Parada anticipada activada en época %s/%s para fold %s (patience=%s).",
                    epoch + 1,
                    self.config.n_epochs_per_fold,
                    fold_idx,
                    self.config.patience,
                )
                # Recargar el mejor checkpoint guardado para este fold
                best_ckpt_path = self.checkpoint_dir / f"best_model_fold_{fold_idx}.pt"
                if best_ckpt_path.exists():
                    self.load_checkpoint(
                        components.model,
                        components.optimizer,
                        components.scaler,
                        str(best_ckpt_path),
                    )
                break

        # Siempre restaurar el mejor checkpoint antes de inferencia (independientemente
        # de si el early stopping llegó a disparar o el training agotó las épocas).
        best_ckpt_path = self.checkpoint_dir / f"best_model_fold_{fold_idx}.pt"
        if best_ckpt_path.exists():
            self.load_checkpoint(
                components.model,
                components.optimizer,
                components.scaler,
                str(best_ckpt_path),
            )

        # Poblar buffer con ventanas del fold recién entrenado (para replay en folds futuros)
        if self.rehearsal_buffer is not None:
            self._populate_buffer_from_loader(components.train_loader)

        fold_preds, fold_gt, fold_samples, fold_quantiles = self._evaluate_model(
            components.model, components.test_loader
        )

        if fold_preds.size == 0:
            logger.warning(
                "Fold %s vacío con salida divergente de predicción", fold_idx
            )
            return None

        if self.wandb_run:
            self.wandb_run.log({"fold": fold_idx, "fold_completed": True})

        # GC Force clean per iteration
        del components, train_subset, test_subset, train_indices, test_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return fold_preds, fold_gt, fold_samples, fold_quantiles

    def _build_fold_indices(
        self, train_end_idx: int, test_end_idx: int
    ) -> tuple[list[int], list[int]]:
        """Aisla perimetralmente vectores que pudiesen provocar fallos ciegos o leaks temporales."""
        train_max_start = train_end_idx - self.config.seq_len - self.config.pred_len
        if train_max_start < 0:
            return [], []
        train_indices = list(range(train_max_start + 1))

        test_max_start = test_end_idx - self.config.seq_len - self.config.pred_len
        if test_max_start < train_end_idx:
            return train_indices, []

        # n_ventanas == total_filas - seq_len - pred_len + 1, consistente con run_backtest
        n_ventanas = (
            len(self.full_dataset.full_data_scaled)
            - self.config.seq_len
            - self.config.pred_len
            + 1
        )
        test_limit = min(test_max_start + 1, n_ventanas)
        test_start = max(0, train_end_idx - self.config.seq_len)
        if test_start >= test_limit:
            return train_indices, []

        test_indices = list(range(test_start, test_limit))
        return train_indices, test_indices

    def _inverse_transform_array(self, values_scaled: np.ndarray) -> np.ndarray:
        """Invierte el escalado de cualquier tensor con targets en el último eje."""
        pipeline = self.full_dataset.preprocessor

        orig_shape = values_scaled.shape
        values_flat = values_scaled.reshape(-1, orig_shape[-1])
        values_unscaled = pipeline.inverse_transform_targets(
            values_flat, self.config.target_features
        ).reshape(orig_shape)

        # Si return_transform != "none" y metric_space == "prices", reconstruir precios
        if pipeline.return_transform != "none" and self.config.metric_space == "prices":
            last_prices = np.array(
                [pipeline.last_prices.get(t, 1.0) for t in self.config.target_features]
            )
            return _cumulative_returns_to_prices(
                values_unscaled, last_prices, pipeline.return_transform
            )

        return values_unscaled

    def _inverse_transform_all(
        self,
        preds_scaled: np.ndarray,
        gt_scaled: np.ndarray,
        samples_scaled: np.ndarray,
        quantiles_scaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aplica inverse_transform_targets y, si aplica, reconstrucción a precios."""
        return (
            self._inverse_transform_array(preds_scaled),
            self._inverse_transform_array(gt_scaled),
            self._inverse_transform_array(samples_scaled),
            self._inverse_transform_array(quantiles_scaled),
        )

    @staticmethod
    def _select_monitor_value(
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
        monitor_metric: str,
    ) -> float:
        """Selecciona el valor escalar a monitorear para early stopping y checkpoint.

        Args:
            train_metrics: Métricas de entrenamiento de la época actual (clave "loss").
            val_metrics: Métricas de validación de la época actual (clave "loss", etc.)
                o None si no hay split de validación.
            monitor_metric: Nombre de la métrica a monitorear.

        Returns:
            Valor escalar a optimizar (sin aplicar el signo de monitor_mode).
        """
        val = val_metrics or {}
        if monitor_metric == "val_loss":
            return val.get("loss", train_metrics["loss"])
        if monitor_metric == "val_pinball_p50":
            if "pinball_p50" in val:
                return val["pinball_p50"]
            logger.warning(
                "val_pinball_p50 no disponible en val_metrics, usando val_loss como fallback"
            )
            return val.get("loss", train_metrics["loss"])
        if monitor_metric == "val_coverage_80":
            if "coverage_80" in val:
                return val["coverage_80"]
            logger.warning(
                "val_coverage_80 no disponible en val_metrics, usando val_loss como fallback"
            )
            return val.get("loss", train_metrics["loss"])
        if monitor_metric == "composite":
            val_loss = val.get("loss", train_metrics["loss"])
            pinball = val.get("pinball_p50", val_loss)
            return 0.5 * val_loss + 0.5 * pinball
        # Fallback seguro (no debería llegar aquí tras validación en config)
        return val.get("loss", train_metrics["loss"])

    def _compute_fold_probabilistic_metrics(
        self,
        gt_real: np.ndarray,
        quantiles_real: np.ndarray,
        samples_real: np.ndarray,
    ) -> dict[str, float]:
        """Calcula métricas probabilísticas para un fold tras la inversión de escala.

        Args:
            gt_real: Ground truth en espacio real, shape (n_windows, pred_len, n_targets).
            quantiles_real: Cuantiles en espacio real, shape (n_q, n_windows, pred_len, n_targets).
            samples_real: Muestras en espacio real, shape (n_samples, n_windows, pred_len, n_targets).

        Returns:
            Dict con claves: pinball_p10, pinball_p50, pinball_p90,
            coverage_80, interval_width_80, crps.
        """
        levels = DEFAULT_QUANTILE_LEVELS  # [0.1, 0.5, 0.9]
        metrics: dict[str, float] = {}

        # Pinball loss por cuantil
        metrics.update(multi_quantile_pinball_loss(gt_real, quantiles_real, levels))

        # Cobertura del intervalo 80% (p10 a p90)
        try:
            metrics["coverage_80"] = coverage_by_quantile_pair(
                gt_real, quantiles_real, levels, 0.1, 0.9
            )
        except ValueError:
            logger.warning(
                "No se pudo calcular coverage_80: niveles p10/p90 no disponibles"
            )
            metrics["coverage_80"] = float("nan")

        # Anchura del intervalo 80%
        try:
            idx_10 = int(np.argmin(np.abs(levels.astype(float) - 0.1)))
            idx_90 = int(np.argmin(np.abs(levels.astype(float) - 0.9)))
            metrics["interval_width_80"] = interval_width(
                quantiles_real[idx_10], quantiles_real[idx_90]
            )
        except (IndexError, ValueError):
            metrics["interval_width_80"] = float("nan")

        # CRPS
        metrics["crps"] = crps_from_samples(gt_real, samples_real)

        # También coverage_gap vía calibration_gap (no renombrado para evitar redundancia)
        for key, val in calibration_gap(gt_real, quantiles_real, levels).items():
            metrics[key] = val

        return metrics

    def run_backtest(self, n_splits: int = 5) -> ForecastOutput:
        """Ejecuta el backtest walk-forward completo y agrega los folds.

        Args:
            n_splits: Numero total de particiones temporales a evaluar.

        Returns:
            `ForecastOutput` agregado con predicciones, muestras, cuantiles y
            metadatos por ventana en todos los folds validos.
        """
        self._initialize_wandb()
        # Reiniciar métricas probabilísticas por fold para evitar acumulación entre runs
        self.fold_probabilistic_metrics = []

        # Usar filas crudas (no ventanas) para que train_end_idx sea un índice
        # de fila consistente con _build_fold_indices y refit_for_cutoff.
        total_size = len(self.full_dataset.full_data_scaled)
        split_size = max(
            total_size // n_splits, self.config.seq_len + self.config.pred_len
        )
        total_folds = max(1, n_splits - 1)

        all_preds: list[np.ndarray] = []
        all_gt: list[np.ndarray] = []
        all_samples: list[np.ndarray] = []
        all_quantiles: list[np.ndarray] = []
        all_fold_ids: list[np.ndarray] = []
        # Resultados en espacio real, invertidos con el scaler propio de cada fold
        all_preds_real: list[np.ndarray] = []
        all_gt_real: list[np.ndarray] = []
        all_samples_real: list[np.ndarray] = []
        all_quantiles_real: list[np.ndarray] = []

        try:
            for fold_idx in range(1, n_splits):
                fold_outputs = self._run_single_fold(
                    fold_idx, split_size, total_size, total_folds
                )
                if not fold_outputs:
                    continue

                preds, gt, samples, quantiles = fold_outputs
                all_preds.append(preds)
                all_gt.append(gt)
                all_samples.append(samples)
                all_quantiles.append(quantiles)
                all_fold_ids.append(np.full(len(preds), fold_idx, dtype=np.int32))

                # Invertir con el scaler activo en ESTE fold, antes de que el siguiente
                # fold llame a refit_for_cutoff y lo sobreescriba.
                preds_r, gt_r, samples_r, quantiles_r = self._inverse_transform_all(
                    preds, gt, samples, quantiles
                )
                all_preds_real.append(preds_r)
                all_gt_real.append(gt_r)
                all_samples_real.append(samples_r)
                all_quantiles_real.append(quantiles_r)

                # Métricas probabilísticas del fold actual (en espacio real)
                fold_prob_metrics = self._compute_fold_probabilistic_metrics(
                    gt_r, quantiles_r, samples_r
                )
                self.fold_probabilistic_metrics.append(fold_prob_metrics)
                self.metrics_tracker.log_metrics(
                    fold_prob_metrics, step=fold_idx - 1, fold=fold_idx - 1
                )

        except (RuntimeError, ValueError):
            logger.exception(
                "Corrupción general forzando colapso del Backtest iterativo walk-forward"
            )
            raise
        finally:
            if self.wandb_run:
                self.wandb_run.finish()

        if not all_preds:
            logger.error(
                "No existió ni un sólo bloque precalculado de inferencias. Colapso."
            )
            empty = np.array([])
            empty_quantiles = np.empty(
                (
                    len(DEFAULT_QUANTILE_LEVELS),
                    0,
                    self.config.pred_len,
                    len(self.config.target_features),
                ),
                dtype=np.float32,
            )
            return ForecastOutput(
                preds_scaled=empty,
                gt_scaled=empty,
                samples_scaled=empty,
                preds_real=empty,
                gt_real=empty,
                samples_real=empty,
                quantiles_scaled=empty_quantiles,
                quantiles_real=empty_quantiles.copy(),
                quantile_levels=DEFAULT_QUANTILE_LEVELS.copy(),
                metric_space=self.config.metric_space,
                return_transform=self.config.sections.preprocessing.return_transform,
                target_names=list(self.config.target_features),
                window_fold_ids=empty.astype(np.int32),
            )

        preds_scaled = np.concatenate(all_preds, axis=0)
        gt_scaled = np.concatenate(all_gt, axis=0)
        samples_scaled = np.concatenate(all_samples, axis=1)
        quantiles_scaled = np.concatenate(all_quantiles, axis=1)
        window_fold_ids = np.concatenate(all_fold_ids, axis=0)

        preds_real = np.concatenate(all_preds_real, axis=0)
        gt_real = np.concatenate(all_gt_real, axis=0)
        samples_real = np.concatenate(all_samples_real, axis=1)
        quantiles_real = np.concatenate(all_quantiles_real, axis=1)

        return ForecastOutput(
            preds_scaled=preds_scaled,
            gt_scaled=gt_scaled,
            samples_scaled=samples_scaled,
            preds_real=preds_real,
            gt_real=gt_real,
            samples_real=samples_real,
            quantiles_scaled=quantiles_scaled,
            quantiles_real=quantiles_real,
            quantile_levels=DEFAULT_QUANTILE_LEVELS.copy(),
            metric_space=self.config.metric_space,
            return_transform=self.config.sections.preprocessing.return_transform,
            target_names=list(self.config.target_features),
            window_fold_ids=window_fold_ids,
        )
