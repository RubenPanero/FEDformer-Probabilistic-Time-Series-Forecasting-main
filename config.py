# -*- coding: utf-8 -*-
"""
Configuracion del sistema Vanguard FEDformer.
"""

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set

import logging

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SequenceSettings:
    """Sequence segmentation lengths for encoder/decoder."""

    seq_len: int = 10
    label_len: int = 5
    pred_len: int = 5


@dataclass
class TransformerSettings:
    """Transformer backbone hyper-parameters."""

    # pylint: disable=too-many-instance-attributes

    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    modes: int = 64
    activation: str = "gelu"
    dropout: float = 0.1


@dataclass
class DecompositionSettings:
    """Moving average kernel configuration for seasonal/trend split."""

    moving_avg: Optional[List[int]] = None


@dataclass
class RegimeSettings:
    """Latent regime embedding configuration."""

    n_regimes: int = 3
    regime_embedding_dim: int = 16


@dataclass
class FlowSettings:
    """Normalizing flow depth and hidden size."""

    n_flow_layers: int = 4
    flow_hidden_dim: int = 64


@dataclass
class ModelSettings:
    """Grouped model-related settings."""

    sequence: SequenceSettings = field(default_factory=SequenceSettings)
    transformer: TransformerSettings = field(default_factory=TransformerSettings)
    decomposition: DecompositionSettings = field(default_factory=DecompositionSettings)
    regime: RegimeSettings = field(default_factory=RegimeSettings)
    flow: FlowSettings = field(default_factory=FlowSettings)


@dataclass
class OptimizationSettings:
    """Optimizer-level hyper-parameters."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    scheduler_type: str = "none"  # opciones: "none", "cosine", "cosine_warmup"


_ALLOWED_MONITOR_METRICS: frozenset[str] = frozenset(
    {"val_loss", "val_pinball_p50", "val_coverage_80", "composite"}
)
_ALLOWED_MONITOR_MODES: frozenset[str] = frozenset({"min", "max"})


@dataclass
class LoopSettings:
    """Training loop batch/epoch controls."""

    n_epochs_per_fold: int = 20
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    gradient_clip_norm: float = 1.0
    patience: int = 5
    min_delta: float = 5e-3
    val_fraction: float = (
        0.15  # fracción del bloque train reservada para validación intra-fold
    )
    monitor_metric: str = (
        "val_loss"  # val_loss | val_pinball_p50 | val_coverage_80 | composite
    )
    monitor_mode: str = "min"  # min | max

    def __post_init__(self) -> None:
        if self.monitor_metric not in _ALLOWED_MONITOR_METRICS:
            raise ValueError(
                f"monitor_metric='{self.monitor_metric}' inválido. "
                f"Valores permitidos: {sorted(_ALLOWED_MONITOR_METRICS)}"
            )
        if self.monitor_mode not in _ALLOWED_MONITOR_MODES:
            raise ValueError(
                f"monitor_mode='{self.monitor_mode}' inválido. "
                f"Valores permitidos: {sorted(_ALLOWED_MONITOR_MODES)}"
            )


@dataclass
class RuntimeSettings:
    """Runtime toggles for training."""

    use_amp: bool = True
    use_gradient_checkpointing: bool = False
    mc_dropout_eval_samples: int = 20
    pin_memory: bool = False
    num_workers: Optional[int] = None
    compile_mode: str = "max-autotune"
    finetune_from: Optional[str] = None
    freeze_backbone: bool = False
    finetune_lr: Optional[float] = None


@dataclass
class RehearsalSettings:
    """Configuración del rehearsal buffer para continual learning."""

    enabled: bool = False
    buffer_size: int = 1000  # capacidad máxima (nº ventanas individuales)
    rehearsal_epochs: int = 1  # pasos de replay por época de entrenamiento
    rehearsal_lr_mult: float = 0.1  # multiplicador de LR para pasos de replay


@dataclass
class TrainingSettings:
    """Grouped training-related settings."""

    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    loop: LoopSettings = field(default_factory=LoopSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    rehearsal: RehearsalSettings = field(default_factory=RehearsalSettings)


@dataclass
class MonitoringSettings:
    """External monitoring/metadata options (W&B, dataset columns)."""

    wandb_project: str = "vanguard-fedformer-flow"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    date_column: Optional[str] = None


@dataclass
class PreprocessingSettings:
    """Configuration for reusable, leakage-safe preprocessing."""

    feature_roles: Dict[str, str] = field(default_factory=dict)
    scaling_strategy: str = "robust"
    missing_policy: str = "impute_median"
    outlier_policy: str = "winsorize"
    fit_scope: str = "fold_train_only"
    persist_artifacts: bool = False
    drift_checks: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "allow_extra_columns": True,
            "null_rate_threshold": 0.2,
            "mean_shift_threshold": 6.0,
            "std_ratio_bounds": [0.2, 5.0],
        }
    )
    strict_mode: bool = True
    categorical_encoding: str = "none"
    time_features: List[str] = field(default_factory=list)
    artifact_dir: str = "reports/preprocessing"
    return_transform: str = "none"  # opciones: "none", "log_return", "simple_return"
    metric_space: str = "returns"  # opciones: "returns" | "prices"


@dataclass
class ReproSettings:
    """Reproducibility toggles and seed configuration."""

    seed: int = 42
    deterministic: bool = False


@dataclass
class DerivedSettings:
    """Derived values computed from the dataset headers."""

    enc_in: Optional[int] = None
    dec_in: Optional[int] = None
    c_out: Optional[int] = None


@dataclass
class ConfigSections:
    """Container for grouped configuration sections."""

    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    reproducibility: ReproSettings = field(default_factory=ReproSettings)
    derived: DerivedSettings = field(default_factory=DerivedSettings)


@dataclass(init=False)
class FEDformerConfig:
    """Enhanced configuration class with validation and better organization."""

    # pylint: disable=missing-function-docstring,too-many-public-methods,too-many-instance-attributes

    target_features: List[str]
    file_path: str
    sections: ConfigSections = field(init=False)

    _ALLOWED_KEYS: ClassVar[Set[str]] = {
        "seq_len",
        "label_len",
        "pred_len",
        "d_model",
        "n_heads",
        "e_layers",
        "d_layers",
        "d_ff",
        "modes",
        "moving_avg",
        "activation",
        "dropout",
        "n_regimes",
        "regime_embedding_dim",
        "n_flow_layers",
        "flow_hidden_dim",
        "learning_rate",
        "weight_decay",
        "n_epochs_per_fold",
        "batch_size",
        "use_amp",
        "use_gradient_checkpointing",
        "mc_dropout_eval_samples",
        "pin_memory",
        "num_workers",
        "gradient_accumulation_steps",
        "gradient_clip_norm",
        "compile_mode",
        "finetune_from",
        "freeze_backbone",
        "finetune_lr",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "date_column",
        "feature_roles",
        "scaling_strategy",
        "missing_policy",
        "outlier_policy",
        "fit_scope",
        "persist_artifacts",
        "drift_checks",
        "strict_mode",
        "categorical_encoding",
        "time_features",
        "artifact_dir",
        "seed",
        "deterministic",
        "return_transform",
        "metric_space",
        "warmup_epochs",
        "min_lr",
        "scheduler_type",
        "patience",
        "min_delta",
        "monitor_metric",
        "monitor_mode",
        "rehearsal_enabled",
        "rehearsal_buffer_size",
        "rehearsal_lr_mult",
    }

    def __init__(
        self,
        target_features: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Set defaults if not provided
        if file_path is None:
            # Dataset permanente NVDA como fallback por defecto
            file_path = os.path.join(
                os.path.dirname(__file__), "data", "NVDA_features.csv"
            )

        # Auto-detect target features if not provided
        if target_features is None:
            try:
                df_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
                # Try to find a price column
                for col in ["Close", "close", "Close_Price", "close_price"]:
                    if col in df_cols:
                        target_features = [col]
                        break
                # If no price column found, use first non-date column
                if target_features is None:
                    non_date_cols = [
                        col for col in df_cols if col.lower() not in ["date", "time"]
                    ]
                    if non_date_cols:
                        target_features = [non_date_cols[0]]
                    else:
                        target_features = [df_cols[0]]
            except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
                # Fallback if file cannot be read, is empty, or has parsing issues
                target_features = ["Close"]

        self.target_features = target_features
        self.file_path = file_path
        self.sections = ConfigSections()

        unexpected = set(kwargs) - self._ALLOWED_KEYS
        if unexpected:
            unexpected_str = ", ".join(sorted(unexpected))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected_str}")

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__post_init__()

    def __post_init__(self) -> None:
        """Sets derived configuration parameters and validates config"""
        if self.moving_avg is None:
            self.moving_avg = [24, 48]

        try:
            df_cols = pd.read_csv(self.file_path, nrows=0).columns
            if self.date_column and self.date_column in df_cols:
                feature_cols = [col for col in df_cols if col != self.date_column]
            else:
                feature_cols = list(df_cols)
            self.enc_in = len(feature_cols)
            self.dec_in = len(feature_cols)
            self.c_out = len(self.target_features)
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as exc:
            logger.error("Failed to read CSV file %s: %s", self.file_path, exc)
            raise

        max_modes = max(1, self.seq_len // 2)
        if self.modes > max_modes:
            logger.warning(
                "modes (%s) > seq_len//2 (%s), clamping modes to %s",
                self.modes,
                max_modes,
                max_modes,
            )
            self.modes = max_modes
        if self.modes < 1:
            logger.warning("modes (%s) < 1, setting to 1", self.modes)
            self.modes = 1

        self.validate(df_cols)

    def validate(self, df_columns: Optional[pd.Index] = None) -> None:
        """Validate configuration consistency"""
        if df_columns is None:
            df_columns = pd.read_csv(self.file_path, nrows=0).columns

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.label_len > self.seq_len:
            raise ValueError(
                f"label_len ({self.label_len}) cannot exceed seq_len ({self.seq_len})"
            )
        if not 1 <= self.modes <= max(1, self.seq_len // 2):
            raise ValueError(
                f"modes ({self.modes}) must be in [1, seq_len//2] ({self.seq_len // 2})"
            )
        if self.activation not in ["gelu", "relu"]:
            raise ValueError(
                f"activation must be 'gelu' or 'relu', got {self.activation}"
            )
        if not all(col in df_columns for col in self.target_features):
            raise ValueError("All target features must exist in the dataset")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {self.dropout}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"Learning rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"Weight decay must be non-negative, got {self.weight_decay}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.mc_dropout_eval_samples <= 0:
            raise ValueError(
                "mc_dropout_eval_samples must be positive, got "
                f"{self.mc_dropout_eval_samples}"
            )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "Gradient accumulation steps must be positive, got "
                f"{self.gradient_accumulation_steps}"
            )
        if self.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm debe ser positivo, got {self.gradient_clip_norm}"
            )
        if self.finetune_lr is not None and self.finetune_lr <= 0:
            raise ValueError(f"finetune_lr must be positive, got {self.finetune_lr}")
        if self.e_layers < 1 or self.d_layers < 1:
            raise ValueError("e_layers and d_layers must be >= 1")
        if self.scheduler_type not in ("none", "cosine", "cosine_warmup"):
            raise ValueError(
                f"scheduler_type debe ser 'none', 'cosine' o 'cosine_warmup', "
                f"got '{self.scheduler_type}'"
            )
        if self.scaling_strategy not in ["standard", "robust", "minmax", "none"]:
            raise ValueError(
                "scaling_strategy must be one of ['standard', 'robust', 'minmax', 'none']"
            )
        if self.missing_policy not in ["drop", "ffill_bfill", "impute_median", "error"]:
            raise ValueError(
                "missing_policy must be one of ['drop', 'ffill_bfill', 'impute_median', 'error']"
            )
        if self.outlier_policy not in ["clip", "winsorize", "none"]:
            raise ValueError(
                "outlier_policy must be one of ['clip', 'winsorize', 'none']"
            )
        if self.fit_scope not in ["global_train", "fold_train_only"]:
            raise ValueError("fit_scope must be 'global_train' or 'fold_train_only'")
        if self.categorical_encoding not in ["none", "ordinal", "onehot"]:
            raise ValueError(
                "categorical_encoding must be one of ['none', 'ordinal', 'onehot']"
            )
        if not isinstance(self.feature_roles, dict):
            raise ValueError("feature_roles must be a dictionary")
        if not isinstance(self.drift_checks, dict):
            raise ValueError("drift_checks must be a dictionary")
        if not isinstance(self.time_features, list):
            raise ValueError("time_features must be a list")
        if self.return_transform not in ("none", "log_return", "simple_return"):
            raise ValueError(
                f"return_transform debe ser 'none', 'log_return' o 'simple_return', got '{self.return_transform}'"
            )
        if self.metric_space not in ("returns", "prices"):
            raise ValueError(
                f"metric_space debe ser 'returns' o 'prices', got '{self.metric_space}'"
            )
        if self.pred_len % 2 != 0:
            logger.warning(
                "pred_len (%s) is odd. For affine coupling, even values are preferred",
                self.pred_len,
            )

    # -- Model settings proxies -------------------------------------------------
    @property
    def seq_len(self) -> int:
        return self.sections.model.sequence.seq_len

    @seq_len.setter
    def seq_len(self, value: int) -> None:
        self.sections.model.sequence.seq_len = value

    @property
    def label_len(self) -> int:
        return self.sections.model.sequence.label_len

    @label_len.setter
    def label_len(self, value: int) -> None:
        self.sections.model.sequence.label_len = value

    @property
    def pred_len(self) -> int:
        return self.sections.model.sequence.pred_len

    @pred_len.setter
    def pred_len(self, value: int) -> None:
        self.sections.model.sequence.pred_len = value

    @property
    def d_model(self) -> int:
        return self.sections.model.transformer.d_model

    @d_model.setter
    def d_model(self, value: int) -> None:
        self.sections.model.transformer.d_model = value

    @property
    def n_heads(self) -> int:
        return self.sections.model.transformer.n_heads

    @n_heads.setter
    def n_heads(self, value: int) -> None:
        self.sections.model.transformer.n_heads = value

    @property
    def e_layers(self) -> int:
        return self.sections.model.transformer.e_layers

    @e_layers.setter
    def e_layers(self, value: int) -> None:
        self.sections.model.transformer.e_layers = value

    @property
    def d_layers(self) -> int:
        return self.sections.model.transformer.d_layers

    @d_layers.setter
    def d_layers(self, value: int) -> None:
        self.sections.model.transformer.d_layers = value

    @property
    def d_ff(self) -> int:
        return self.sections.model.transformer.d_ff

    @d_ff.setter
    def d_ff(self, value: int) -> None:
        self.sections.model.transformer.d_ff = value

    @property
    def modes(self) -> int:
        return self.sections.model.transformer.modes

    @modes.setter
    def modes(self, value: int) -> None:
        self.sections.model.transformer.modes = value

    @property
    def moving_avg(self) -> Optional[List[int]]:
        return self.sections.model.decomposition.moving_avg

    @moving_avg.setter
    def moving_avg(self, value: Optional[List[int]]) -> None:
        self.sections.model.decomposition.moving_avg = value

    @property
    def activation(self) -> str:
        return self.sections.model.transformer.activation

    @activation.setter
    def activation(self, value: str) -> None:
        self.sections.model.transformer.activation = value

    @property
    def dropout(self) -> float:
        return self.sections.model.transformer.dropout

    @dropout.setter
    def dropout(self, value: float) -> None:
        self.sections.model.transformer.dropout = value

    @property
    def n_regimes(self) -> int:
        return self.sections.model.regime.n_regimes

    @n_regimes.setter
    def n_regimes(self, value: int) -> None:
        self.sections.model.regime.n_regimes = value

    @property
    def regime_embedding_dim(self) -> int:
        return self.sections.model.regime.regime_embedding_dim

    @regime_embedding_dim.setter
    def regime_embedding_dim(self, value: int) -> None:
        self.sections.model.regime.regime_embedding_dim = value

    @property
    def n_flow_layers(self) -> int:
        return self.sections.model.flow.n_flow_layers

    @n_flow_layers.setter
    def n_flow_layers(self, value: int) -> None:
        self.sections.model.flow.n_flow_layers = value

    @property
    def flow_hidden_dim(self) -> int:
        return self.sections.model.flow.flow_hidden_dim

    @flow_hidden_dim.setter
    def flow_hidden_dim(self, value: int) -> None:
        self.sections.model.flow.flow_hidden_dim = value

    # -- Training settings proxies --------------------------------------------
    @property
    def learning_rate(self) -> float:
        return self.sections.training.optimization.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.sections.training.optimization.learning_rate = value

    @property
    def weight_decay(self) -> float:
        return self.sections.training.optimization.weight_decay

    @weight_decay.setter
    def weight_decay(self, value: float) -> None:
        self.sections.training.optimization.weight_decay = value

    @property
    def n_epochs_per_fold(self) -> int:
        return self.sections.training.loop.n_epochs_per_fold

    @n_epochs_per_fold.setter
    def n_epochs_per_fold(self, value: int) -> None:
        self.sections.training.loop.n_epochs_per_fold = value

    @property
    def batch_size(self) -> int:
        return self.sections.training.loop.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.sections.training.loop.batch_size = value

    @property
    def use_amp(self) -> bool:
        return self.sections.training.runtime.use_amp

    @use_amp.setter
    def use_amp(self, value: bool) -> None:
        self.sections.training.runtime.use_amp = value

    @property
    def use_gradient_checkpointing(self) -> bool:
        return self.sections.training.runtime.use_gradient_checkpointing

    @use_gradient_checkpointing.setter
    def use_gradient_checkpointing(self, value: bool) -> None:
        self.sections.training.runtime.use_gradient_checkpointing = value

    @property
    def mc_dropout_eval_samples(self) -> int:
        return self.sections.training.runtime.mc_dropout_eval_samples

    @mc_dropout_eval_samples.setter
    def mc_dropout_eval_samples(self, value: int) -> None:
        self.sections.training.runtime.mc_dropout_eval_samples = value

    @property
    def pin_memory(self) -> bool:
        return self.sections.training.runtime.pin_memory

    @pin_memory.setter
    def pin_memory(self, value: bool) -> None:
        self.sections.training.runtime.pin_memory = value

    @property
    def num_workers(self) -> Optional[int]:
        return self.sections.training.runtime.num_workers

    @num_workers.setter
    def num_workers(self, value: Optional[int]) -> None:
        self.sections.training.runtime.num_workers = value

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.sections.training.loop.gradient_accumulation_steps

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, value: int) -> None:
        self.sections.training.loop.gradient_accumulation_steps = value

    @property
    def compile_mode(self) -> str:
        return self.sections.training.runtime.compile_mode

    @compile_mode.setter
    def compile_mode(self, value: str) -> None:
        self.sections.training.runtime.compile_mode = value

    @property
    def finetune_from(self) -> Optional[str]:
        return self.sections.training.runtime.finetune_from

    @finetune_from.setter
    def finetune_from(self, value: Optional[str]) -> None:
        self.sections.training.runtime.finetune_from = value

    @property
    def freeze_backbone(self) -> bool:
        return self.sections.training.runtime.freeze_backbone

    @freeze_backbone.setter
    def freeze_backbone(self, value: bool) -> None:
        self.sections.training.runtime.freeze_backbone = value

    @property
    def finetune_lr(self) -> Optional[float]:
        return self.sections.training.runtime.finetune_lr

    @finetune_lr.setter
    def finetune_lr(self, value: Optional[float]) -> None:
        self.sections.training.runtime.finetune_lr = value

    @property
    def warmup_epochs(self) -> int:
        return self.sections.training.optimization.warmup_epochs

    @warmup_epochs.setter
    def warmup_epochs(self, value: int) -> None:
        self.sections.training.optimization.warmup_epochs = value

    @property
    def min_lr(self) -> float:
        return self.sections.training.optimization.min_lr

    @min_lr.setter
    def min_lr(self, value: float) -> None:
        self.sections.training.optimization.min_lr = value

    @property
    def scheduler_type(self) -> str:
        return self.sections.training.optimization.scheduler_type

    @scheduler_type.setter
    def scheduler_type(self, value: str) -> None:
        self.sections.training.optimization.scheduler_type = value

    @property
    def patience(self) -> int:
        return self.sections.training.loop.patience

    @patience.setter
    def patience(self, value: int) -> None:
        self.sections.training.loop.patience = value

    @property
    def min_delta(self) -> float:
        return self.sections.training.loop.min_delta

    @min_delta.setter
    def min_delta(self, value: float) -> None:
        self.sections.training.loop.min_delta = value

    @property
    def val_fraction(self) -> float:
        return self.sections.training.loop.val_fraction

    @val_fraction.setter
    def val_fraction(self, value: float) -> None:
        self.sections.training.loop.val_fraction = value

    @property
    def gradient_clip_norm(self) -> float:
        return self.sections.training.loop.gradient_clip_norm

    @gradient_clip_norm.setter
    def gradient_clip_norm(self, value: float) -> None:
        self.sections.training.loop.gradient_clip_norm = value

    @property
    def monitor_metric(self) -> str:
        return self.sections.training.loop.monitor_metric

    @monitor_metric.setter
    def monitor_metric(self, value: str) -> None:
        if value not in _ALLOWED_MONITOR_METRICS:
            raise ValueError(
                f"monitor_metric='{value}' inválido. "
                f"Valores permitidos: {sorted(_ALLOWED_MONITOR_METRICS)}"
            )
        self.sections.training.loop.monitor_metric = value

    @property
    def monitor_mode(self) -> str:
        return self.sections.training.loop.monitor_mode

    @monitor_mode.setter
    def monitor_mode(self, value: str) -> None:
        if value not in _ALLOWED_MONITOR_MODES:
            raise ValueError(
                f"monitor_mode='{value}' inválido. "
                f"Valores permitidos: {sorted(_ALLOWED_MONITOR_MODES)}"
            )
        self.sections.training.loop.monitor_mode = value

    # -- Preprocessing settings proxies ---------------------------------------
    @property
    def feature_roles(self) -> Dict[str, str]:
        return self.sections.preprocessing.feature_roles

    @feature_roles.setter
    def feature_roles(self, value: Dict[str, str]) -> None:
        self.sections.preprocessing.feature_roles = value

    @property
    def scaling_strategy(self) -> str:
        return self.sections.preprocessing.scaling_strategy

    @scaling_strategy.setter
    def scaling_strategy(self, value: str) -> None:
        self.sections.preprocessing.scaling_strategy = value

    @property
    def missing_policy(self) -> str:
        return self.sections.preprocessing.missing_policy

    @missing_policy.setter
    def missing_policy(self, value: str) -> None:
        self.sections.preprocessing.missing_policy = value

    @property
    def outlier_policy(self) -> str:
        return self.sections.preprocessing.outlier_policy

    @outlier_policy.setter
    def outlier_policy(self, value: str) -> None:
        self.sections.preprocessing.outlier_policy = value

    @property
    def fit_scope(self) -> str:
        return self.sections.preprocessing.fit_scope

    @fit_scope.setter
    def fit_scope(self, value: str) -> None:
        self.sections.preprocessing.fit_scope = value

    @property
    def persist_artifacts(self) -> bool:
        return self.sections.preprocessing.persist_artifacts

    @persist_artifacts.setter
    def persist_artifacts(self, value: bool) -> None:
        self.sections.preprocessing.persist_artifacts = value

    @property
    def drift_checks(self) -> Dict[str, Any]:
        return self.sections.preprocessing.drift_checks

    @drift_checks.setter
    def drift_checks(self, value: Dict[str, Any]) -> None:
        self.sections.preprocessing.drift_checks = value

    @property
    def strict_mode(self) -> bool:
        return self.sections.preprocessing.strict_mode

    @strict_mode.setter
    def strict_mode(self, value: bool) -> None:
        self.sections.preprocessing.strict_mode = value

    @property
    def categorical_encoding(self) -> str:
        return self.sections.preprocessing.categorical_encoding

    @categorical_encoding.setter
    def categorical_encoding(self, value: str) -> None:
        self.sections.preprocessing.categorical_encoding = value

    @property
    def time_features(self) -> List[str]:
        return self.sections.preprocessing.time_features

    @time_features.setter
    def time_features(self, value: List[str]) -> None:
        self.sections.preprocessing.time_features = value

    @property
    def artifact_dir(self) -> str:
        return self.sections.preprocessing.artifact_dir

    @artifact_dir.setter
    def artifact_dir(self, value: str) -> None:
        self.sections.preprocessing.artifact_dir = value

    @property
    def return_transform(self) -> str:
        return self.sections.preprocessing.return_transform

    @return_transform.setter
    def return_transform(self, value: str) -> None:
        self.sections.preprocessing.return_transform = value

    @property
    def metric_space(self) -> str:
        return self.sections.preprocessing.metric_space

    @metric_space.setter
    def metric_space(self, value: str) -> None:
        self.sections.preprocessing.metric_space = value

    # -- Monitoring settings proxies -----------------------------------------
    @property
    def wandb_project(self) -> str:
        return self.sections.monitoring.wandb_project

    @wandb_project.setter
    def wandb_project(self, value: str) -> None:
        self.sections.monitoring.wandb_project = value

    @property
    def wandb_entity(self) -> Optional[str]:
        return self.sections.monitoring.wandb_entity

    @wandb_entity.setter
    def wandb_entity(self, value: Optional[str]) -> None:
        self.sections.monitoring.wandb_entity = value

    @property
    def wandb_run_name(self) -> Optional[str]:
        return self.sections.monitoring.wandb_run_name

    @wandb_run_name.setter
    def wandb_run_name(self, value: Optional[str]) -> None:
        self.sections.monitoring.wandb_run_name = value

    @property
    def date_column(self) -> Optional[str]:
        return self.sections.monitoring.date_column

    @date_column.setter
    def date_column(self, value: Optional[str]) -> None:
        self.sections.monitoring.date_column = value

    # -- Reproducibility settings proxies ------------------------------------
    @property
    def seed(self) -> int:
        return self.sections.reproducibility.seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.sections.reproducibility.seed = value

    @property
    def deterministic(self) -> bool:
        return self.sections.reproducibility.deterministic

    @deterministic.setter
    def deterministic(self, value: bool) -> None:
        self.sections.reproducibility.deterministic = value

    # -- Derived values proxies ----------------------------------------------
    @property
    def enc_in(self) -> Optional[int]:
        return self.sections.derived.enc_in

    @enc_in.setter
    def enc_in(self, value: Optional[int]) -> None:
        self.sections.derived.enc_in = value

    @property
    def dec_in(self) -> Optional[int]:
        return self.sections.derived.dec_in

    @dec_in.setter
    def dec_in(self, value: Optional[int]) -> None:
        self.sections.derived.dec_in = value

    @property
    def c_out(self) -> Optional[int]:
        return self.sections.derived.c_out

    @c_out.setter
    def c_out(self, value: Optional[int]) -> None:
        self.sections.derived.c_out = value

    # -- Rehearsal settings proxies ------------------------------------------
    @property
    def rehearsal_enabled(self) -> bool:
        return self.sections.training.rehearsal.enabled

    @rehearsal_enabled.setter
    def rehearsal_enabled(self, value: bool) -> None:
        self.sections.training.rehearsal.enabled = value

    @property
    def rehearsal_buffer_size(self) -> int:
        return self.sections.training.rehearsal.buffer_size

    @rehearsal_buffer_size.setter
    def rehearsal_buffer_size(self, value: int) -> None:
        self.sections.training.rehearsal.buffer_size = value

    @property
    def rehearsal_lr_mult(self) -> float:
        return self.sections.training.rehearsal.rehearsal_lr_mult

    @rehearsal_lr_mult.setter
    def rehearsal_lr_mult(self, value: float) -> None:
        self.sections.training.rehearsal.rehearsal_lr_mult = value


# ---------------------------------------------------------------------------
# Presets de entrenamiento
# ---------------------------------------------------------------------------

# Presets de entrenamiento — definen overrides sobre la config base
TRAINING_PRESETS: dict[str, dict] = {
    "debug": {
        # Modelo pequeño, pocas épocas, sin compilación, sin AMP
        # Útil para verificar que el pipeline funciona rápidamente
        "seq_len": 32,
        "pred_len": 8,
        "batch_size": 16,
        "n_epochs_per_fold": 3,
        "use_amp": False,
        "compile_mode": "none",
        "num_workers": 0,
        "pin_memory": False,
    },
    "cpu_safe": {
        # Sin AMP, sin compilación, num_workers=0, pin_memory=False
        # Para entornos sin GPU o con GPU limitada
        "use_amp": False,
        "compile_mode": "none",
        "num_workers": 0,
        "pin_memory": False,
    },
    "gpu_research": {
        # AMP activo, compilación si disponible, batch mayor
        # Para experimentos con GPU potente
        "use_amp": True,
        "compile_mode": "default",
        "batch_size": 128,
        "num_workers": 4,
        "pin_memory": True,
    },
    "fourier_optimized": {
        # Perfil de optimización Phase 1 para comparar 64 vs 48 modos
        "modes": 48,
    },
    "probabilistic_eval": {
        # Monitor por pinball_p50, paciencia mayor
        # Para evaluación probabilística rigurosa
        "monitor_metric": "val_pinball_p50",
        "monitor_mode": "min",
        "patience": 10,
    },
}

# Mapeo de claves de preset a atributos de FEDformerConfig
_PRESET_KEY_SETTERS: dict[str, str] = {
    "seq_len": "seq_len",
    "pred_len": "pred_len",
    "batch_size": "batch_size",
    "n_epochs_per_fold": "n_epochs_per_fold",
    "use_amp": "use_amp",
    "mc_dropout_eval_samples": "mc_dropout_eval_samples",
    "compile_mode": "compile_mode",
    "num_workers": "num_workers",
    "pin_memory": "pin_memory",
    "modes": "modes",
    "monitor_metric": "monitor_metric",
    "monitor_mode": "monitor_mode",
    "patience": "patience",
}


def apply_preset(config: "FEDformerConfig", preset_name: str) -> "FEDformerConfig":
    """Aplica un preset de entrenamiento sobre la config base.

    Prioridad: defaults < preset < overrides CLI explícitos.
    Los overrides CLI se aplican DESPUÉS de apply_preset, por lo que
    los flags explícitos siempre tienen precedencia sobre el preset.

    Args:
        config: FEDformerConfig ya inicializado con defaults.
        preset_name: nombre del preset
            (debug, cpu_safe, gpu_research, probabilistic_eval).

    Returns:
        Misma instancia config mutada con los valores del preset.

    Raises:
        ValueError: si preset_name no está en TRAINING_PRESETS.
    """
    if preset_name not in TRAINING_PRESETS:
        raise ValueError(
            f"Preset '{preset_name}' no reconocido. "
            f"Valores permitidos: {sorted(TRAINING_PRESETS)}"
        )
    overrides = TRAINING_PRESETS[preset_name]
    for key, value in overrides.items():
        attr = _PRESET_KEY_SETTERS.get(key, key)
        setattr(config, attr, value)
    return config
