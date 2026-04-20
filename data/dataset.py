# -*- coding: utf-8 -*-
"""
Manejo de dataset y detección de regímenes para entrenamiento de series temporales.
Refactorizado con tipado estricto Python 3.10+ y mejores prácticas.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import FEDformerConfig
from .preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detector de regímenes basado en cuantiles de volatilidad móvil (rolling)."""

    def __init__(self, n_regimes: int) -> None:
        self.n_regimes = n_regimes
        self.quantiles: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        """Ajustar cuantiles según series numéricas empíricas."""
        try:
            returns = np.diff(data, axis=0) / (np.abs(data[:-1]) + 1e-9)
            rolling_vol = (
                pd.DataFrame(returns)
                .rolling(window=min(24, max(2, len(returns) // 2)), min_periods=1)
                .std(ddof=1)
            )
            volatility = rolling_vol.dropna().values.std(axis=0)
            if len(volatility) > 1:
                self.quantiles = np.quantile(
                    volatility, np.linspace(0, 1, self.n_regimes + 1)[1:-1]
                )
            else:
                self.quantiles = np.zeros(self.n_regimes - 1)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Fallo en ajuste de detector de regímenes: %s. Usando defaults (0).",
                exc,
            )
            self.quantiles = np.zeros(self.n_regimes - 1)

    def get_regime(self, sequence: np.ndarray) -> int:
        """Categoriza una secuencia temporal dada en función del umbral computado."""
        if self.quantiles is None:
            raise RuntimeError("El detector no ha sido ajustado (fitted).")

        try:
            returns = np.diff(sequence, axis=0) / (np.abs(sequence[:-1]) + 1e-9)
            sequence_vol = np.std(returns, axis=1).mean()
            return min(np.digitize(sequence_vol, self.quantiles), self.n_regimes - 1)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Fallo en detección de régimen: %s. Empleando régimen 0.", exc
            )
            return 0


class TimeSeriesDataset(Dataset):
    """Dataset de series de tiempo que delega preprocesamiento a PreprocessingPipeline."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        config: FEDformerConfig,
        flag: str,
        fit_end_idx: int | None = None,
        strict: bool | None = None,
        preprocessor: PreprocessingPipeline | None = None,
    ) -> None:
        self.config = config
        self.flag = flag
        self.strict = config.strict_mode if strict is None else strict
        self.preprocessor = preprocessor or PreprocessingPipeline.from_config(
            config,
            target_features=config.target_features,
            date_column=config.date_column,
            strict_mode=self.strict,
        )

        self.raw_df = self._read_raw_data()
        self.regime_detector = RegimeDetector(n_regimes=self.config.n_regimes)

        self._fit_and_transform(
            fit_end_idx=fit_end_idx, force_refit=not self.preprocessor.fitted
        )
        self._set_split_view()

    def _read_raw_data(self) -> pd.DataFrame:
        """Carga perezosa de la fuente cruda en base al config provisto."""
        parse_cols = [self.config.date_column] if self.config.date_column else None
        return pd.read_csv(self.config.file_path, parse_dates=parse_cols)

    def _fit_and_transform(self, fit_end_idx: int | None, force_refit: bool) -> None:
        """Ejecuta los protocolos de preprocesado numérico si corresponde."""
        fit_scope = self.preprocessor.fit_scope
        should_refit = (
            force_refit
            or fit_scope == "fold_train_only"
            or not self.preprocessor.fitted
        )
        if should_refit:
            default_cutoff = max(1, int(len(self.raw_df) * 0.7))
            cutoff = fit_end_idx if fit_end_idx is not None else default_cutoff
            self.preprocessor.fit(self.raw_df, fit_end_idx=cutoff)

        transformed_df = self.preprocessor.transform(self.raw_df)
        self.feature_columns = list(transformed_df.columns)
        self.target_indices = list(self.preprocessor.target_indices)
        self.full_data_scaled = transformed_df.values.astype(np.float32)
        self.scaler = self.preprocessor.scaler

        self.config.enc_in = len(self.feature_columns)
        self.config.dec_in = len(self.feature_columns)
        self.config.c_out = len(self.config.target_features)

        cutoff = self.preprocessor.fit_end_idx or max(
            1, int(len(self.full_data_scaled) * 0.7)
        )
        self.regime_detector.fit(self.full_data_scaled[:cutoff, self.target_indices])
        self._get_regime_cached.cache_clear()

        if self.config.persist_artifacts:
            self.preprocessor.save_artifacts(self.config.artifact_dir)

    def _set_split_view(self) -> None:
        """Vista estática 70/20/10 (train/val/test) para uso standalone o en tests.

        El pipeline walk-forward usa flag="all" + Subset con índices calculados
        por WalkForwardTrainer; este método no interviene en ese flujo.
        """
        n_rows = len(self.full_data_scaled)
        num_train = int(n_rows * 0.7)
        num_val = int(n_rows * 0.2)
        # num_test = n_rows - num_train - num_val  (~10%)

        border1s = {
            "train": 0,
            "val": max(0, num_train - self.config.seq_len),
            "test": max(0, num_train + num_val - self.config.seq_len),
            "all": 0,
        }
        border2s = {
            "train": num_train,
            "val": num_train + num_val,
            "test": n_rows,
            "all": n_rows,
        }
        if self.flag not in border1s:
            raise ValueError(
                f"La bandera 'flag' debe ser {list(border1s.keys())}. Recibido: {self.flag}"
            )

        self.data_x = self.full_data_scaled[border1s[self.flag] : border2s[self.flag]]
        valid_len = len(self.data_x) - self.config.seq_len - self.config.pred_len + 1
        self._valid_indices = list(range(max(0, valid_len)))

    def refit_for_cutoff(self, fit_end_idx: int | None = None) -> None:
        """Realiza actualización de las métricas en modo secuencial."""
        if self.preprocessor.fit_scope == "global_train" and self.preprocessor.fitted:
            self._set_split_view()
            return
        self._fit_and_transform(fit_end_idx=fit_end_idx, force_refit=True)
        self._set_split_view()

    @lru_cache(maxsize=2048)
    def _get_regime_cached(self, seq_hash: tuple[float, ...]) -> int:
        """Cálculo memoizado (LRU) de regímenes, dependiente de secuencia serializada."""
        # Se recrea matriz, idealmente seguro al tratarse de tuplas chatas per feature array
        seq_array = np.array(seq_hash).reshape(-1, len(self.target_indices))
        return self.regime_detector.get_regime(seq_array)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Recupera la secuencia iterada temporal y su objetivo predictivo."""
        try:
            if index < 0 or index >= len(self):
                raise IndexError(
                    f"Índice {index} fuera de los bordes del dataset (longitud {len(self)})"
                )

            s_end = index + self.config.seq_len
            r_end = s_end + self.config.pred_len

            seq_x = self.data_x[index:s_end]
            seq_dec_input = self.data_x[s_end - self.config.label_len : r_end]
            seq_y_true = seq_dec_input[-self.config.pred_len :, self.target_indices]

            # Convert to distinct flat tuple for reliable hashing caching
            seq_hash = tuple(float(x) for x in seq_x[:, self.target_indices].flatten())
            regime = self._get_regime_cached(seq_hash)

            return {
                "x_enc": torch.from_numpy(seq_x.astype(np.float32)),
                "x_dec": torch.from_numpy(seq_dec_input.astype(np.float32)),
                "y_true": torch.from_numpy(seq_y_true.astype(np.float32)),
                "x_regime": torch.tensor([regime], dtype=torch.long),
            }
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f"Error despachando item {index}: {exc}") from exc

            logger.warning("Error despachando item %s: %s", index, exc)
            return {
                "x_enc": torch.zeros(
                    (self.config.seq_len, self.config.enc_in), dtype=torch.float32
                ),
                "x_dec": torch.zeros(
                    (self.config.label_len + self.config.pred_len, self.config.dec_in),
                    dtype=torch.float32,
                ),
                "y_true": torch.zeros(
                    (self.config.pred_len, self.config.c_out), dtype=torch.float32
                ),
                "x_regime": torch.tensor([0], dtype=torch.long),
            }

    def __len__(self) -> int:
        return len(self._valid_indices)
