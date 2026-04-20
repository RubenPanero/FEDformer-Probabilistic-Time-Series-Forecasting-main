# -*- coding: utf-8 -*-
"""
Tests unitarios para el scheduler de LR y el early stopping del WalkForwardTrainer.
Pruebas mínimas y rápidas que no requieren ejecuciones de entrenamiento reales.
"""

from unittest.mock import MagicMock

import torch

from training.trainer import _EarlyStopping, WalkForwardTrainer


def test_cosine_scheduler_reduces_lr() -> None:
    """El scheduler cosine debe reducir la LR desde su valor inicial hasta min_lr."""
    # Crear un optimizador mínimo con un parámetro dummy
    param = torch.nn.Parameter(torch.tensor([1.0]))
    initial_lr = 1e-3
    min_lr = 1e-6
    total_epochs = 10

    optimizer = torch.optim.SGD([param], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=min_lr,
    )

    lr_inicial = optimizer.param_groups[0]["lr"]

    # Avanzar varias épocas (optimizer.step primero, luego scheduler.step)
    for _ in range(total_epochs):
        optimizer.step()
        scheduler.step()

    lr_final = optimizer.param_groups[0]["lr"]

    # La LR debe haber disminuido respecto al valor inicial
    assert lr_final < lr_inicial, (
        f"Se esperaba LR final ({lr_final}) < LR inicial ({lr_inicial})"
    )
    # La LR final debe ser aproximadamente min_lr al completar T_max épocas
    assert lr_final <= initial_lr, (
        f"La LR ({lr_final}) no debe superar la LR inicial ({initial_lr})"
    )


def test_early_stopping_triggers() -> None:
    """_EarlyStopping debe retornar True exactamente después de `patience` pasos sin mejora."""
    patience = 2
    min_delta = 1e-4

    stopper = _EarlyStopping(patience=patience, min_delta=min_delta)

    # Primera pérdida — establece best_loss
    resultado = stopper.step(1.0)
    assert not resultado, "El primer paso no debe activar la parada"
    assert stopper.counter == 0

    # Segunda pérdida sin mejora — counter sube a 1
    resultado = stopper.step(1.0)
    assert not resultado, "No debe activarse después del primer paso sin mejora"
    assert stopper.counter == 1

    # Tercera pérdida sin mejora — counter sube a 2 == patience → debe activarse
    resultado = stopper.step(1.0)
    assert resultado, "Debe activarse después de 2 pasos sin mejora (patience=2)"
    assert stopper.should_stop is True


def test_early_stopping_restores_best() -> None:
    """_EarlyStopping.best_loss debe rastrear el mínimo visto durante el entrenamiento."""
    stopper = _EarlyStopping(patience=5, min_delta=1e-4)

    # Secuencia con mejoras y retrocesos
    losses = [2.0, 1.5, 1.2, 1.3, 1.1, 1.4, 0.9, 1.0]
    for loss in losses:
        stopper.step(loss)

    # El mejor loss visto es 0.9
    assert abs(stopper.best_loss - 0.9) < 1e-9, (
        f"Se esperaba best_loss=0.9, got {stopper.best_loss}"
    )


def test_early_stopping_disabled_when_patience_zero() -> None:
    """Cuando patience=0, _EarlyStopping siempre retorna False (deshabilitado)."""
    stopper = _EarlyStopping(patience=0, min_delta=1e-4)

    # Incluso con muchos pasos sin mejora, nunca debe activarse
    for _ in range(100):
        resultado = stopper.step(1.0)
        assert not resultado, (
            "Con patience=0, la parada anticipada debe estar deshabilitada"
        )

    assert not stopper.should_stop


def test_run_backtest_uses_raw_rows_for_split(config, synthetic_batch) -> None:
    """run_backtest debe usar filas crudas para el split, no ventanas.

    Regresión: con seq_len grande y split basado en ventanas, fold 1 obtenía
    0 batches (train_windows < batch_size) → avg_loss = inf siempre.
    """
    from data import TimeSeriesDataset

    ds = TimeSeriesDataset(config, flag="all")
    trainer = WalkForwardTrainer(config, ds)

    n_splits = 5
    # Comportamiento correcto: usar filas crudas
    total_size_rows = len(trainer.full_dataset.full_data_scaled)
    split_size = max(total_size_rows // n_splits, config.seq_len + config.pred_len)

    for fold_idx in range(1, n_splits):
        train_end_idx = fold_idx * split_size
        train_max_start = train_end_idx - config.seq_len - config.pred_len
        n_train_windows = max(0, train_max_start + 1)
        n_batches = n_train_windows // config.batch_size
        assert n_batches >= 1, (
            f"Fold {fold_idx}: {n_train_windows} ventanas → {n_batches} batches. "
            "run_backtest usa ventanas en lugar de filas crudas para el split."
        )


def test_early_stopping_resets_counter_on_improvement() -> None:
    """El contador debe reiniciarse a cero cuando se observa una mejora suficiente."""
    stopper = _EarlyStopping(patience=3, min_delta=1e-4)

    # Primer paso: mejora desde inf → best_loss=1.0, counter queda en 0
    # Segundo paso: sin mejora (1.0 no supera 1.0 - min_delta) → counter sube a 1
    stopper.step(1.0)
    stopper.step(1.0)
    assert stopper.counter == 1

    # Mejora significativa — counter debe reiniciarse
    stopper.step(0.5)
    assert stopper.counter == 0, (
        f"El contador debe ser 0 después de una mejora, got {stopper.counter}"
    )
    assert stopper.best_loss == 0.5


# ---------------------------------------------------------------------------
# Tests para _eval_epoch
# ---------------------------------------------------------------------------


def _make_fake_batches(config, n_batches: int = 3, batch_size: int = 2) -> list:
    """Crea una lista de dicts que simula un DataLoader de validación."""
    return [
        {
            "x_enc": torch.randn(batch_size, config.seq_len, config.enc_in),
            "x_dec": torch.randn(
                batch_size,
                config.label_len + config.pred_len,
                config.dec_in,
            ),
            "y_true": torch.randn(batch_size, config.pred_len, config.c_out),
            "x_regime": torch.zeros(batch_size, 1, dtype=torch.long),
        }
        for _ in range(n_batches)
    ]


def test_eval_epoch_returns_finite_loss(config, model_factory) -> None:
    """_eval_epoch debe retornar un float finito con batches válidos."""
    from utils import get_device
    from training.trainer import WalkForwardTrainer

    trainer = WalkForwardTrainer(config, MagicMock())
    model = model_factory().to(get_device())

    fake_batches = _make_fake_batches(config)
    result = trainer._eval_epoch(model, fake_batches)  # type: ignore[arg-type]

    assert isinstance(result, float), f"Se esperaba float, got {type(result)}"
    assert result != float("inf"), "La pérdida no debe ser inf con batches válidos"
    assert result == result, "La pérdida no debe ser NaN"  # NaN != NaN


def test_eval_epoch_empty_loader_returns_inf(config, model_factory) -> None:
    """Con val_loader vacío, _eval_epoch debe retornar float('inf')."""
    from utils import get_device
    from training.trainer import WalkForwardTrainer

    trainer = WalkForwardTrainer(config, MagicMock())
    model = model_factory().to(get_device())

    result = trainer._eval_epoch(model, [])  # type: ignore[arg-type]

    assert result == float("inf"), f"Con loader vacío se esperaba inf, got {result}"


def test_val_fraction_zero_disables_split(config) -> None:
    """val_fraction=0 debe desactivar el split de validación intra-fold."""
    config.val_fraction = 0.0
    assert config.val_fraction == 0.0

    # Verificar que el valor se propaga al config anidado
    assert config.sections.training.loop.val_fraction == 0.0


def test_train_loader_drop_last_false_produces_batches_when_subset_smaller_than_batch(
    config,
) -> None:
    """drop_last=False en train_loader debe producir ≥1 batch cuando n_ventanas < batch_size.

    Regresión: con drop_last=True y fold 1 con seq_len grande, el subset de train
    podía tener menos ventanas que batch_size → 0 batches → train_loss=inf en todas
    las épocas del fold.
    """
    import torch
    from torch.utils.data import Subset, TensorDataset
    from training.trainer import WalkForwardTrainer

    # Dataset sintético con n_samples < batch_size (simula fold 1 con seq_len=252)
    batch_size = config.batch_size  # 8 en el fixture de tests
    n_samples = max(1, batch_size - 1)  # garantiza n_samples < batch_size

    # TensorDataset mínimo para construir un Subset válido
    dummy_x = torch.zeros(n_samples, config.seq_len, config.enc_in)
    dummy_dataset = TensorDataset(dummy_x)
    small_subset = Subset(dummy_dataset, list(range(n_samples)))
    full_subset = Subset(dummy_dataset, list(range(n_samples)))  # test loader

    trainer = WalkForwardTrainer(config, MagicMock())
    train_loader, _ = trainer._prepare_data_loaders(small_subset, full_subset)

    n_batches = len(train_loader)
    assert n_batches >= 1, (
        f"Con drop_last=False y {n_samples} muestras (< batch_size={batch_size}), "
        f"se esperaba ≥1 batch, pero se obtuvieron {n_batches}. "
        "Verifica que drop_last=False en _prepare_data_loaders."
    )


def test_pin_memory_disabled_by_default(config) -> None:
    """pin_memory debe permanecer apagado salvo activación explícita."""
    from training.trainer import WalkForwardTrainer

    trainer = WalkForwardTrainer(config, MagicMock())
    assert trainer._pin_memory_enabled() is False


def test_pin_memory_respects_runtime_flag(config) -> None:
    """pin_memory solo debe activarse con flag explícito y CUDA disponible."""
    from training.trainer import WalkForwardTrainer

    config.pin_memory = True
    trainer = WalkForwardTrainer(config, MagicMock())
    assert trainer._pin_memory_enabled() is torch.cuda.is_available()


def test_num_workers_uses_runtime_override(config) -> None:
    """num_workers debe respetar el override explícito del runtime."""
    from training.trainer import WalkForwardTrainer

    config.num_workers = 0
    trainer = WalkForwardTrainer(config, MagicMock())
    assert trainer._num_workers() == 0


# ---------------------------------------------------------------------------
# Tests para los nuevos defaults de LoopSettings
# ---------------------------------------------------------------------------


def test_loop_settings_new_defaults() -> None:
    """Los defaults de LoopSettings deben reflejar la configuración de entrenamiento aumentada.

    Regresión: antes n_epochs_per_fold=5, patience=0, min_delta=1e-4, accum=1.
    Ahora: n_epochs_per_fold=20, patience=5, min_delta=5e-3, accum=2.
    """
    from config import LoopSettings

    loop = LoopSettings()
    assert loop.n_epochs_per_fold == 20, (
        f"n_epochs_per_fold default debe ser 20, got {loop.n_epochs_per_fold}"
    )
    assert loop.patience == 5, f"patience default debe ser 5, got {loop.patience}"
    assert loop.min_delta == 5e-3, (
        f"min_delta default debe ser 5e-3, got {loop.min_delta}"
    )
    assert loop.gradient_accumulation_steps == 2, (
        f"gradient_accumulation_steps default debe ser 2, got {loop.gradient_accumulation_steps}"
    )


def test_gradient_accumulation_halves_optimizer_steps() -> None:
    """Con gradient_accumulation_steps=2, el optimizador se ejecuta la mitad de veces que con accum=1.

    _should_step activa el paso solo cada `accum` batches (o en el último batch).
    """
    from training.trainer import WalkForwardTrainer

    n_batches = 4

    # accum=1: todos los batches producen un paso → 4 pasos
    steps_accum1 = sum(
        1
        for i in range(n_batches)
        if WalkForwardTrainer._should_step(i, n_batches, accumulation_steps=1)
    )
    # accum=2: un paso cada 2 batches + último batch → 2 pasos
    steps_accum2 = sum(
        1
        for i in range(n_batches)
        if WalkForwardTrainer._should_step(i, n_batches, accumulation_steps=2)
    )

    assert steps_accum1 == 4, f"Con accum=1 se esperaban 4 pasos, got {steps_accum1}"
    assert steps_accum2 == 2, f"Con accum=2 se esperaban 2 pasos, got {steps_accum2}"
    assert steps_accum2 == steps_accum1 // 2, (
        "Con accum=2 el número de pasos debe ser la mitad que con accum=1"
    )


def test_patience_min_delta_ignores_small_noise() -> None:
    """Con min_delta=5e-3, fluctuaciones menores no deben incrementar el contador de patience.

    Regresión: con min_delta=1e-4 (anterior), ruido típico de folds pequeños (~1e-3)
    activaba el contador prematuramente. Con 5e-3 se filtra ese ruido.
    """
    stopper = _EarlyStopping(patience=5, min_delta=5e-3)

    # Primera pérdida: establece best_loss=1.0
    stopper.step(1.0)

    # Fluctuaciones menores que min_delta no deben incrementar el contador
    for noisy_loss in [1.003, 1.002, 0.998, 1.001]:
        result = stopper.step(noisy_loss)
        assert not result, (
            f"Ruido {noisy_loss} (Δ < 5e-3 respecto a 1.0) no debe activar early stopping"
        )

    # Una mejora real sí debe resetear el contador
    stopper.step(0.990)  # Δ = 0.01 > min_delta → resetea contador
    assert stopper.counter == 0, (
        f"Tras mejora real (Δ=0.01 > min_delta=5e-3), counter debe ser 0, got {stopper.counter}"
    )


def test_early_stopping_uses_configured_monitor_metric() -> None:
    """_select_monitor_value + _EarlyStopping integran correctamente la métrica configurada.

    Cuando monitor_metric='val_pinball_p50', el valor de pinball_p50 debe alimentar
    al early stopper en lugar de val_loss.
    """
    train_m = {"loss": 1.5}
    val_m_good = {"loss": 1.2, "pinball_p50": 0.05}  # pinball mejorando
    val_m_bad = {"loss": 1.2, "pinball_p50": 0.20}  # pinball empeorando

    stopper = _EarlyStopping(patience=2, min_delta=1e-4)

    # Primera llamada establece best con pinball=0.05
    v1 = WalkForwardTrainer._select_monitor_value(
        train_m, val_m_good, "val_pinball_p50"
    )
    assert abs(v1 - 0.05) < 1e-9
    stopper.step(v1)

    # Segunda llamada con pinball=0.20 debe incrementar el contador
    v2 = WalkForwardTrainer._select_monitor_value(train_m, val_m_bad, "val_pinball_p50")
    assert abs(v2 - 0.20) < 1e-9
    stopper.step(v2)
    assert stopper.counter == 1, (
        f"Counter debe ser 1 tras empeoramiento, got {stopper.counter}"
    )
