import pytest
import numpy as np
from simulations import RiskSimulator, PortfolioSimulator
from typing import Tuple


@pytest.fixture
def sample_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # (n_samples, n_timesteps, n_assets)
    samples = np.random.randn(100, 50, 1)
    predictions = np.mean(samples, axis=0)
    ground_truth = predictions + np.random.randn(50, 1) * 0.1
    return samples, predictions, ground_truth


def test_risk_simulator(sample_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    samples, _, _ = sample_data
    risk_sim = RiskSimulator(samples)
    var = risk_sim.calculate_var()
    cvar = risk_sim.calculate_cvar()
    assert var.shape == (50, 1)
    assert cvar.shape == (50, 1)
    assert np.all(var >= 0)
    assert np.all(cvar >= 0)
    assert np.all(var <= cvar)


def test_portfolio_simulator(
    sample_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _, predictions, ground_truth = sample_data
    portfolio_sim = PortfolioSimulator(predictions, ground_truth)
    strategy_returns = portfolio_sim.run_simple_strategy()
    metrics = portfolio_sim.calculate_metrics(strategy_returns)
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "volatility" in metrics
    assert "sortino_ratio" in metrics


def test_portfolio_uses_all_timesteps() -> None:
    """La señal debe usar el primer y último timestep predicho (no solo los 2 primeros)."""
    rng = np.random.default_rng(42)
    pred_len = 24
    n_samples = 10
    n_assets = 1

    # Predicciones con tendencia fuertemente positiva en toda la ventana
    predictions = np.linspace(1.0, 2.0, pred_len)[None, :, None].repeat(
        n_samples, axis=0
    )
    predictions += rng.normal(0, 0.01, predictions.shape)

    # Ground truth también creciente
    ground_truth = np.linspace(1.0, 2.0, pred_len)[None, :, None].repeat(
        n_samples, axis=0
    )
    ground_truth += rng.normal(0, 0.01, ground_truth.shape)

    sim = PortfolioSimulator(predictions, ground_truth)
    returns = sim.run_simple_strategy()

    # Con tendencia positiva clara, la mayoría de señales deben ser positivas
    assert returns.shape == (n_samples - 1, n_assets)
    assert returns.mean() > 0, (
        "Señal de momentum debe ser positiva con tendencia alcista"
    )


def test_portfolio_single_timestep_fallback() -> None:
    """Fallback para pred_len=1 no debe fallar."""
    rng = np.random.default_rng(0)
    predictions = rng.normal(0, 1, (20, 1, 1))
    ground_truth = rng.normal(0, 1, (20, 1, 1))
    sim = PortfolioSimulator(predictions, ground_truth)
    returns = sim.run_simple_strategy()
    assert returns.shape[0] == 19


def test_risk_simulator_accepts_forecast_output() -> None:
    """RiskSimulator acepta ForecastOutput y usa samples_real vía samples_for_metrics.

    samples_for_metrics siempre apunta a samples_real (espacio desescalado interpretable).
    El VaR calculado sobre samples_real debe diferir del calculado sobre samples_scaled
    porque son distintos arrays (samples_real = samples_scaled * 50 + 100).
    """
    from training.forecast_output import ForecastOutput

    n_samples, n_windows, pred_len, n_targets = 100, 20, 5, 1
    rng = np.random.default_rng(7)
    samples_scaled = rng.normal(0, 1, (n_samples, n_windows, pred_len, n_targets))
    samples_real = samples_scaled * 50 + 100  # precios simulados positivos

    # ForecastOutput con metric_space="returns": samples_for_metrics → samples_real
    fo = ForecastOutput(
        preds_scaled=np.zeros((n_windows, pred_len, n_targets)),
        gt_scaled=np.zeros((n_windows, pred_len, n_targets)),
        samples_scaled=samples_scaled,
        preds_real=np.zeros((n_windows, pred_len, n_targets)),
        gt_real=np.zeros((n_windows, pred_len, n_targets)),
        samples_real=samples_real,
        metric_space="returns",
        return_transform="none",
        target_names=["Close"],
    )

    # Verificar que samples_for_metrics apunta a samples_real (no a samples_scaled)
    assert np.array_equal(fo.samples_for_metrics, samples_real)
    assert not np.array_equal(fo.samples_for_metrics, samples_scaled)

    # RiskSimulator usa samples_for_metrics internamente
    risk_real = RiskSimulator(fo)
    risk_scaled = RiskSimulator(samples_scaled)  # directo, backward compat

    var_real = risk_real.calculate_var()
    var_scaled = risk_scaled.calculate_var()

    assert var_real.shape == (pred_len, n_targets)
    # VaR en escala real (precios ~100) debe diferir del VaR en escala z-score (~0)
    assert not np.allclose(var_real, var_scaled), (
        "VaR sobre precios reales debe diferir del VaR sobre valores estandarizados"
    )


def test_risk_simulator_price_space_converts_to_cumulative_returns() -> None:
    """Con return_transform='none', RiskSimulator convierte precios a retornos acumulados.

    VaR sobre niveles de precio absolutos no es VaR financiero estándar.
    La conversión a retorno acumulado (P_t - P_0) / P_0 asegura que la escala
    sea comparable entre activos y que el VaR tenga interpretación económica.
    """
    from training.forecast_output import ForecastOutput

    n_samples, n_windows, pred_len, n_targets = 200, 10, 5, 1
    rng = np.random.default_rng(42)

    # Precios NVDA simulados: ~400 USD con volatilidad diaria ~1%
    base_price = 400.0
    daily_vol = 0.01
    price_paths = base_price * np.exp(
        np.cumsum(
            rng.normal(0, daily_vol, (n_samples, n_windows, pred_len, n_targets)),
            axis=2,
        )
    )

    fo = ForecastOutput(
        preds_scaled=np.zeros((n_windows, pred_len, n_targets)),
        gt_scaled=np.zeros((n_windows, pred_len, n_targets)),
        samples_scaled=np.zeros((n_samples, n_windows, pred_len, n_targets)),
        preds_real=np.zeros((n_windows, pred_len, n_targets)),
        gt_real=np.zeros((n_windows, pred_len, n_targets)),
        samples_real=price_paths,
        metric_space="returns",
        return_transform="none",  # → in_price_space=True
        target_names=["Close"],
    )

    risk = RiskSimulator(fo)
    var = risk.calculate_var()

    assert var.shape == (pred_len, n_targets)
    # t=0: retorno acumulado desde inicio = 0 siempre → VaR en t=0 ≈ 0
    assert abs(var[0, 0]) < 1e-9, "VaR en t=0 debe ser cero (retorno acumulado inicial)"
    # t>0: VaR en escala de retornos (fracción), no en USD (~400)
    assert abs(var[-1, 0]) < 1.0, (
        f"VaR en t={pred_len - 1} debe estar en escala de retornos (<1), "
        f"no en precio (~{base_price}). Got {var[-1, 0]:.2f}"
    )


def test_portfolio_simulator_return_space_uses_gt_as_returns() -> None:
    """Con return_transform='log_return' + metric_space='returns', run_simple_strategy
    usa gt directamente como retornos, sin calcular diff(gt)/gt (segunda derivada).
    """
    from training.forecast_output import ForecastOutput

    n_windows, pred_len, n_targets = 30, 6, 1
    rng = np.random.default_rng(99)

    # Log-returns con sesgo positivo claro (bull market simulado)
    log_returns = rng.normal(0.005, 0.01, (n_windows, pred_len, n_targets))
    preds_returns = log_returns + rng.normal(0, 0.001, log_returns.shape)

    fo = ForecastOutput(
        preds_scaled=np.zeros_like(preds_returns),
        gt_scaled=np.zeros_like(log_returns),
        samples_scaled=np.zeros((10, n_windows, pred_len, n_targets)),
        preds_real=preds_returns,
        gt_real=log_returns,
        samples_real=np.zeros((10, n_windows, pred_len, n_targets)),
        metric_space="returns",
        return_transform="log_return",  # → _in_return_space=True
        target_names=["Close"],
    )

    ps = PortfolioSimulator(fo)
    assert ps._in_return_space is True, "Debe detectar espacio de retornos"

    strategy_returns = ps.run_simple_strategy()
    assert strategy_returns.shape == (n_windows - 1, n_targets)
    # Con log-returns positivos, la estrategia debe capturar retornos positivos
    assert strategy_returns.mean() > 0, (
        "Con bull market simulado, la estrategia momentum debe ser positiva"
    )


def test_calculate_metrics_sharpe_corrects_overlapping_window_bias() -> None:
    """calculate_metrics debe usar ventanas no solapadas y sqrt(252/pred_len).

    Con pred_len > 1, strategy_returns consecutivos comparten pred_len-1 pasos.
    Esto reduce el std ~sqrt(pred_len) x → infla Sharpe ~sqrt(pred_len) x.
    La corrección: subsamplear a cada pred_len-ésima muestra + anualizar con
    sqrt(252/pred_len) en lugar de sqrt(252).
    """
    rng = np.random.default_rng(7)
    pred_len = 20
    n_windows = 300

    # Señales constantes para aislar el efecto del pred_len en Sharpe
    strategy_returns = rng.normal(0.001, 0.01, n_windows - 1)

    # PortfolioSimulator con pred_len=20 (precios, sin ForecastOutput)
    preds_20 = np.ones((n_windows, pred_len, 1))  # pred_len=20
    ps_20 = PortfolioSimulator(preds_20, preds_20)

    # PortfolioSimulator con pred_len=1 para comparar annualization naïve
    preds_1 = np.ones((n_windows, 1, 1))
    ps_1 = PortfolioSimulator(preds_1, preds_1)

    metrics_20 = ps_20.calculate_metrics(strategy_returns)
    metrics_1 = ps_1.calculate_metrics(strategy_returns)

    # Con pred_len=20, el Sharpe debe ser menor que con pred_len=1
    # (pred_len=1 usa sqrt(252), pred_len=20 usa sqrt(252/20) ≈ 3.55)
    assert metrics_20["sharpe_ratio"] < metrics_1["sharpe_ratio"], (
        f"Sharpe con pred_len=20 ({metrics_20['sharpe_ratio']:.3f}) debe ser "
        f"menor que con pred_len=1 ({metrics_1['sharpe_ratio']:.3f}). "
        "Verifica la corrección de sesgo por ventanas solapadas."
    )


def test_portfolio_simulator_accepts_forecast_output() -> None:
    """PortfolioSimulator acepta ForecastOutput y usa preds/gt_for_metrics."""
    from training.forecast_output import ForecastOutput

    n_windows, pred_len, n_targets = 20, 10, 1
    rng = np.random.default_rng(13)
    preds_scaled = rng.normal(0, 0.01, (n_windows, pred_len, n_targets))
    gt_scaled = preds_scaled + rng.normal(0, 0.001, (n_windows, pred_len, n_targets))
    samples_scaled = rng.normal(0, 0.01, (50, n_windows, pred_len, n_targets))

    fo = ForecastOutput(
        preds_scaled=preds_scaled,
        gt_scaled=gt_scaled,
        samples_scaled=samples_scaled,
        preds_real=preds_scaled * 100 + 200,
        gt_real=gt_scaled * 100 + 200,
        samples_real=samples_scaled * 100 + 200,
        metric_space="returns",
        return_transform="none",
        target_names=["Close"],
    )

    ps = PortfolioSimulator(fo)
    strategy_returns = ps.run_simple_strategy()
    metrics = ps.calculate_metrics(strategy_returns)

    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "volatility" in metrics
    assert "sortino_ratio" in metrics
