# -*- coding: utf-8 -*-
"""
Tests unitarios para tune_hyperparams.py:
  - Poda de trials con seq_len < pred_len*3
  - Penalizaciones por VaR > 0.08 y Sortino < 0
  - Que --save-canonical no se incluye en el cmd de trials
  - Parsing de CSV de portafolio y riesgo
  - download_extra_tickers omite los que ya existen
  - _parse_probabilistic_csv: lectura y fallback
  - _compose_trial_score: modos sharpe, composite, fallback y sin NaN
"""

import math
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pandas as pd
import pytest

import tune_hyperparams as th


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_portfolio_csv(
    tmp_path: Path,
    ticker_stem: str,
    sharpe: float,
    sortino: float,
    include_ticker_suffix: bool = False,
) -> Path:
    """Escribe un CSV de portafolio mínimo para tests."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{ticker_stem}" if include_ticker_suffix else ""
    p = results_dir / f"portfolio_metrics_{ts}{suffix}.csv"
    pd.DataFrame(
        {
            "metric": ["sharpe_ratio", "sortino_ratio", "max_drawdown", "volatility"],
            "value": [sharpe, sortino, -0.4, 0.02],
        }
    ).to_csv(p, index=False)
    return p


def _write_risk_csv(
    tmp_path: Path,
    ticker_stem: str,
    var_95: float,
    include_ticker_suffix: bool = False,
) -> Path:
    """Escribe un CSV de riesgo mínimo para tests."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{ticker_stem}" if include_ticker_suffix else ""
    p = results_dir / f"risk_metrics_{ts}{suffix}.csv"
    pd.DataFrame(
        {
            "step": [0, 1],
            "target_idx": [0, 0],
            "var_95": [var_95, var_95],
            "cvar_95": [var_95 * 1.3, var_95 * 1.3],
        }
    ).to_csv(p, index=False)
    return p


def _write_probabilistic_csv(
    tmp_path: Path,
    ticker_stem: str,
    pinball_p50: float = 0.02,
    coverage_80: float = 0.78,
    interval_width_80: float = 0.05,
    crps: float = 0.03,
) -> Path:
    """Escribe un CSV probabilístico en formato largo (igual que io_experiment.py)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Formato: fold, metric, value, space, aggregation — dos folds sintéticos
    rows = []
    for fold in range(2):
        for metric, val in [
            ("pinball_p50", pinball_p50),
            ("coverage_80", coverage_80),
            ("interval_width_80", interval_width_80),
            ("crps", crps),
        ]:
            rows.append(
                {
                    "fold": fold,
                    "metric": metric,
                    "value": val,
                    "space": "real",
                    "aggregation": "mean",
                }
            )
    p = results_dir / f"probabilistic_metrics_{ts}_{ticker_stem}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Tests de _parse_portfolio_csv y _parse_risk_csv
# ---------------------------------------------------------------------------


def test_parse_portfolio_csv_returns_correct_sharpe(tmp_path: Path) -> None:
    """_parse_portfolio_csv extrae el Sharpe correcto del formato real de main.py."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.75, sortino=1.2)

    result = th._parse_portfolio_csv(tmp_path / "results", "MSFT_features", ts_before)
    assert abs(result["sharpe"] - 0.75) < 1e-6
    assert abs(result["sortino"] - 1.2) < 1e-6


def test_parse_portfolio_csv_accepts_legacy_ticker_suffix(tmp_path: Path) -> None:
    """_parse_portfolio_csv sigue aceptando el formato legado con ticker."""
    ts_before = time.time() - 1
    _write_portfolio_csv(
        tmp_path,
        "MSFT_features",
        sharpe=0.55,
        sortino=0.8,
        include_ticker_suffix=True,
    )

    result = th._parse_portfolio_csv(tmp_path / "results", "MSFT_features", ts_before)
    assert abs(result["sharpe"] - 0.55) < 1e-6
    assert abs(result["sortino"] - 0.8) < 1e-6


def test_parse_portfolio_csv_returns_sentinel_when_missing(tmp_path: Path) -> None:
    """_parse_portfolio_csv retorna Sharpe=-1.0 si no hay CSV para el ticker."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result = th._parse_portfolio_csv(results_dir, "UNKNOWN_features", time.time())
    assert result["sharpe"] == -1.0


def test_parse_risk_csv_returns_mean_var(tmp_path: Path) -> None:
    """_parse_risk_csv retorna el VaR medio del formato real de main.py."""
    ts_before = time.time() - 1
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.06)
    result = th._parse_risk_csv(tmp_path / "results", "MSFT_features", ts_before)
    assert abs(result - 0.06) < 1e-6


def test_parse_risk_csv_accepts_legacy_ticker_suffix(tmp_path: Path) -> None:
    """_parse_risk_csv sigue aceptando el formato legado con ticker."""
    ts_before = time.time() - 1
    _write_risk_csv(
        tmp_path,
        "MSFT_features",
        var_95=0.04,
        include_ticker_suffix=True,
    )
    result = th._parse_risk_csv(tmp_path / "results", "MSFT_features", ts_before)
    assert abs(result - 0.04) < 1e-6


# ---------------------------------------------------------------------------
# Tests de la función objective
# ---------------------------------------------------------------------------


def _mock_trial(
    seq_len: int, pred_len: int, batch_size: int = 64, clip: float = 0.5
) -> MagicMock:
    """Construye un trial mock con suggest_categorical predefinido.

    El orden de llamadas en objective es:
    seq_len, pred_len, batch_size, gradient_clip_norm, e_layers, d_layers,
    n_flow_layers, flow_hidden_dim, label_len, dropout.
    """
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    trial.suggest_categorical.side_effect = [
        seq_len,
        pred_len,
        batch_size,
        clip,
        2,
        1,
        4,
        64,
        48,
        0.05,
    ]
    trial.set_user_attr = MagicMock()
    return trial


def test_objective_prunes_when_seq_len_too_short() -> None:
    """objective lanza TrialPruned si seq_len < pred_len * 3."""
    # seq_len=48, pred_len=20 → 48 < 20*3=60 → debe podarse
    trial = _mock_trial(seq_len=48, pred_len=20)

    with pytest.raises(optuna.TrialPruned):
        th.objective(trial, "data/MSFT_features.csv", 4, Path("results"))


def test_objective_prunes_when_label_len_exceeds_seq_len() -> None:
    """objective lanza TrialPruned si label_len supera seq_len."""
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    trial.suggest_categorical.side_effect = [
        96,
        20,
        64,
        0.5,
        2,
        1,
        4,
        64,
        128,
        0.1,
    ]
    trial.set_user_attr = MagicMock()

    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_proc),
        pytest.raises(optuna.TrialPruned),
    ):
        th.objective(trial, "data/MSFT_features.csv", 4, Path("results"))


def test_objective_penalizes_high_var(tmp_path: Path) -> None:
    """objective aplica penalización ×0.5 cuando VaR_95 > 0.08."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.8, sortino=1.0)
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.09)  # > 0.08

    trial = _mock_trial(seq_len=96, pred_len=20)
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_proc),
        patch("tune_hyperparams._current_time", return_value=ts_before),
    ):
        result = th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    # Sharpe base = 0.8; penalización VaR → 0.8 × 0.5 = 0.4
    assert abs(result - 0.4) < 1e-6


def test_objective_penalizes_negative_sortino(tmp_path: Path) -> None:
    """objective aplica penalización -0.3 cuando Sortino < 0."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.6, sortino=-0.1)
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.05)  # VaR OK

    trial = _mock_trial(seq_len=96, pred_len=20)
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_proc),
        patch("tune_hyperparams._current_time", return_value=ts_before),
    ):
        result = th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    # Sharpe base = 0.6; penalización Sortino → 0.6 - 0.3 = 0.3
    assert abs(result - 0.3) < 1e-6


def test_objective_returns_minus_one_on_subprocess_failure(tmp_path: Path) -> None:
    """objective retorna -1.0 si el subproceso falla (returncode != 0)."""
    trial = _mock_trial(seq_len=96, pred_len=20)
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "error simulado"

    with patch("subprocess.run", return_value=mock_proc):
        result = th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    assert result == -1.0


def test_objective_cmd_excludes_save_canonical(tmp_path: Path) -> None:
    """Los trials nunca incluyen --save-canonical en el comando de entrenamiento."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.7, sortino=1.0)
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.05)

    trial = _mock_trial(seq_len=96, pred_len=20)
    captured_cmd = []

    def mock_run(cmd, **kwargs):  # noqa: ANN001
        captured_cmd.extend(cmd)
        m = MagicMock()
        m.returncode = 0
        return m

    with (
        patch("subprocess.run", side_effect=mock_run),
        patch("tune_hyperparams._current_time", return_value=ts_before),
    ):
        th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    assert "--save-canonical" not in captured_cmd


def test_objective_cmd_includes_fase6_hp_flags(tmp_path: Path) -> None:
    """El subprocess recibe los 6 flags nuevos de la fase 6."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.7, sortino=1.0)
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.05)

    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    trial.suggest_categorical.side_effect = [
        96,
        20,
        64,
        0.5,
        2,
        1,
        4,
        64,
        48,
        0.1,
    ]
    trial.set_user_attr = MagicMock()
    captured_cmd: list[str] = []

    def mock_run(cmd, **kwargs):  # noqa: ANN001
        captured_cmd.extend(cmd)
        m = MagicMock()
        m.returncode = 0
        return m

    with (
        patch("subprocess.run", side_effect=mock_run),
        patch("tune_hyperparams._current_time", return_value=ts_before),
    ):
        th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    expected_flags = [
        "--e-layers",
        "--d-layers",
        "--n-flow-layers",
        "--flow-hidden-dim",
        "--label-len",
        "--dropout",
    ]
    for flag in expected_flags:
        assert flag in captured_cmd


def test_objective_cmd_includes_seed_and_compile_mode(tmp_path: Path) -> None:
    """El subprocess recibe --seed y --compile-mode cuando se pasan a objective()."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.5, sortino=0.8)
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.03)

    trial = _mock_trial(seq_len=96, pred_len=20)
    captured_cmd: list[str] = []

    def mock_run(cmd, **kwargs):  # noqa: ANN001
        captured_cmd.extend(cmd)
        m = MagicMock()
        m.returncode = 0
        return m

    with (
        patch("subprocess.run", side_effect=mock_run),
        patch("tune_hyperparams._current_time", return_value=ts_before),
    ):
        th.objective(
            trial,
            "data/MSFT_features.csv",
            4,
            tmp_path / "results",
            seed=7,
            compile_mode="",
        )

    # --seed 7 debe estar en el comando
    seed_idx = captured_cmd.index("--seed")
    assert captured_cmd[seed_idx + 1] == "7"

    # --compile-mode "" debe estar en el comando
    cm_idx = captured_cmd.index("--compile-mode")
    assert captured_cmd[cm_idx + 1] == ""


def test_objective_disables_wandb_in_subprocess_env(tmp_path: Path) -> None:
    """objective desactiva wandb para evitar bloqueos de red en trials."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.5, sortino=0.8)
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.03)

    trial = _mock_trial(seq_len=96, pred_len=20)
    captured_env: dict[str, str] = {}

    def mock_run(cmd, **kwargs):  # noqa: ANN001
        captured_env.update(kwargs["env"])
        m = MagicMock()
        m.returncode = 0
        return m

    with (
        patch("subprocess.run", side_effect=mock_run),
        patch("tune_hyperparams._current_time", return_value=ts_before),
    ):
        th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    assert captured_env["WANDB_MODE"] == "disabled"
    assert captured_env["WANDB_DISABLED"] == "true"
    assert captured_env["MPLBACKEND"] == "Agg"
    assert "MPLCONFIGDIR" in captured_env


def test_objective_logs_expanded_search_space(tmp_path: Path, caplog) -> None:
    """objective registra los HPs nuevos en el log del trial."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.5, sortino=0.8)
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.03)

    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    trial.suggest_categorical.side_effect = [
        96,
        20,
        64,
        0.5,
        2,
        1,
        4,
        64,
        48,
        0.1,
    ]
    trial.set_user_attr = MagicMock()

    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_proc),
        patch("tune_hyperparams._current_time", return_value=ts_before),
        caplog.at_level("INFO", logger="tune_hyperparams"),
    ):
        th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    assert any("label_len" in record.message for record in caplog.records)
    assert any("e_layers" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Tests del resumen de trials
# ---------------------------------------------------------------------------


def test_build_completed_trials_dataframe_includes_state() -> None:
    """_build_completed_trials_dataframe filtra usando la columna state sin romper."""
    study = optuna.create_study(direction="maximize")

    def _objective(trial: optuna.Trial) -> float:
        trial.suggest_categorical("seq_len", [96])
        trial.set_user_attr("sortino", 1.1)
        return 0.42

    study.optimize(_objective, n_trials=1)

    completed_df = th._build_completed_trials_dataframe(study)

    assert not completed_df.empty
    assert completed_df.iloc[0]["state"] == "COMPLETE"
    assert completed_df.iloc[0]["value"] == pytest.approx(0.42)


def test_count_search_space_combinations_reflects_phase6_space() -> None:
    """El cálculo del espacio total refleja solo combinaciones válidas tras poda."""
    assert th._count_search_space_combinations() == 31104


# ---------------------------------------------------------------------------
# Tests de download_extra_tickers
# ---------------------------------------------------------------------------


def test_download_extra_tickers_skips_existing(tmp_path: Path, monkeypatch) -> None:
    """download_extra_tickers omite los tickers que ya tienen CSV en data/."""
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Crear CSV existente para AMD
    (data_dir / "AMD_features.csv").write_text("col\n1\n2\n")

    called_for = []

    def mock_run(cmd, **kwargs):  # noqa: ANN001
        # Detectar el ticker del comando
        ticker = None
        for i, arg in enumerate(cmd):
            if arg == "--symbol" and i + 1 < len(cmd):
                ticker = cmd[i + 1]
                called_for.append(ticker)
                break
        if ticker is not None:
            (data_dir / f"{ticker}_features.csv").write_text(
                "date,Close\n2024-01-01,1\n"
            )
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("subprocess.run", side_effect=mock_run):
        th.download_extra_tickers()

    # AMD ya existe → no debe llamarse
    assert "AMD" not in called_for
    # Los otros 7 sí deben haberse intentado
    for ticker in th.EXTRA_TICKERS:
        if ticker != "AMD":
            assert ticker in called_for


def test_download_extra_tickers_continues_when_csv_is_not_generated(
    tmp_path: Path, monkeypatch
) -> None:
    """download_extra_tickers no debe fallar si el builder devuelve 0 sin crear CSV."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc):
        th.download_extra_tickers()


# ---------------------------------------------------------------------------
# Tests de _parse_probabilistic_csv (Épica 6)
# ---------------------------------------------------------------------------


def test_parse_probabilistic_csv_returns_empty_when_no_file(tmp_path: Path) -> None:
    """_parse_probabilistic_csv retorna {} cuando no existe ningún CSV probabilístico."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    after_ts = time.strftime("%Y%m%d_%H%M%S")

    result = th._parse_probabilistic_csv(results_dir, "NVDA_features", after_ts)

    assert result == {}


def test_parse_probabilistic_csv_reads_matching_file(tmp_path: Path) -> None:
    """_parse_probabilistic_csv lee correctamente un CSV probabilístico sintético."""
    # Crear CSV antes de registrar el timestamp de inicio
    after_ts_float = time.time() - 1
    after_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(after_ts_float))

    _write_probabilistic_csv(
        tmp_path,
        "NVDA_features",
        pinball_p50=0.025,
        coverage_80=0.81,
        interval_width_80=0.06,
        crps=0.035,
    )

    result = th._parse_probabilistic_csv(
        tmp_path / "results", "NVDA_features", after_ts
    )

    assert "pinball_p50" in result
    assert "coverage_80" in result
    assert "interval_width_80" in result
    assert "crps" in result
    assert abs(result["pinball_p50"] - 0.025) < 1e-6
    assert abs(result["coverage_80"] - 0.81) < 1e-6
    assert abs(result["interval_width_80"] - 0.06) < 1e-6
    assert abs(result["crps"] - 0.035) < 1e-6


def test_parse_probabilistic_csv_returns_empty_for_invalid_timestamp(
    tmp_path: Path,
) -> None:
    """_parse_probabilistic_csv retorna {} si el formato de after_ts es inválido."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    result = th._parse_probabilistic_csv(results_dir, "NVDA_features", "INVALID_TS")

    assert result == {}


# ---------------------------------------------------------------------------
# Tests de _compose_trial_score (Épica 6)
# ---------------------------------------------------------------------------


def _make_portfolio(sharpe: float = 0.7, sortino: float = 1.0) -> dict:
    """Construye un dict de métricas de portafolio mínimo para tests."""
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": -0.3,
        "volatility": 0.02,
    }


def _make_prob_metrics(
    pinball_p50: float = 0.02,
    coverage_80: float = 0.80,
) -> dict:
    """Construye un dict de métricas probabilísticas mínimo para tests."""
    return {
        "pinball_p50": pinball_p50,
        "coverage_80": coverage_80,
        "interval_width_80": 0.05,
        "crps": 0.03,
    }


def test_compose_trial_score_sharpe_mode() -> None:
    """mode='sharpe' retorna directamente el Sharpe, ignorando prob_metrics."""
    portfolio = _make_portfolio(sharpe=0.75)
    risk = {"var_95": 0.05}
    prob = _make_prob_metrics(pinball_p50=0.1, coverage_80=0.50)

    score = th._compose_trial_score(portfolio, risk, prob, mode="sharpe")

    # En modo sharpe, ignora completamente las métricas probabilísticas
    assert abs(score - 0.75) < 1e-6


def test_compose_trial_score_composite_mode_returns_finite(tmp_path: Path) -> None:
    """mode='composite' con métricas válidas retorna un score finito."""
    portfolio = _make_portfolio(sharpe=0.65, sortino=1.1)
    risk = {"var_95": 0.05}
    prob = _make_prob_metrics(pinball_p50=0.02, coverage_80=0.80)

    score = th._compose_trial_score(portfolio, risk, prob, mode="composite")

    # El score debe ser finito y mayor que -1 (no es el centinela de fallo)
    assert math.isfinite(score)
    assert score > -1.0


def test_compose_trial_score_empty_prob_metrics_falls_back_to_sharpe() -> None:
    """mode='composite' con prob_metrics={} hace fallback a Sharpe puro sin NaN."""
    portfolio = _make_portfolio(sharpe=0.55)
    risk = {"var_95": 0.05}

    score = th._compose_trial_score(portfolio, risk, prob_metrics={}, mode="composite")

    # Fallback: debe ser igual al Sharpe puro
    assert abs(score - 0.55) < 1e-6
    assert math.isfinite(score)


def test_compose_trial_score_no_nan() -> None:
    """_compose_trial_score nunca retorna NaN con cualquier combinación de inputs."""
    inputs = [
        # (portfolio, risk, prob, mode)
        (_make_portfolio(sharpe=float("nan")), {}, {}, "sharpe"),
        (_make_portfolio(sharpe=0.0), {}, {}, "sharpe"),
        (_make_portfolio(sharpe=-1.0), {}, {}, "composite"),
        (_make_portfolio(sharpe=0.5), {}, _make_prob_metrics(), "composite"),
        (
            _make_portfolio(sharpe=float("nan")),
            {},
            _make_prob_metrics(pinball_p50=float("nan")),
            "composite",
        ),
        (_make_portfolio(sharpe=0.3), {}, _make_prob_metrics(), "multi-objective"),
    ]

    for portfolio, risk, prob, mode in inputs:
        score = th._compose_trial_score(portfolio, risk, prob, mode=mode)
        assert math.isfinite(score), (
            f"Score NaN/inf para mode={mode}, portfolio={portfolio}, prob={prob}"
        )


def test_compose_trial_score_coverage_penalty() -> None:
    """mode='composite' penaliza coverage alejado del 80% nominal."""
    portfolio = _make_portfolio(sharpe=0.65)
    risk = {"var_95": 0.05}

    # Coverage perfecto (0.80)
    prob_perfect = _make_prob_metrics(coverage_80=0.80)
    score_perfect = th._compose_trial_score(
        portfolio, risk, prob_perfect, mode="composite"
    )

    # Coverage muy malo (0.20)
    prob_bad = _make_prob_metrics(coverage_80=0.20)
    score_bad = th._compose_trial_score(portfolio, risk, prob_bad, mode="composite")

    # Coverage perfecto debe dar score más alto
    assert score_perfect > score_bad


# ---------------------------------------------------------------------------
# Tests de flags nuevos (AR #3, #4, #8, #9)
# ---------------------------------------------------------------------------


def test_results_and_optuna_dirs_created_if_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """main() crea results/ y optuna_studies/ aunque no existan previamente."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "MSFT_features.csv").write_text("date,Close\n2024-01-01,1\n")

    mock_study = MagicMock()
    mock_study.trials = []
    mock_study.optimize = MagicMock()

    with (
        patch("optuna.create_study", return_value=mock_study),
        patch(
            "sys.argv",
            [
                "tune_hyperparams.py",
                "--csv",
                str(tmp_path / "data" / "MSFT_features.csv"),
                "--n-trials",
                "0",
            ],
        ),
    ):
        th.main()

    assert (tmp_path / "results").is_dir(), "results/ debe crearse si no existe"
    assert (tmp_path / "optuna_studies").is_dir(), (
        "optuna_studies/ debe crearse si no existe"
    )


def test_clean_results_removes_stale_csvs(tmp_path: Path, monkeypatch) -> None:
    """--clean-results elimina todos los CSVs de results/ antes de optimizar."""
    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stale = [
        results_dir / "portfolio_metrics_20260101_120000.csv",
        results_dir / "risk_metrics_20260101_120000.csv",
    ]
    for f in stale:
        f.write_text("metric,value\nsharpe_ratio,-1.0\n")

    (tmp_path / "data").mkdir()
    csv_path = tmp_path / "data" / "MSFT_features.csv"
    csv_path.write_text("date,Close\n2024-01-01,1\n")

    mock_study = MagicMock()
    mock_study.trials = []
    mock_study.optimize = MagicMock()

    with (
        patch("optuna.create_study", return_value=mock_study),
        patch(
            "sys.argv",
            [
                "tune_hyperparams.py",
                "--csv",
                str(csv_path),
                "--n-trials",
                "0",
                "--clean-results",
            ],
        ),
    ):
        th.main()

    remaining = list(results_dir.glob("*.csv"))
    assert remaining == [], (
        f"--clean-results debe eliminar todos los CSVs; quedan: {remaining}"
    )


def test_sampler_receives_configured_params(tmp_path: Path, monkeypatch) -> None:
    """TPESampler recibe seed y n_startup_trials según los flags CLI."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    csv_path = tmp_path / "data" / "MSFT_features.csv"
    csv_path.write_text("date,Close\n2024-01-01,1\n")

    captured: dict = {}

    def mock_tpe(**kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return MagicMock()

    mock_study = MagicMock()
    mock_study.trials = []
    mock_study.optimize = MagicMock()

    with (
        patch("optuna.samplers.TPESampler", side_effect=mock_tpe),
        patch("optuna.create_study", return_value=mock_study),
        patch(
            "sys.argv",
            [
                "tune_hyperparams.py",
                "--csv",
                str(csv_path),
                "--n-trials",
                "0",
                "--n-startup-trials",
                "10",
                "--sampler-seed",
                "99",
            ],
        ),
    ):
        th.main()

    assert captured.get("seed") == 99
    assert captured.get("n_startup_trials") == 10


def test_enqueue_canonical_calls_enqueue_trial(tmp_path: Path, monkeypatch) -> None:
    """--enqueue-canonical (default=True) llama enqueue_trial con la config canónica exacta."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    csv_path = tmp_path / "data" / "MSFT_features.csv"
    csv_path.write_text("date,Close\n2024-01-01,1\n")

    mock_study = MagicMock()
    mock_study.trials = []
    mock_study.optimize = MagicMock()

    with (
        patch("optuna.create_study", return_value=mock_study),
        patch(
            "sys.argv",
            [
                "tune_hyperparams.py",
                "--csv",
                str(csv_path),
                "--n-trials",
                "0",
            ],
        ),
    ):
        th.main()

    mock_study.enqueue_trial.assert_called_once_with(
        {
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
    )


def test_run_best_params_backfills_phase6_defaults_for_legacy_studies(
    tmp_path: Path,
) -> None:
    """_run_best_params rellena HPs Fase 6 ausentes para estudios legacy."""
    captured_cmd: list[str] = []

    def mock_run(cmd, **kwargs):  # noqa: ANN001
        captured_cmd.extend(cmd)
        result = MagicMock()
        result.returncode = 0
        return result

    with patch("subprocess.run", side_effect=mock_run):
        th._run_best_params(
            csv_path=str(tmp_path / "MSFT_features.csv"),
            best_params={
                "seq_len": 96,
                "pred_len": 20,
                "batch_size": 64,
                "gradient_clip_norm": 0.5,
            },
            n_splits=4,
            save_canonical=False,
        )

    assert captured_cmd[captured_cmd.index("--e-layers") + 1] == "2"
    assert captured_cmd[captured_cmd.index("--d-layers") + 1] == "1"
    assert captured_cmd[captured_cmd.index("--n-flow-layers") + 1] == "4"
    assert captured_cmd[captured_cmd.index("--flow-hidden-dim") + 1] == "64"
    assert captured_cmd[captured_cmd.index("--label-len") + 1] == "48"
    assert captured_cmd[captured_cmd.index("--dropout") + 1] == "0.1"


def test_no_enqueue_canonical_skips_enqueue_trial(tmp_path: Path, monkeypatch) -> None:
    """--no-enqueue-canonical omite la llamada a enqueue_trial."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    csv_path = tmp_path / "data" / "MSFT_features.csv"
    csv_path.write_text("date,Close\n2024-01-01,1\n")

    mock_study = MagicMock()
    mock_study.trials = []
    mock_study.optimize = MagicMock()

    with (
        patch("optuna.create_study", return_value=mock_study),
        patch(
            "sys.argv",
            [
                "tune_hyperparams.py",
                "--csv",
                str(csv_path),
                "--n-trials",
                "0",
                "--no-enqueue-canonical",
            ],
        ),
    ):
        th.main()

    mock_study.enqueue_trial.assert_not_called()
