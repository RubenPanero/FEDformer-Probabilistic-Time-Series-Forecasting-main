"""Probabilistic validation of FEDformer model on inference CSV output.

Metrics computed:
  - Empirical coverage p10-p90   (nominal: 80%)
  - Pinball loss at p10, p50, p90
  - MAE on p50 predictions
  - Interval Score 80% (Winkler)
  - Directional accuracy at step 1

Example:
    # Evaluate GOOGL predictions vs GOOGL ground truth
    python3 scripts/validate_forecast.py \\
        --pred results/inference_googl.csv \\
        --features data/GOOGL_features.csv \\
        --ticker GOOGL \\
        --seq-len 96
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


# ── funciones de métricas ──────────────────────────────────────────────────────


def compute_coverage(gt: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Fraction of GT values that fall within the interval [lower, upper]."""
    return float(((gt >= lower) & (gt <= upper)).mean())


def pinball_loss(gt: np.ndarray, q_hat: np.ndarray, q: float) -> float:
    """Pinball loss (quantile loss) at quantile q."""
    gt = np.asarray(gt, dtype=float)
    q_hat = np.asarray(q_hat, dtype=float)
    e = gt - q_hat
    return float(np.where(e >= 0, q * e, (q - 1) * e).mean())


def mae(gt: np.ndarray, p50: np.ndarray) -> float:
    """MAE on median p50 predictions."""
    return float(
        np.abs(np.asarray(gt, dtype=float) - np.asarray(p50, dtype=float)).mean()
    )


def interval_score(
    gt: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Winkler Interval Score for a (1-alpha)*100% interval.

    IS = (upper-lower) + (2/alpha)*max(lower-gt, 0) + (2/alpha)*max(gt-upper, 0)
    """
    gt = np.asarray(gt, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    penalty_low = (2 / alpha) * np.maximum(lower - gt, 0.0)
    penalty_high = (2 / alpha) * np.maximum(gt - upper, 0.0)
    return float((width + penalty_low + penalty_high).mean())


def directional_accuracy(
    gt_next: np.ndarray,
    p50_next: np.ndarray,
    last_price: np.ndarray,
) -> float:
    """Fraction of times the sign of (p50-last) matches the sign of (gt-last)."""
    gt_dir = np.sign(
        np.asarray(gt_next, dtype=float) - np.asarray(last_price, dtype=float)
    )
    pred_dir = np.sign(
        np.asarray(p50_next, dtype=float) - np.asarray(last_price, dtype=float)
    )
    return float((gt_dir == pred_dir).mean())


# ── función principal de cálculo ───────────────────────────────────────────────


def compute_all_metrics(
    pred_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target: str = "Close",
    seq_len: int = 96,
) -> dict[str, float]:
    """Computes all validation metrics.

    Args:
        pred_df:     DataFrame from inference CSV exported by the Inference API.
        features_df: Original features DataFrame ('date' column + target).
        target:      Target column name (default 'Close').
        seq_len:     Input sequence length of the model.

    Returns:
        Dictionary with all computed metrics.
    """
    gt = pred_df[f"gt_{target}"]
    p10 = pred_df[f"p10_{target}"]
    p50 = pred_df[f"p50_{target}"]
    p90 = pred_df[f"p90_{target}"]

    # Directional accuracy: first step of each window (can be 0 or 1)
    first_step = int(pred_df["step"].min())
    step1 = pred_df[pred_df["step"] == first_step].reset_index(drop=True)

    # In returns space, the reference for direction is 0 (neutral return).
    # In prices space, it's the last known price from the input sequence.
    predictions_are_returns = gt.abs().max() < 5.0  # heuristic: returns << prices
    if predictions_are_returns:
        last_prices = np.zeros(len(step1))
    else:
        windows_step1 = step1["window"].values
        last_prices = features_df[target].iloc[windows_step1 + seq_len - 1].values

    return {
        "coverage_p10_p90": compute_coverage(gt, p10, p90),
        "pinball_p10": pinball_loss(gt.values, p10.values, 0.10),
        "pinball_p50": pinball_loss(gt.values, p50.values, 0.50),
        "pinball_p90": pinball_loss(gt.values, p90.values, 0.90),
        "mae_p50": mae(gt.values, p50.values),
        "interval_score_80": interval_score(
            gt.values, p10.values, p90.values, alpha=0.20
        ),
        "directional_acc_step1": directional_accuracy(
            step1[f"gt_{target}"].values,
            step1[f"p50_{target}"].values,
            last_prices,
        ),
        "n_windows": int(pred_df["window"].nunique()),
        "n_obs": len(pred_df),
    }


# ── reporte ────────────────────────────────────────────────────────────────────


def print_report(metrics: dict[str, float], ticker: str, target: str) -> None:
    """Prints validation report in human-readable format."""
    cov = metrics["coverage_p10_p90"]
    print(f"\n{'=' * 55}")
    print(f"  Probabilistic Validation — {ticker} / {target}")
    print("=" * 55)
    print(f"  Windows evaluated  : {metrics['n_windows']:,}")
    print(f"  Observations       : {metrics['n_obs']:,}")
    print("-" * 55)
    print(f"  Coverage p10-p90   : {cov:.1%}  (nominal: 80%)")
    gap = cov - 0.80
    tag = (
        "✓ calibrated"
        if abs(gap) < 0.05
        else ("▲ overestimated" if gap > 0 else "▼ underestimated")
    )
    print(f"    → gap vs nominal : {gap:+.1%}  {tag}")
    print("-" * 55)
    print(f"  Pinball p10        : {metrics['pinball_p10']:.4f}")
    print(f"  Pinball p50        : {metrics['pinball_p50']:.4f}")
    print(f"  Pinball p90        : {metrics['pinball_p90']:.4f}")
    print(f"  MAE p50            : {metrics['mae_p50']:.4f}")
    print(
        f"  Interval Score 80% : {metrics['interval_score_80']:.4f}  (lower = better)"
    )
    print("-" * 55)
    print(f"  Directional accuracy (step 1): {metrics['directional_acc_step1']:.1%}")
    print(f"{'=' * 55}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FEDformer Probabilistic Validation")
    parser.add_argument(
        "--pred", required=True, help="Predictions CSV (inference output)"
    )
    parser.add_argument("--features", required=True, help="Original features CSV")
    parser.add_argument(
        "--target", default="Close", help="Target column (default: Close)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=96, help="Model seq_len (default: 96)"
    )
    parser.add_argument("--ticker", default="NVDA", help="Ticker for report")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    pred_df = pd.read_csv(args.pred)
    features_df = pd.read_csv(args.features)

    metrics = compute_all_metrics(pred_df, features_df, args.target, args.seq_len)
    print_report(metrics, args.ticker, args.target)
