"""
Monte Carlo analysis for TrendEMAPullback strategy using walk-forward OOS trades.

This module performs Monte Carlo resampling of out-of-sample trade results to assess:
- Drawdown risk under different scenarios
- Robustness of edge across different trade orderings
- Safe risk-per-trade settings for FTMO-style prop accounts

Usage:
    python monte_carlo_trend.py --input "walkforward_*_OOS_trades.csv" \
        --instruments "USDCAD,USDJPY" --n-paths 10000 --block-size 5
"""

from __future__ import annotations

import argparse
import glob
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
#                         DATA LOADING & VALIDATION
# ============================================================================

def load_trades(input_pattern: str) -> Tuple[pd.DataFrame, str]:
    """
    Load OOS trades from CSV file(s) matching the input pattern.

    Args:
        input_pattern: Glob pattern for CSV files (e.g., "walkforward_*_OOS_trades.csv")

    Returns:
        (df, r_column_name): DataFrame with all trades and the detected R column name

    Raises:
        FileNotFoundError: If no files match the pattern
        ValueError: If required columns are missing or no R column is found
    """
    # Expand glob pattern
    files = glob.glob(input_pattern)
    if not files:
        raise FileNotFoundError(
            f"No files found matching pattern: {input_pattern}\n"
            f"Make sure you've run walkforward_runner.py first to generate OOS trade CSVs."
        )

    print(f"[load_trades] Found {len(files)} file(s) matching pattern '{input_pattern}'")
    for f in files:
        print(f"  - {f}")

    # Load and concatenate all CSVs
    dfs = []
    for filepath in files:
        try:
            df_tmp = pd.read_csv(filepath)
            dfs.append(df_tmp)
        except Exception as e:
            warnings.warn(f"Failed to load {filepath}: {e}")

    if not dfs:
        raise ValueError("No valid CSV files could be loaded")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[load_trades] Loaded {len(df)} total trades from {len(dfs)} file(s)")

    # Parse entry_time as datetime if present
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")

    # Validate required columns
    required_cols = ["instrument", "strategy_name"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Detect R column
    r_col = _detect_r_column(df)
    print(f"[load_trades] Detected R column: '{r_col}'")

    return df, r_col


def _detect_r_column(df: pd.DataFrame) -> str:
    """
    Robustly detect the R-multiple column in the trades DataFrame.

    Priority:
    1. Exact match: "R"
    2. Case-insensitive match containing "R" that is numeric
    3. Raise error if nothing found

    Args:
        df: Trades DataFrame

    Returns:
        Column name containing R multiples

    Raises:
        ValueError: If no suitable R column found
    """
    # First try exact match
    if "R" in df.columns and pd.api.types.is_numeric_dtype(df["R"]):
        return "R"

    # Try case-insensitive partial match
    candidates = [
        col for col in df.columns
        if "r" in col.lower()
        and pd.api.types.is_numeric_dtype(df[col])
        and col.lower() not in ["direction", "spread", "error"]  # exclude false positives
    ]

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Prefer shorter names (R_net over R_multiple_adjusted)
        candidates.sort(key=len)
        warnings.warn(
            f"Multiple R-like columns found: {candidates}. Using '{candidates[0]}'"
        )
        return candidates[0]

    # Nothing found
    raise ValueError(
        f"No suitable R column found in DataFrame.\n"
        f"Available numeric columns: {[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]}\n"
        f"Make sure the CSV contains trade R-multiples."
    )


# ============================================================================
#                         MONTE CARLO SIMULATION
# ============================================================================

def monte_carlo_paths(
    returns: np.ndarray,
    n_paths: int,
    block_size: int = 1,
    max_trades: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate Monte Carlo equity paths by resampling trade returns.

    Args:
        returns: 1D numpy array of R-multiples from OOS trades (single instrument)
        n_paths: Number of Monte Carlo paths to simulate
        block_size: Block bootstrap length in trades (1 = iid sampling)
        max_trades: Optional cap on path length; if None, use len(returns)
        rng: NumPy random number generator for reproducibility

    Returns:
        equity_paths: shape (n_paths, T), cumulative equity in R units (starting at 0)

    Notes:
        - Each path represents a possible ordering of the empirical trade distribution
        - Block size > 1 preserves local correlation structure (e.g., regime persistence)
        - Equity starts at 0; equity[t] = sum(returns[0:t+1])
    """
    if rng is None:
        rng = np.random.default_rng()

    T = max_trades if max_trades is not None else len(returns)
    equity_paths = np.zeros((n_paths, T), dtype=np.float64)

    for path_idx in range(n_paths):
        if block_size == 1:
            # Simple iid resampling
            sampled_returns = rng.choice(returns, size=T, replace=True)
        else:
            # Block bootstrap
            sampled_returns = _block_bootstrap(returns, T, block_size, rng)

        # Compute cumulative equity for this path
        equity_paths[path_idx, :] = np.cumsum(sampled_returns)

    return equity_paths


def _block_bootstrap(
    returns: np.ndarray,
    target_length: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simple block bootstrap: sample blocks of consecutive trades.

    Args:
        returns: Original returns array
        target_length: Desired length of resampled series
        block_size: Length of each block
        rng: Random number generator

    Returns:
        Resampled returns of length target_length
    """
    n = len(returns)
    if block_size >= n:
        # Block size too large; just resample with replacement
        return rng.choice(returns, size=target_length, replace=True)

    blocks = []
    total_len = 0

    while total_len < target_length:
        # Random starting index
        start_idx = rng.integers(0, n - block_size + 1)
        block = returns[start_idx : start_idx + block_size]
        blocks.append(block)
        total_len += len(block)

    # Concatenate and truncate to target length
    resampled = np.concatenate(blocks)[:target_length]
    return resampled


# ============================================================================
#                         STATISTICS & ANALYSIS
# ============================================================================

def summarise_mc_paths(equity_paths: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics over Monte Carlo equity paths (in R).

    Args:
        equity_paths: shape (n_paths, T), cumulative equity in R

    Returns:
        Dictionary with keys:
        - n_paths: number of simulated paths
        - T: path length (number of trades)
        - median_final_R: median final equity
        - p05_final_R: 5th percentile final equity
        - p95_final_R: 95th percentile final equity
        - median_maxDD_R: median max drawdown in R
        - p95_maxDD_R: 95th percentile max drawdown (worst case)
        - median_losing_streak: median longest losing streak
        - p95_losing_streak: 95th percentile losing streak
    """
    n_paths, T = equity_paths.shape

    # Final equity distribution
    final_equity = equity_paths[:, -1]
    median_final_R = float(np.median(final_equity))
    p05_final_R = float(np.percentile(final_equity, 5))
    p95_final_R = float(np.percentile(final_equity, 95))

    # Max drawdown per path
    max_dds = np.zeros(n_paths)
    for i in range(n_paths):
        equity_curve = equity_paths[i, :]
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve - running_max
        max_dds[i] = np.min(drawdowns)

    median_maxDD_R = float(np.median(max_dds))
    p95_maxDD_R = float(np.percentile(max_dds, 5))  # 5th percentile = worst 5% of DDs

    # Losing streaks (count consecutive losing trades per path)
    # For this, we need per-trade returns, not cumulative equity
    # Reconstruct returns from equity diff
    losing_streaks = np.zeros(n_paths)
    for i in range(n_paths):
        if T == 1:
            returns_i = equity_paths[i, :]
        else:
            returns_i = np.diff(equity_paths[i, :], prepend=0)

        # Find longest streak of negative returns
        losing_streaks[i] = _longest_losing_streak(returns_i)

    median_losing_streak = float(np.median(losing_streaks))
    p95_losing_streak = float(np.percentile(losing_streaks, 95))

    return {
        "n_paths": n_paths,
        "T": T,
        "median_final_R": median_final_R,
        "p05_final_R": p05_final_R,
        "p95_final_R": p95_final_R,
        "median_maxDD_R": median_maxDD_R,
        "p95_maxDD_R": p95_maxDD_R,
        "median_losing_streak": median_losing_streak,
        "p95_losing_streak": p95_losing_streak,
    }


def _longest_losing_streak(returns: np.ndarray) -> int:
    """
    Compute the longest consecutive run of losing trades (R < 0).

    Args:
        returns: 1D array of per-trade R multiples

    Returns:
        Length of longest losing streak
    """
    if len(returns) == 0:
        return 0

    max_streak = 0
    current_streak = 0

    for r in returns:
        if r < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


# ============================================================================
#                         R TO EQUITY % CONVERSION
# ============================================================================

def scale_R_to_equity(
    equity_paths_R: np.ndarray,
    risk_per_trade_pct: float,
) -> np.ndarray:
    """
    Convert R-based equity paths into percentage account PnL paths.

    If each trade risks 'risk_per_trade_pct' of account (e.g., 0.5 for 0.5%),
    then equity in % = equity_in_R * risk_per_trade_pct.

    Args:
        equity_paths_R: Monte Carlo paths in R units, shape (n_paths, T)
        risk_per_trade_pct: Risk per trade as % of account (e.g., 0.5 for 0.5%)

    Returns:
        equity_paths_pct: Same shape, but in % of initial account

    Example:
        If a path ends at +10R and risk_per_trade = 0.5%,
        then final equity = +10 * 0.5 = +5.0% gain on account.

    Notes:
        This is a simplified linear mapping. In reality, compounding effects
        would require iterative calculation of position size based on current equity.
        For FTMO-style risk limits (typically < 1% per trade), the linear approximation
        is reasonable for short-to-medium term paths.
    """
    return equity_paths_R * risk_per_trade_pct


# ============================================================================
#                         COMMAND LINE INTERFACE
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo analysis for TrendEMAPullback strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="walkforward_*_OOS_trades.csv",
        help="Glob pattern for OOS trade CSV files",
    )

    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated list of instruments to analyze (e.g., 'USDCAD,USDJPY'). "
        "If not provided, analyze all instruments in data.",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="TrendEMAPullback",
        help="Strategy name to filter trades",
    )

    parser.add_argument(
        "--n-paths",
        type=int,
        default=10000,
        help="Number of Monte Carlo paths to simulate per instrument",
    )

    parser.add_argument(
        "--max-trades",
        type=int,
        default=None,
        help="Optional cap on number of trades per path (default: use all available trades)",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Block bootstrap size in trades (1 = iid resampling, >1 = block bootstrap)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--output-summary",
        type=str,
        default="mc_trend_summary.csv",
        help="Output CSV path for summary statistics",
    )

    parser.add_argument(
        "--output-paths",
        type=str,
        default=None,
        help="Optional: save sample equity paths to CSV (instrument, path_id, t, equity_R)",
    )

    parser.add_argument(
        "--min-trades",
        type=int,
        default=50,
        help="Minimum number of trades required to run MC for an instrument",
    )

    return parser.parse_args()


def main():
    """Main Monte Carlo analysis workflow."""
    args = parse_args()

    print("=" * 80)
    print("Monte Carlo Analysis for TrendEMAPullback Strategy")
    print("=" * 80)
    print(f"Input pattern: {args.input}")
    print(f"Strategy filter: {args.strategy}")
    print(f"Instruments: {args.instruments if args.instruments else 'ALL'}")
    print(f"MC paths per instrument: {args.n_paths:,}")
    print(f"Block size: {args.block_size}")
    print(f"Random seed: {args.seed}")
    print(f"Min trades per instrument: {args.min_trades}")
    print("=" * 80)

    # 1. Load trades
    df_all, r_col = load_trades(args.input)

    # 2. Filter by strategy
    df_strategy = df_all[df_all["strategy_name"] == args.strategy].copy()
    if df_strategy.empty:
        raise ValueError(
            f"No trades found for strategy '{args.strategy}'. "
            f"Available strategies: {df_all['strategy_name'].unique()}"
        )

    print(f"\n[main] Filtered to {len(df_strategy)} trades for strategy '{args.strategy}'")

    # 3. Filter by instruments if specified
    if args.instruments:
        instrument_list = [s.strip() for s in args.instruments.split(",")]
        df_strategy = df_strategy[df_strategy["instrument"].isin(instrument_list)].copy()
        if df_strategy.empty:
            raise ValueError(
                f"No trades found for instruments: {instrument_list}. "
                f"Available: {df_all['instrument'].unique()}"
            )
        print(f"[main] Filtered to instruments: {instrument_list}")

    instruments = df_strategy["instrument"].unique()
    print(f"[main] Running MC for {len(instruments)} instrument(s): {list(instruments)}")

    # 4. Run MC per instrument
    summary_rows = []
    all_sample_paths = []

    for inst_idx, instrument in enumerate(instruments):
        print(f"\n{'=' * 60}")
        print(f"[{inst_idx + 1}/{len(instruments)}] Instrument: {instrument}")
        print('=' * 60)

        df_inst = df_strategy[df_strategy["instrument"] == instrument].copy()
        returns = df_inst[r_col].dropna().to_numpy()

        n_trades = len(returns)
        print(f"  OOS trades: {n_trades}")

        if n_trades < args.min_trades:
            print(f"  WARNING: Only {n_trades} trades (< {args.min_trades}). Skipping MC.")
            continue

        print(f"  Mean R: {returns.mean():.3f}")
        print(f"  Std R: {returns.std():.3f}")
        print(f"  Sharpe (approx): {returns.mean() / returns.std() if returns.std() > 0 else 0:.3f}")

        # Create instrument-specific RNG for reproducibility
        inst_seed = args.seed + inst_idx
        rng = np.random.default_rng(inst_seed)

        # Run Monte Carlo
        print(f"  Running {args.n_paths:,} Monte Carlo paths...")
        equity_paths = monte_carlo_paths(
            returns=returns,
            n_paths=args.n_paths,
            block_size=args.block_size,
            max_trades=args.max_trades,
            rng=rng,
        )

        # Compute summary stats
        stats = summarise_mc_paths(equity_paths)
        stats["instrument"] = instrument
        stats["strategy"] = args.strategy
        stats["n_trades_oos"] = n_trades
        stats["mean_R_oos"] = float(returns.mean())
        stats["std_R_oos"] = float(returns.std())

        summary_rows.append(stats)

        print(f"  Results:")
        print(f"    Median final R: {stats['median_final_R']:.2f}R")
        print(f"    P05-P95 final R: [{stats['p05_final_R']:.2f}R, {stats['p95_final_R']:.2f}R]")
        print(f"    Median max DD: {stats['median_maxDD_R']:.2f}R")
        print(f"    P95 max DD (worst 5%): {stats['p95_maxDD_R']:.2f}R")
        print(f"    Median losing streak: {stats['median_losing_streak']:.0f} trades")
        print(f"    P95 losing streak: {stats['p95_losing_streak']:.0f} trades")

        # Optionally save sample paths
        if args.output_paths:
            n_sample_paths = min(100, args.n_paths)
            for path_id in range(n_sample_paths):
                for t in range(equity_paths.shape[1]):
                    all_sample_paths.append({
                        "instrument": instrument,
                        "path_id": path_id,
                        "t": t,
                        "equity_R": equity_paths[path_id, t],
                    })

    # 5. Save results
    if not summary_rows:
        print("\n[main] No instruments had sufficient trades for MC analysis.")
        return

    df_summary = pd.DataFrame(summary_rows)

    # Reorder columns for readability
    col_order = [
        "instrument",
        "strategy",
        "n_trades_oos",
        "mean_R_oos",
        "std_R_oos",
        "n_paths",
        "T",
        "median_final_R",
        "p05_final_R",
        "p95_final_R",
        "median_maxDD_R",
        "p95_maxDD_R",
        "median_losing_streak",
        "p95_losing_streak",
    ]
    df_summary = df_summary[col_order]

    df_summary.to_csv(args.output_summary, index=False)
    print(f"\n{'=' * 80}")
    print(f"Summary saved to: {args.output_summary}")
    print('=' * 80)
    print("\nSummary Statistics:")
    print(df_summary.to_string(index=False))

    if args.output_paths and all_sample_paths:
        df_paths = pd.DataFrame(all_sample_paths)
        df_paths.to_csv(args.output_paths, index=False)
        print(f"\nSample equity paths saved to: {args.output_paths}")
        print(f"  (First {min(100, args.n_paths)} paths per instrument)")


if __name__ == "__main__":
    main()


# Example usage:
# python monte_carlo_trend.py --input "walkforward_*_OOS_trades.csv" \
#     --instruments "USDCAD,USDJPY" --n-paths 10000 --block-size 5 \
#     --output-summary "mc_trend_summary_fx.csv"
