# portfolio_mc_analysis.py
"""
Portfolio-level Monte Carlo simulation.

Runs Monte Carlo resampling at the portfolio level, combining trades from
multiple instruments with equal-weight risk allocation.

Usage:
    python portfolio_mc_analysis.py --strategy TrendEMAPullback --basket-yaml live_basket_phase2.yml
"""

import argparse
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ============================================================================
#                         DATA LOADING
# ============================================================================

def load_basket_config(yaml_path: str, include_optional: bool = False) -> List[str]:
    """
    Load instrument basket from YAML configuration.

    Args:
        yaml_path: Path to basket YAML file
        include_optional: If True, include optional instruments

    Returns:
        List of instrument names
    """
    if not Path(yaml_path).exists():
        raise FileNotFoundError(f"Basket config not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    if "instruments" not in config:
        raise ValueError(f"Invalid YAML: missing 'instruments' key in {yaml_path}")

    instruments = []

    # Handle both formats: simple list or core/optional dict
    if isinstance(config["instruments"], list):
        instruments = config["instruments"]
    elif isinstance(config["instruments"], dict):
        if "core" in config["instruments"]:
            instruments.extend(config["instruments"]["core"])
        if include_optional and "optional" in config["instruments"]:
            instruments.extend(config["instruments"]["optional"])
    else:
        raise ValueError(f"Invalid YAML: 'instruments' must be list or dict with 'core' key")

    if not instruments:
        raise ValueError(f"No instruments found in {yaml_path}")

    return instruments


def load_trades_by_instrument(
    pattern: str,
    strategy: str,
    instruments: List[str],
) -> Dict[str, np.ndarray]:
    """
    Load OOS trades and extract R-series for each instrument.

    Args:
        pattern: Glob pattern for CSV files
        strategy: Strategy name to filter
        instruments: List of instruments to include

    Returns:
        Dictionary mapping instrument name to numpy array of R-multiples
    """
    # Expand glob pattern
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No files found matching pattern: {pattern}\n"
            f"Run walkforward_multi_instrument.py first."
        )

    print(f"[load_trades] Found {len(files)} file(s) matching pattern '{pattern}'")

    # Load and concatenate
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

    # Validate columns
    required_cols = ["instrument", "R"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter by strategy if column exists
    if "strategy_name" in df.columns:
        df = df[df["strategy_name"] == strategy].copy()
        print(f"[load_trades] Filtered to strategy '{strategy}': {len(df)} trades")

    # Filter by instruments
    df = df[df["instrument"].isin(instruments)].copy()
    print(f"[load_trades] Filtered to {len(instruments)} instruments: {len(df)} trades")

    if len(df) == 0:
        raise ValueError("No trades remaining after filtering!")

    # Extract R-series for each instrument
    R_by_instrument = {}
    for instrument in instruments:
        df_inst = df[df["instrument"] == instrument]
        if len(df_inst) > 0:
            R_by_instrument[instrument] = df_inst["R"].values
            print(f"  {instrument}: {len(df_inst)} trades, mean R = {df_inst['R'].mean():.3f}")
        else:
            warnings.warn(f"No trades found for {instrument}")

    if not R_by_instrument:
        raise ValueError("No instruments have trades!")

    return R_by_instrument


# ============================================================================
#                         PORTFOLIO MONTE CARLO
# ============================================================================

def run_portfolio_mc(
    R_by_instrument: Dict[str, np.ndarray],
    n_paths: int,
    block_size: int = 1,
    seed: int = 42,
    path_length: Optional[int] = None,
) -> np.ndarray:
    """
    Run portfolio-level Monte Carlo simulation.

    For each path:
    - Resample returns for each instrument independently
    - Combine using equal-weight (1/N)
    - Compute cumulative portfolio equity

    Args:
        R_by_instrument: Dict mapping instrument name to R-multiples array
        n_paths: Number of Monte Carlo paths
        block_size: Block bootstrap size (1 = iid)
        seed: Random seed
        path_length: Optional fixed path length; if None, use mean of instrument lengths

    Returns:
        equity_paths: shape (n_paths, T), cumulative portfolio equity in R
    """
    rng = np.random.default_rng(seed)
    instruments = list(R_by_instrument.keys())
    N = len(instruments)

    # Determine path length
    if path_length is None:
        lengths = [len(R) for R in R_by_instrument.values()]
        path_length = int(np.mean(lengths))
        print(f"\n[run_portfolio_mc] Using mean path length: {path_length} trades")
    else:
        print(f"\n[run_portfolio_mc] Using specified path length: {path_length} trades")

    print(f"  Number of paths: {n_paths}")
    print(f"  Block size: {block_size}")
    print(f"  Portfolio instruments: {N}")
    print(f"  Equal weight per instrument: 1/{N} = {1/N:.4f}")

    # Initialize equity paths
    equity_paths = np.zeros((n_paths, path_length), dtype=np.float64)

    # Run Monte Carlo
    for path_idx in range(n_paths):
        if path_idx % 1000 == 0 and path_idx > 0:
            print(f"  Progress: {path_idx}/{n_paths} paths completed")

        # Resample returns for each instrument
        portfolio_returns = np.zeros(path_length)

        for instrument in instruments:
            returns = R_by_instrument[instrument]

            # Resample with block bootstrap
            if block_size == 1:
                sampled_returns = rng.choice(returns, size=path_length, replace=True)
            else:
                sampled_returns = _block_bootstrap(returns, path_length, block_size, rng)

            # Add to portfolio with equal weight
            portfolio_returns += sampled_returns / N

        # Compute cumulative equity for this path
        equity_paths[path_idx, :] = np.cumsum(portfolio_returns)

    print(f"  Completed {n_paths} paths")

    return equity_paths


def _block_bootstrap(
    returns: np.ndarray,
    target_length: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Block bootstrap: sample blocks of consecutive trades.

    Args:
        returns: Original returns array
        target_length: Desired length
        block_size: Block length
        rng: Random generator

    Returns:
        Resampled returns
    """
    n = len(returns)
    if block_size >= n:
        return rng.choice(returns, size=target_length, replace=True)

    blocks = []
    total_len = 0

    while total_len < target_length:
        start_idx = rng.integers(0, n - block_size + 1)
        block = returns[start_idx : start_idx + block_size]
        blocks.append(block)
        total_len += len(block)

    resampled = np.concatenate(blocks)[:target_length]
    return resampled


# ============================================================================
#                         STATISTICS
# ============================================================================

def summarise_mc_paths(equity_paths: np.ndarray) -> Dict[str, float]:
    """
    Calculate summary statistics from MC paths.

    Args:
        equity_paths: shape (n_paths, T)

    Returns:
        Dictionary with summary stats
    """
    n_paths, T = equity_paths.shape

    # Final R distribution
    final_R = equity_paths[:, -1]
    median_final = np.median(final_R)
    p05_final = np.percentile(final_R, 5)
    p95_final = np.percentile(final_R, 95)

    # Max drawdown per path
    max_DD_per_path = np.zeros(n_paths)
    for i in range(n_paths):
        running_max = np.maximum.accumulate(equity_paths[i, :])
        drawdown = equity_paths[i, :] - running_max
        max_DD_per_path[i] = drawdown.min()

    median_maxDD = np.median(max_DD_per_path)
    p95_maxDD = np.percentile(max_DD_per_path, 95)  # Worst 5% of drawdowns

    # Losing streaks per path
    losing_streaks = np.array([_longest_losing_streak_from_equity(equity_paths[i, :]) for i in range(n_paths)])
    median_streak = np.median(losing_streaks)
    p95_streak = np.percentile(losing_streaks, 95)

    return {
        "n_paths": n_paths,
        "T": T,
        "median_final_R": median_final,
        "p05_final_R": p05_final,
        "p95_final_R": p95_final,
        "median_maxDD_R": median_maxDD,
        "p95_maxDD_R": p95_maxDD,
        "median_losing_streak": median_streak,
        "p95_losing_streak": p95_streak,
    }


def _longest_losing_streak_from_equity(equity_curve: np.ndarray) -> int:
    """
    Calculate longest losing streak from equity curve.

    A losing trade is one where equity decreases.

    Args:
        equity_curve: Cumulative equity array

    Returns:
        Length of longest losing streak
    """
    if len(equity_curve) < 2:
        return 0

    # Calculate returns
    returns = np.diff(equity_curve)

    # Find losing trades
    losses = returns < 0

    if not np.any(losses):
        return 0

    # Find longest streak
    max_streak = 0
    current_streak = 0

    for is_loss in losses:
        if is_loss:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return int(max_streak)


# ============================================================================
#                         OUTPUT
# ============================================================================

def save_summary_csv(
    stats: Dict[str, float],
    output_path: str,
    portfolio_id: str,
    n_instruments: int,
) -> None:
    """Save MC summary to CSV."""
    df = pd.DataFrame([{
        "portfolio_id": portfolio_id,
        "n_instruments": n_instruments,
        "n_paths": int(stats["n_paths"]),
        "T": int(stats["T"]),
        "median_final_R": stats["median_final_R"],
        "p05_final_R": stats["p05_final_R"],
        "p95_final_R": stats["p95_final_R"],
        "median_maxDD_R": stats["median_maxDD_R"],
        "p95_maxDD_R": stats["p95_maxDD_R"],
        "median_losing_streak": stats["median_losing_streak"],
        "p95_losing_streak": stats["p95_losing_streak"],
    }])

    df.to_csv(output_path, index=False)
    print(f"\n[save] Saved MC summary to: {output_path}")


def save_sample_paths(
    equity_paths: np.ndarray,
    output_dir: str,
    n_samples: int = 100,
) -> None:
    """
    Save sample equity paths for visualization.

    Args:
        equity_paths: shape (n_paths, T)
        output_dir: Directory to save CSV files
        n_samples: Number of paths to save
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_paths = min(n_samples, equity_paths.shape[0])
    indices = np.random.choice(equity_paths.shape[0], n_paths, replace=False)

    for i, path_idx in enumerate(indices):
        df = pd.DataFrame({
            "step": np.arange(equity_paths.shape[1]),
            "equity_R": equity_paths[path_idx, :],
        })
        filepath = output_path / f"path_{i:03d}.csv"
        df.to_csv(filepath, index=False)

    print(f"\n[save] Saved {n_paths} sample paths to: {output_dir}/")


# ============================================================================
#                         MAIN CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Portfolio-level Monte Carlo simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="walkforward_*_OOS_trades.csv",
        help="Glob pattern for OOS trade CSV files",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="TrendEMAPullback",
        help="Strategy name to filter trades",
    )

    parser.add_argument(
        "--basket-yaml",
        type=str,
        default="live_basket_phase2.yml",
        help="Path to basket configuration YAML",
    )

    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional instruments from basket",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Block bootstrap size (1 = iid sampling)",
    )

    parser.add_argument(
        "--paths",
        type=int,
        default=10000,
        help="Number of Monte Carlo paths",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--path-length",
        type=int,
        default=None,
        help="Fixed path length (if None, use mean of instrument lengths)",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="portfolio_mc_summary.csv",
        help="Output CSV file for MC summary",
    )

    parser.add_argument(
        "--output-paths",
        type=str,
        default=None,
        help="Optional directory to save sample equity paths",
    )

    parser.add_argument(
        "--portfolio-id",
        type=str,
        default="phase2_core",
        help="Portfolio identifier for output",
    )

    return parser.parse_args()


def main():
    """Main execution workflow."""
    args = parse_args()

    print("=" * 80)
    print("Portfolio Monte Carlo Simulation")
    print("=" * 80)
    print(f"Strategy: {args.strategy}")
    print(f"Basket config: {args.basket_yaml}")
    print(f"Include optional: {args.include_optional}")
    print(f"Portfolio ID: {args.portfolio_id}")
    print("=" * 80)

    # Step 1: Load basket configuration
    print("\n[1/4] Loading basket configuration...")
    instruments = load_basket_config(args.basket_yaml, args.include_optional)
    print(f"  Loaded {len(instruments)} instruments: {', '.join(instruments)}")

    # Step 2: Load trades by instrument
    print("\n[2/4] Loading OOS trades by instrument...")
    R_by_instrument = load_trades_by_instrument(
        pattern=args.pattern,
        strategy=args.strategy,
        instruments=instruments,
    )

    # Step 3: Run Monte Carlo
    print("\n[3/4] Running portfolio Monte Carlo simulation...")
    equity_paths = run_portfolio_mc(
        R_by_instrument=R_by_instrument,
        n_paths=args.paths,
        block_size=args.block_size,
        seed=args.seed,
        path_length=args.path_length,
    )

    # Step 4: Calculate and save statistics
    print("\n[4/4] Calculating statistics...")
    stats = summarise_mc_paths(equity_paths)

    # Print summary
    print("\n" + "=" * 80)
    print("PORTFOLIO MONTE CARLO RESULTS")
    print("=" * 80)
    print(f"Portfolio: {args.portfolio_id}")
    print(f"Instruments: {len(R_by_instrument)}")
    print(f"MC paths: {stats['n_paths']}")
    print(f"Path length: {stats['T']} trades")
    print(f"\nFinal Equity Distribution:")
    print(f"  Median: {stats['median_final_R']:.2f}R")
    print(f"  P05: {stats['p05_final_R']:.2f}R")
    print(f"  P95: {stats['p95_final_R']:.2f}R")
    print(f"\nMax Drawdown Distribution:")
    print(f"  Median: {stats['median_maxDD_R']:.2f}R")
    print(f"  P95 (worst 5%): {stats['p95_maxDD_R']:.2f}R")
    print(f"\nLosing Streak Distribution:")
    print(f"  Median: {stats['median_losing_streak']:.0f} trades")
    print(f"  P95: {stats['p95_losing_streak']:.0f} trades")
    print("=" * 80)

    # Save outputs
    save_summary_csv(
        stats=stats,
        output_path=args.output_csv,
        portfolio_id=args.portfolio_id,
        n_instruments=len(R_by_instrument),
    )

    if args.output_paths:
        save_sample_paths(
            equity_paths=equity_paths,
            output_dir=args.output_paths,
            n_samples=100,
        )

    print("\n" + "=" * 80)
    print("Monte Carlo simulation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
