# portfolio_wf_analysis.py
"""
Portfolio-level walk-forward OOS analysis.

Builds a portfolio equity curve from per-instrument OOS trades,
assuming equal-weight risk allocation across instruments.

Usage:
    python portfolio_wf_analysis.py --strategy TrendEMAPullback --basket-yaml live_basket_phase2.yml
"""

import argparse
import glob
import json
import warnings
from datetime import datetime
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

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML structure is invalid
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
        # Simple list format (phase1)
        instruments = config["instruments"]
    elif isinstance(config["instruments"], dict):
        # Structured format (phase2)
        if "core" in config["instruments"]:
            instruments.extend(config["instruments"]["core"])
        if include_optional and "optional" in config["instruments"]:
            instruments.extend(config["instruments"]["optional"])
    else:
        raise ValueError(f"Invalid YAML: 'instruments' must be list or dict with 'core' key")

    if not instruments:
        raise ValueError(f"No instruments found in {yaml_path}")

    return instruments


def load_trades(
    pattern: str,
    strategy: str,
    instruments: List[str],
) -> pd.DataFrame:
    """
    Load OOS trades from CSV files matching the pattern.

    Args:
        pattern: Glob pattern for CSV files
        strategy: Strategy name to filter
        instruments: List of instruments to include

    Returns:
        DataFrame with filtered trades

    Raises:
        FileNotFoundError: If no files match pattern
        ValueError: If no trades remain after filtering
    """
    # Expand glob pattern
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No files found matching pattern: {pattern}\n"
            f"Run walkforward_multi_instrument.py first to generate OOS trade CSVs."
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

    # Validate required columns
    required_cols = ["instrument", "R", "exit_time"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Filter by strategy if column exists
    if "strategy_name" in df.columns:
        df = df[df["strategy_name"] == strategy].copy()
        print(f"[load_trades] Filtered to strategy '{strategy}': {len(df)} trades")

    # Filter by instruments
    df = df[df["instrument"].isin(instruments)].copy()
    print(f"[load_trades] Filtered to {len(instruments)} instruments: {len(df)} trades")

    if len(df) == 0:
        raise ValueError(
            f"No trades remaining after filtering!\n"
            f"Strategy: {strategy}\n"
            f"Instruments: {instruments}"
        )

    # Parse exit_time as datetime
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")

    # Drop rows with invalid timestamps
    before_drop = len(df)
    df = df.dropna(subset=["exit_time"])
    if len(df) < before_drop:
        warnings.warn(f"Dropped {before_drop - len(df)} trades with invalid timestamps")

    # Sort by exit time
    df = df.sort_values("exit_time").reset_index(drop=True)

    return df


# ============================================================================
#                         PORTFOLIO EQUITY CURVE
# ============================================================================

def build_portfolio_equity(
    trades_df: pd.DataFrame,
    instruments: List[str],
) -> pd.DataFrame:
    """
    Build portfolio equity curve from individual trades.

    Assumes equal-weight risk allocation: each instrument gets 1/N of portfolio risk.
    Each trade's R is scaled by 1/N and added to cumulative portfolio equity.

    Args:
        trades_df: DataFrame with trades (must have 'exit_time', 'instrument', 'R')
        instruments: List of instruments in basket (for N)

    Returns:
        DataFrame with columns:
            - timestamp: Trade exit time
            - instrument: Instrument name
            - R: Trade R-multiple
            - portfolio_R_increment: R / N
            - portfolio_equity_R: Cumulative portfolio equity
    """
    if trades_df.empty:
        raise ValueError("Empty trades DataFrame")

    N = len(instruments)
    print(f"\n[build_portfolio_equity] Building portfolio with N={N} instruments")
    print(f"  Equal weight per instrument: 1/{N} = {1/N:.4f} of portfolio risk")

    # Create portfolio equity DataFrame
    portfolio = trades_df[["exit_time", "instrument", "R"]].copy()
    portfolio.rename(columns={"exit_time": "timestamp"}, inplace=True)

    # Calculate portfolio contribution for each trade
    portfolio["portfolio_R_increment"] = portfolio["R"] / N

    # Calculate cumulative portfolio equity
    portfolio["portfolio_equity_R"] = portfolio["portfolio_R_increment"].cumsum()

    print(f"  Total trades: {len(portfolio)}")
    print(f"  Final portfolio equity: {portfolio['portfolio_equity_R'].iloc[-1]:.2f}R")

    return portfolio


def calculate_portfolio_stats(
    portfolio_equity: pd.DataFrame,
) -> Dict:
    """
    Calculate summary statistics for portfolio equity curve.

    Args:
        portfolio_equity: DataFrame from build_portfolio_equity

    Returns:
        Dictionary with summary statistics
    """
    if portfolio_equity.empty:
        raise ValueError("Empty portfolio equity DataFrame")

    equity_R = portfolio_equity["portfolio_equity_R"].values
    timestamps = portfolio_equity["timestamp"]

    # Basic stats
    total_trades = len(portfolio_equity)
    final_R = equity_R[-1]
    mean_R_per_trade = portfolio_equity["portfolio_R_increment"].mean()
    std_R_per_trade = portfolio_equity["portfolio_R_increment"].std()

    # Max drawdown
    running_max = np.maximum.accumulate(equity_R)
    drawdown = equity_R - running_max
    max_dd_R = drawdown.min()
    max_dd_idx = drawdown.argmin()
    max_dd_date = timestamps.iloc[max_dd_idx]

    # Time-based stats (annualized)
    first_date = timestamps.iloc[0]
    last_date = timestamps.iloc[-1]
    days_elapsed = (last_date - first_date).total_seconds() / 86400
    years_elapsed = days_elapsed / 365.25

    if years_elapsed > 0:
        annualized_return_R = final_R / years_elapsed
        # For volatility, calculate daily returns and annualize
        daily_returns = portfolio_equity.set_index("timestamp")["portfolio_R_increment"].resample("D").sum()
        daily_returns = daily_returns[daily_returns != 0]  # Remove zero-volume days
        if len(daily_returns) > 1:
            daily_vol = daily_returns.std()
            annualized_vol_R = daily_vol * np.sqrt(252)
            sharpe_ratio = annualized_return_R / annualized_vol_R if annualized_vol_R > 0 else 0
        else:
            annualized_vol_R = 0
            sharpe_ratio = 0
    else:
        annualized_return_R = 0
        annualized_vol_R = 0
        sharpe_ratio = 0

    # Monthly stats
    portfolio_equity_monthly = portfolio_equity.copy()
    portfolio_equity_monthly["month"] = portfolio_equity_monthly["timestamp"].dt.to_period("M")
    monthly_returns = portfolio_equity_monthly.groupby("month")["portfolio_R_increment"].sum()
    positive_months = (monthly_returns > 0).sum()
    negative_months = (monthly_returns < 0).sum()
    total_months = len(monthly_returns)

    # Win rate
    win_rate = (portfolio_equity["portfolio_R_increment"] > 0).mean() * 100

    stats = {
        "total_trades": int(total_trades),
        "instruments_in_basket": len(portfolio_equity["instrument"].unique()),
        "first_trade_date": str(first_date),
        "last_trade_date": str(last_date),
        "days_elapsed": float(days_elapsed),
        "years_elapsed": float(years_elapsed),
        "final_equity_R": float(final_R),
        "mean_R_per_trade": float(mean_R_per_trade),
        "std_R_per_trade": float(std_R_per_trade),
        "win_rate_pct": float(win_rate),
        "max_drawdown_R": float(max_dd_R),
        "max_drawdown_date": str(max_dd_date),
        "annualized_return_R": float(annualized_return_R),
        "annualized_volatility_R": float(annualized_vol_R),
        "sharpe_ratio": float(sharpe_ratio),
        "total_months": int(total_months),
        "positive_months": int(positive_months),
        "negative_months": int(negative_months),
        "pct_positive_months": float(positive_months / total_months * 100 if total_months > 0 else 0),
    }

    return stats


# ============================================================================
#                         PLOTTING
# ============================================================================

def plot_portfolio_equity(
    portfolio_equity: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Generate equity curve and drawdown plots.

    Args:
        portfolio_equity: DataFrame from build_portfolio_equity
        output_path: Path to save PNG file
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        warnings.warn("matplotlib not available, skipping plot generation")
        return

    equity_R = portfolio_equity["portfolio_equity_R"].values
    timestamps = portfolio_equity["timestamp"]

    # Calculate drawdown
    running_max = np.maximum.accumulate(equity_R)
    drawdown = equity_R - running_max

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Equity curve
    ax1.plot(timestamps, equity_R, linewidth=1.5, color="blue")
    ax1.set_ylabel("Portfolio Equity (R)", fontsize=12)
    ax1.set_title("Portfolio Walk-Forward OOS Equity Curve", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Plot 2: Drawdown
    ax2.fill_between(timestamps, drawdown, 0, color="red", alpha=0.3)
    ax2.plot(timestamps, drawdown, linewidth=1, color="darkred")
    ax2.set_ylabel("Drawdown (R)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_title("Portfolio Drawdown", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot] Saved equity curve plot to: {output_path}")
    plt.close()


# ============================================================================
#                         MAIN CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Portfolio-level walk-forward OOS analysis",
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
        "--output-csv",
        type=str,
        default="portfolio_wf_equity.csv",
        help="Output CSV file for portfolio equity curve",
    )

    parser.add_argument(
        "--output-report",
        type=str,
        default="portfolio_wf_report.json",
        help="Output file for summary statistics (JSON or MD)",
    )

    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional path to save equity curve plot (PNG)",
    )

    return parser.parse_args()


def main():
    """Main execution workflow."""
    args = parse_args()

    print("=" * 80)
    print("Portfolio Walk-Forward OOS Analysis")
    print("=" * 80)
    print(f"Strategy: {args.strategy}")
    print(f"Basket config: {args.basket_yaml}")
    print(f"Include optional: {args.include_optional}")
    print("=" * 80)

    # Step 1: Load basket configuration
    print("\n[1/5] Loading basket configuration...")
    instruments = load_basket_config(args.basket_yaml, args.include_optional)
    print(f"  Loaded {len(instruments)} instruments: {', '.join(instruments)}")

    # Step 2: Load trades
    print("\n[2/5] Loading OOS trades...")
    trades = load_trades(
        pattern=args.pattern,
        strategy=args.strategy,
        instruments=instruments,
    )

    # Step 3: Build portfolio equity curve
    print("\n[3/5] Building portfolio equity curve...")
    portfolio_equity = build_portfolio_equity(trades, instruments)

    # Step 4: Calculate statistics
    print("\n[4/5] Calculating portfolio statistics...")
    stats = calculate_portfolio_stats(portfolio_equity)

    # Print summary
    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)
    print(f"Total trades: {stats['total_trades']}")
    print(f"Instruments: {stats['instruments_in_basket']}")
    print(f"Date range: {stats['first_trade_date']} to {stats['last_trade_date']}")
    print(f"Duration: {stats['years_elapsed']:.2f} years")
    print(f"\nReturns:")
    print(f"  Final equity: {stats['final_equity_R']:.2f}R")
    print(f"  Annualized return: {stats['annualized_return_R']:.2f}R/year")
    print(f"  Mean R per trade: {stats['mean_R_per_trade']:.4f}R")
    print(f"  Win rate: {stats['win_rate_pct']:.1f}%")
    print(f"\nRisk:")
    print(f"  Max drawdown: {stats['max_drawdown_R']:.2f}R (on {stats['max_drawdown_date']})")
    print(f"  Annualized volatility: {stats['annualized_volatility_R']:.2f}R")
    print(f"  Sharpe ratio: {stats['sharpe_ratio']:.3f}")
    print(f"\nMonthly performance:")
    print(f"  Total months: {stats['total_months']}")
    print(f"  Positive months: {stats['positive_months']} ({stats['pct_positive_months']:.1f}%)")
    print(f"  Negative months: {stats['negative_months']}")
    print("=" * 80)

    # Step 5: Save outputs
    print("\n[5/5] Saving outputs...")

    # Save equity curve CSV
    portfolio_equity.to_csv(args.output_csv, index=False)
    print(f"  Saved portfolio equity curve to: {args.output_csv}")

    # Save report
    if args.output_report.endswith(".json"):
        with open(args.output_report, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics report to: {args.output_report}")
    elif args.output_report.endswith(".md"):
        with open(args.output_report, "w") as f:
            f.write("# Portfolio Walk-Forward OOS Analysis Report\n\n")
            f.write(f"**Strategy**: {args.strategy}\n\n")
            f.write(f"**Basket**: {args.basket_yaml}\n\n")
            f.write(f"**Instruments**: {', '.join(instruments)}\n\n")
            f.write("## Summary Statistics\n\n")
            for key, value in stats.items():
                f.write(f"- **{key}**: {value}\n")
        print(f"  Saved statistics report to: {args.output_report}")
    else:
        with open(args.output_report, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics report to: {args.output_report}")

    # Generate plot if requested
    if args.plot:
        plot_portfolio_equity(portfolio_equity, args.plot)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
