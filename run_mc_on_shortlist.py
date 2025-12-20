# run_mc_on_shortlist.py
"""
Run Monte Carlo analysis on shortlisted instruments from walk-forward.

This script:
1. Reads the shortlist from walkforward_shortlist.csv
2. Runs Monte Carlo simulation for each instrument
3. Generates FTMO risk recommendations
4. Creates a portfolio-level summary
"""

import pandas as pd
import subprocess
import sys
from pathlib import Path


def main():
    print("="*80)
    print("Monte Carlo Analysis on Shortlisted Instruments")
    print("="*80)

    # 1) Load shortlist
    shortlist_path = "walkforward_shortlist.csv"
    if not Path(shortlist_path).exists():
        print(f"\n[ERROR] Shortlist file not found: {shortlist_path}")
        print("Please run walkforward_multi_instrument.py first")
        return

    df_shortlist = pd.read_csv(shortlist_path)

    if df_shortlist.empty:
        print("\n[ERROR] Shortlist is empty!")
        return

    instruments = df_shortlist["instrument"].tolist()
    print(f"\nShortlisted instruments: {len(instruments)}")
    for inst in instruments:
        print(f"  - {inst}")

    # 2) Run Monte Carlo for all instruments together
    print(f"\n{'='*80}")
    print("Running Monte Carlo simulation...")
    print(f"{'='*80}")

    instruments_str = ",".join(instruments)
    cmd = [
        sys.executable,
        "monte_carlo_trend.py",
        "--input", "walkforward_*_OOS_trades.csv",
        "--instruments", instruments_str,
        "--strategy", "TrendEMAPullback",
        "--n-paths", "10000",
        "--output-summary", "mc_shortlist_summary.csv",
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("\n[ERROR] Monte Carlo simulation failed!")
        return

    # 3) Run FTMO risk analysis
    print(f"\n{'='*80}")
    print("Analyzing FTMO Risk Recommendations...")
    print(f"{'='*80}")

    # Check if mc_shortlist_summary.csv exists
    if not Path("mc_shortlist_summary.csv").exists():
        print("\n[ERROR] Monte Carlo summary not found!")
        return

    # Run analysis
    cmd_analyze = [
        sys.executable,
        "analyze_mc_results.py",
        "--summary", "mc_shortlist_summary.csv",
    ]

    subprocess.run(cmd_analyze, capture_output=False)

    # 4) Create portfolio summary
    print(f"\n{'='*80}")
    print("Portfolio Summary")
    print(f"{'='*80}")

    df_mc = pd.read_csv("mc_shortlist_summary.csv")

    # Merge with shortlist to get full info
    df_portfolio = df_shortlist.merge(
        df_mc[["instrument", "n_trades_oos", "mean_R_oos", "std_R_oos",
               "median_final_R", "p95_maxDD_R", "p95_losing_streak"]],
        on="instrument",
        how="inner"
    )

    print("\nPortfolio Instruments:")
    print(df_portfolio[[
        "instrument", "oos_trades", "oos_avgR", "oos_sharpe",
        "median_final_R", "p95_maxDD_R"
    ]].to_string(index=False))

    # Save portfolio summary
    portfolio_path = "portfolio_summary.csv"
    df_portfolio.to_csv(portfolio_path, index=False)
    print(f"\nPortfolio summary saved to: {portfolio_path}")

    # Calculate portfolio-level stats (simple average - assumes equal weighting)
    print(f"\n{'='*80}")
    print("Portfolio-Level Statistics (Equal Weight)")
    print(f"{'='*80}")
    print(f"Instruments: {len(df_portfolio)}")
    print(f"Total OOS Trades: {df_portfolio['oos_trades'].sum()}")
    print(f"Avg R (portfolio): {df_portfolio['oos_avgR'].mean():.3f}")
    print(f"Avg Sharpe (portfolio): {df_portfolio['oos_sharpe'].mean():.3f}")
    print(f"Median Final R (avg): {df_portfolio['median_final_R'].mean():.2f}R")
    print(f"Worst P95 DD: {df_portfolio['p95_maxDD_R'].min():.2f}R")
    print(f"Avg P95 DD: {df_portfolio['p95_maxDD_R'].mean():.2f}R")

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print(f"  - mc_shortlist_summary.csv (Monte Carlo results)")
    print(f"  - portfolio_summary.csv (Combined WF + MC stats)")


if __name__ == "__main__":
    main()
