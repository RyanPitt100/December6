"""
Quick analysis tool for Monte Carlo results to determine safe risk-per-trade %.

This script:
1. Loads MC summary statistics
2. Calculates safe risk-per-trade for FTMO limits (5% daily, 10% max DD)
3. Projects expected equity curves at different risk levels
4. Provides actionable recommendations
"""

import argparse
import pandas as pd
import numpy as np


def analyze_mc_results(
    summary_file: str = "mc_trend_summary.csv",
    ftmo_daily_limit_pct: float = 5.0,
    ftmo_max_dd_pct: float = 10.0,
    safety_buffer: float = 0.8,
):
    """
    Analyze Monte Carlo results and provide risk-per-trade recommendations.

    Args:
        summary_file: Path to MC summary CSV
        ftmo_daily_limit_pct: FTMO daily loss limit (%)
        ftmo_max_dd_pct: FTMO max drawdown limit (%)
        safety_buffer: Safety factor (0.8 = use only 80% of limit)
    """
    print("=" * 80)
    print("Monte Carlo Results Analysis - FTMO Risk Calibration")
    print("=" * 80)

    # Load results
    df = pd.read_csv(summary_file)
    print(f"\nLoaded {len(df)} instrument(s) from {summary_file}\n")

    # FTMO limits with safety buffer
    safe_daily_limit = ftmo_daily_limit_pct * safety_buffer
    safe_max_dd = ftmo_max_dd_pct * safety_buffer

    print(f"FTMO Limits (with {safety_buffer:.0%} safety buffer):")
    print(f"  Daily Loss Limit: {safe_daily_limit:.1f}%")
    print(f"  Max Drawdown: {safe_max_dd:.1f}%")
    print()

    # Analyze each instrument
    for idx, row in df.iterrows():
        instrument = row["instrument"]
        strategy = row["strategy"]
        n_trades = int(row["n_trades_oos"])
        mean_R = row["mean_R_oos"]
        std_R = row["std_R_oos"]
        sharpe = mean_R / std_R if std_R > 0 else 0

        median_final_R = row["median_final_R"]
        p05_final_R = row["p05_final_R"]
        p95_final_R = row["p95_final_R"]

        median_maxDD_R = row["median_maxDD_R"]
        p95_maxDD_R = row["p95_maxDD_R"]

        median_streak = int(row["median_losing_streak"])
        p95_streak = int(row["p95_losing_streak"])

        print("=" * 80)
        print(f"Instrument: {instrument} ({strategy})")
        print("=" * 80)

        print(f"\nOOS Performance:")
        print(f"  Trades: {n_trades}")
        print(f"  Mean R: {mean_R:.3f}")
        print(f"  Std R: {std_R:.3f}")
        print(f"  Sharpe: {sharpe:.3f}")

        print(f"\nMonte Carlo Results (10,000 paths):")
        print(f"  Median Final: {median_final_R:.2f}R")
        print(f"  P05-P95: [{p05_final_R:.2f}R, {p95_final_R:.2f}R]")
        print(f"  Median Max DD: {median_maxDD_R:.2f}R")
        print(f"  P95 Max DD: {p95_maxDD_R:.2f}R (worst 5% scenarios)")
        print(f"  Median Losing Streak: {median_streak} trades")
        print(f"  P95 Losing Streak: {p95_streak} trades")

        # Calculate safe risk per trade based on max DD
        # Formula: risk_pct = (FTMO_limit * safety) / abs(p95_maxDD_R)
        max_safe_risk_from_dd = safe_max_dd / abs(p95_maxDD_R)

        # Also check daily limit - assume worst case: all losses in one day
        # Conservative: if p95_streak losses all hit in one day
        worst_day_R = -p95_streak  # Assume -1R per loss in worst case
        max_safe_risk_from_daily = safe_daily_limit / abs(worst_day_R)

        # Take the more conservative limit
        recommended_risk = min(max_safe_risk_from_dd, max_safe_risk_from_daily)

        print(f"\n{'─' * 80}")
        print("Risk-Per-Trade Recommendations:")
        print('─' * 80)

        print(f"\nBased on Max DD ({abs(p95_maxDD_R):.2f}R):")
        print(f"  Max safe risk: {max_safe_risk_from_dd:.3f}%")

        print(f"\nBased on Daily Limit (worst {p95_streak} losses in one day):")
        print(f"  Max safe risk: {max_safe_risk_from_daily:.3f}%")

        print(f"\n{'─' * 80}")
        print(f"RECOMMENDED RISK PER TRADE: {recommended_risk:.3f}%")
        print('─' * 80)

        # Project outcomes at recommended risk
        expected_final_pct = median_final_R * recommended_risk
        p05_final_pct = p05_final_R * recommended_risk
        p95_final_pct = p95_final_R * recommended_risk
        expected_dd_pct = median_maxDD_R * recommended_risk
        worst_dd_pct = p95_maxDD_R * recommended_risk

        print(f"\nProjected Outcomes at {recommended_risk:.3f}% risk per trade:")
        print(f"  Expected Final Equity: {expected_final_pct:+.2f}%")
        print(f"  P05-P95 Range: [{p05_final_pct:+.2f}%, {p95_final_pct:+.2f}%]")
        print(f"  Expected Max DD: {expected_dd_pct:.2f}%")
        print(f"  Worst Case DD (P95): {worst_dd_pct:.2f}%")

        # Risk levels table
        print(f"\n{'─' * 80}")
        print("Risk Level Comparison:")
        print('─' * 80)
        print(f"{'Risk/Trade':<12} {'Final Equity':<15} {'Median DD':<12} {'P95 DD':<12} {'Safe?':<8}")
        print('─' * 80)

        for risk_pct in [0.25, 0.50, 0.75, 1.00]:
            final = median_final_R * risk_pct
            med_dd = median_maxDD_R * risk_pct
            p95_dd = p95_maxDD_R * risk_pct

            # Check if safe
            is_safe_max_dd = abs(p95_dd) <= safe_max_dd
            is_safe_daily = risk_pct * p95_streak <= safe_daily_limit
            is_safe = "[OK] Yes" if (is_safe_max_dd and is_safe_daily) else "[X] No"

            print(f"{risk_pct:.2f}%       {final:+.2f}%          {med_dd:.2f}%       {p95_dd:.2f}%       {is_safe}")

        print()

    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)

    # Summary recommendations
    print("\nKey Takeaways:")
    print("1. Start with the recommended risk-per-trade from the analysis above")
    print("2. Monitor actual drawdown during live trading and reduce risk if approaching limits")
    print("3. P95 Max DD represents worst 5% of scenarios - expect this once every ~20 cycles")
    print("4. Losing streaks of up to P95 length are normal - do not panic")
    print("5. Consider starting at 50-75% of recommended risk for first month")

    print("\nNext Steps:")
    print("1. Run walk-forward for more instruments to diversify")
    print("2. Test with different block sizes (--block-size 3-5) to assess regime clustering")
    print("3. Generate equity path samples (--output-paths) for visualization")
    print("4. Monitor live trading against MC projections")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Monte Carlo results for FTMO risk calibration"
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="mc_trend_summary.csv",
        help="Path to MC summary CSV",
    )
    parser.add_argument(
        "--ftmo-daily-limit",
        type=float,
        default=5.0,
        help="FTMO daily loss limit (%)",
    )
    parser.add_argument(
        "--ftmo-max-dd",
        type=float,
        default=10.0,
        help="FTMO max drawdown limit (%)",
    )
    parser.add_argument(
        "--safety-buffer",
        type=float,
        default=0.8,
        help="Safety factor (0.8 = use 80% of limit)",
    )

    args = parser.parse_args()

    analyze_mc_results(
        summary_file=args.summary,
        ftmo_daily_limit_pct=args.ftmo_daily_limit,
        ftmo_max_dd_pct=args.ftmo_max_dd,
        safety_buffer=args.safety_buffer,
    )


if __name__ == "__main__":
    main()


# Example:
# python analyze_mc_results.py
# python analyze_mc_results.py --summary mc_trend_summary.csv --safety-buffer 0.7
