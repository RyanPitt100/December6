# walkforward_multi_instrument.py
"""
Multi-instrument walk-forward evaluation for Trend_EMA_Pullback strategy.

Process:
1. Run walk-forward on all instruments
2. Calculate aggregate OOS metrics
3. Filter to shortlist based on performance criteria
4. Save shortlist for Monte Carlo analysis
"""

from typing import List, Dict, Optional
import pandas as pd
from dateutil.relativedelta import relativedelta

from multi_tf_builder import build_multi_tf_frame
from strategies import TrendEMAPullback
from run_experiments import backtest_signals
from backtest_framework import _compute_metrics


# ========= CONFIG =========

INSTRUMENTS = [
    "AUDJPY",
    "AUDUSD",
    "AUS200",
    "EU50",
    "EURGBP",
    "EURUSD",
    "GBPJPY",
    "GBPUSD",
    "GER40",
    "HK50",
    "JP225",
    "NZDUSD",
    "UK100",
    "UKOIL",
    "US30",
    "US100",
    "US500",
    "US2000",
    "USDCAD",
    "USDCHF",
    "USDJPY",
    "USOIL",
]

STRATEGY_CLASS = TrendEMAPullback

# Walk-forward parameters
TRAIN_YEARS = 2
TEST_MONTHS = 6
STEP_MONTHS = 6

# Shortlist criteria (all must pass)
MIN_OOS_TRADES = 50          # Minimum trades for statistical significance
MIN_AVG_R = 0.0              # Minimum average R (must be positive edge)
MAX_OOS_DD = -15.0           # Maximum drawdown in R (e.g., -15R)
MIN_SHARPE = 0.0             # Minimum Sharpe ratio
MIN_PROFIT_FACTOR = 1.0      # Minimum profit factor

OUTPUT_DIR = "./"


# ========= HELPERS =========

def make_walkforward_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_years: int,
    test_months: int,
    step_months: int,
) -> List[Dict]:
    """Build rolling train/test windows."""
    windows: List[Dict] = []
    train_end = start + relativedelta(years=train_years)

    while True:
        test_start = train_end
        test_end = test_start + relativedelta(months=test_months)

        if test_end > end:
            break

        train_start = train_end - relativedelta(years=train_years)

        windows.append(
            dict(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        train_end = train_end + relativedelta(months=step_months)

    return windows


def run_walkforward_for_instrument(instrument: str) -> Optional[Dict]:
    """
    Run walk-forward analysis for a single instrument.

    Returns:
        Dictionary with aggregated OOS metrics, or None if failed.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {instrument}")
    print(f"{'='*80}")

    try:
        # 1) Load data
        df_all = build_multi_tf_frame(instrument)
        if df_all is None or df_all.empty:
            print(f"  [SKIP] No data for {instrument}")
            return None

        df_all = df_all.sort_index()
        start, end = df_all.index[0], df_all.index[-1]
        print(f"  Data range: {start} -> {end}")

        # 2) Define windows
        windows = make_walkforward_windows(
            start=start,
            end=end,
            train_years=TRAIN_YEARS,
            test_months=TEST_MONTHS,
            step_months=STEP_MONTHS,
        )

        if not windows:
            print(f"  [SKIP] No valid walk-forward windows for {instrument}")
            return None

        print(f"  Walk-forward windows: {len(windows)}")

        # 3) Run WF
        window_stats = []
        all_test_trades = []

        for i, w in enumerate(windows, 1):
            train_start = w["train_start"]
            train_end = w["train_end"]
            test_start = w["test_start"]
            test_end = w["test_end"]

            df_train = df_all.loc[train_start:train_end]
            df_test = df_all.loc[test_start:test_end]

            if len(df_train) == 0 or len(df_test) == 0:
                continue

            strategy = STRATEGY_CLASS()

            # TRAIN
            sig_train = strategy.generate_signals(df_train, instrument=instrument)
            trades_train = backtest_signals(df_train, sig_train)
            m_train = _compute_metrics(trades_train, instrument=instrument)

            # TEST (OOS)
            sig_test = strategy.generate_signals(df_test, instrument=instrument)
            trades_test = backtest_signals(df_test, sig_test)
            m_test = _compute_metrics(trades_test, instrument=instrument)

            window_stats.append(
                {
                    "instrument": instrument,
                    "strategy": STRATEGY_CLASS.__name__,
                    "window": i,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "trades_train": m_train["trades"],
                    "trades_test": m_test["trades"],
                    "pf_train": m_train["pf"],
                    "pf_test": m_test["pf"],
                    "winrate_train": m_train["winrate"],
                    "winrate_test": m_test["winrate"],
                    "avgR_train": m_train["avgR"],
                    "avgR_test": m_test["avgR"],
                    "maxDD_train": m_train["maxDD"],
                    "maxDD_test": m_test["maxDD"],
                }
            )

            # Collect OOS trades
            if not trades_test.empty:
                df_trades_test = trades_test.copy()
                df_trades_test.insert(0, "strategy_name", STRATEGY_CLASS.__name__)
                df_trades_test.insert(0, "instrument", instrument)
                df_trades_test["window_train_start"] = train_start
                df_trades_test["window_train_end"] = train_end
                df_trades_test["window_test_start"] = test_start
                df_trades_test["window_test_end"] = test_end
                all_test_trades.append(df_trades_test)

        if not all_test_trades:
            print(f"  [SKIP] No OOS trades for {instrument}")
            return None

        # 4) Save per-instrument files
        df_win = pd.DataFrame(window_stats)
        win_path = f"{OUTPUT_DIR}walkforward_{instrument}_{STRATEGY_CLASS.__name__}_windows.csv"
        df_win.to_csv(win_path, index=False)

        df_oos = pd.concat(all_test_trades, ignore_index=True)
        oos_path = f"{OUTPUT_DIR}walkforward_{instrument}_{STRATEGY_CLASS.__name__}_OOS_trades.csv"
        df_oos.to_csv(oos_path, index=False)

        # 5) Compute aggregate OOS metrics
        m_oos = _compute_metrics(df_oos, instrument=instrument)

        # Calculate Sharpe ratio approximation
        if len(df_oos) > 0:
            mean_r = df_oos["R"].mean()
            std_r = df_oos["R"].std()
            sharpe = mean_r / std_r if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        print(f"\n  Aggregated OOS Results:")
        print(f"    Trades: {m_oos['trades']}")
        print(f"    Avg R: {m_oos['avgR']:.3f}")
        print(f"    Std R: {std_r:.3f}")
        print(f"    Sharpe: {sharpe:.3f}")
        print(f"    Win Rate: {m_oos['winrate']:.1f}%")
        print(f"    Profit Factor: {m_oos['pf']:.2f}")
        print(f"    Max DD: {m_oos['maxDD']:.2f}R")

        return {
            "instrument": instrument,
            "strategy": STRATEGY_CLASS.__name__,
            "oos_trades": m_oos["trades"],
            "oos_avgR": m_oos["avgR"],
            "oos_stdR": std_r,
            "oos_sharpe": sharpe,
            "oos_winrate": m_oos["winrate"],
            "oos_pf": m_oos["pf"],
            "oos_maxDD": m_oos["maxDD"],
            "windows_file": win_path,
            "trades_file": oos_path,
        }

    except FileNotFoundError as e:
        print(f"  [SKIP] Missing data for {instrument}: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] Failed processing {instrument}: {e}")
        import traceback
        traceback.print_exc()
        return None


def apply_shortlist_criteria(results: List[Dict]) -> pd.DataFrame:
    """
    Filter results based on performance criteria.

    Returns:
        DataFrame with shortlisted instruments, sorted by Sharpe ratio.
    """
    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"Applying Shortlist Criteria")
    print(f"{'='*80}")
    print(f"Total instruments processed: {len(df)}")
    print(f"\nCriteria:")
    print(f"  Min OOS Trades: {MIN_OOS_TRADES}")
    print(f"  Min Avg R: {MIN_AVG_R}")
    print(f"  Max DD (worst case): {MAX_OOS_DD}R")
    print(f"  Min Sharpe: {MIN_SHARPE}")
    print(f"  Min Profit Factor: {MIN_PROFIT_FACTOR}")

    # Apply filters
    mask = (
        (df["oos_trades"] >= MIN_OOS_TRADES) &
        (df["oos_avgR"] >= MIN_AVG_R) &
        (df["oos_maxDD"] >= MAX_OOS_DD) &
        (df["oos_sharpe"] >= MIN_SHARPE) &
        (df["oos_pf"] >= MIN_PROFIT_FACTOR)
    )

    df_shortlist = df[mask].copy()

    # Sort by Sharpe ratio (best first)
    df_shortlist = df_shortlist.sort_values("oos_sharpe", ascending=False)

    print(f"\nInstruments passing criteria: {len(df_shortlist)}/{len(df)}")

    if len(df_shortlist) > 0:
        print(f"\nShortlisted Instruments (sorted by Sharpe):")
        print(df_shortlist[["instrument", "oos_trades", "oos_avgR", "oos_sharpe",
                           "oos_winrate", "oos_pf", "oos_maxDD"]].to_string(index=False))
    else:
        print("\n[WARNING] No instruments passed the criteria!")
        print("\nAll instruments stats:")
        print(df[["instrument", "oos_trades", "oos_avgR", "oos_sharpe",
                 "oos_winrate", "oos_pf", "oos_maxDD"]].to_string(index=False))

    return df_shortlist


def main():
    print(f"{'='*80}")
    print(f"Multi-Instrument Walk-Forward Analysis")
    print(f"Strategy: {STRATEGY_CLASS.__name__}")
    print(f"Instruments: {len(INSTRUMENTS)}")
    print(f"{'='*80}")

    # 1) Run walk-forward for all instruments
    all_results = []
    for instrument in INSTRUMENTS:
        result = run_walkforward_for_instrument(instrument)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\n[ERROR] No instruments produced valid results!")
        return

    # 2) Save all results
    df_all = pd.DataFrame(all_results)
    all_results_path = f"{OUTPUT_DIR}walkforward_all_instruments_summary.csv"
    df_all.to_csv(all_results_path, index=False)
    print(f"\n{'='*80}")
    print(f"All results saved to: {all_results_path}")
    print(f"{'='*80}")

    # 3) Apply shortlist criteria
    df_shortlist = apply_shortlist_criteria(all_results)

    # 4) Save shortlist
    if not df_shortlist.empty:
        shortlist_path = f"{OUTPUT_DIR}walkforward_shortlist.csv"
        df_shortlist.to_csv(shortlist_path, index=False)
        print(f"\nShortlist saved to: {shortlist_path}")

        # 5) Create Monte Carlo input list
        mc_instruments = df_shortlist["instrument"].tolist()
        mc_list_path = f"{OUTPUT_DIR}montecarlo_instruments.txt"
        with open(mc_list_path, "w") as f:
            f.write(",".join(mc_instruments))
        print(f"Monte Carlo instrument list saved to: {mc_list_path}")
        print(f"\nTo run Monte Carlo on shortlisted instruments:")
        print(f'  python monte_carlo_trend.py --input "walkforward_*_OOS_trades.csv" --instruments "{",".join(mc_instruments)}"')
    else:
        print("\n[WARNING] No instruments in shortlist - cannot create Monte Carlo input")
        print("Consider relaxing the criteria in the CONFIG section")

    print(f"\n{'='*80}")
    print(f"Multi-Instrument Walk-Forward Complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
