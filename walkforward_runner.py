# walkforward_runner.py
"""
Walk-forward evaluation for a single instrument/strategy, with costs.

Uses:
- multi_tf_builder.build_multi_tf_frame(instrument)
- strategies.TrendEMAPullback (you can swap for others)
- backtest_framework.backtest_signals / _compute_metrics
"""

from typing import List, Dict

import pandas as pd
from dateutil.relativedelta import relativedelta

from multi_tf_builder import build_multi_tf_frame
from strategies import TrendEMAPullback  # swap / add others if you want
from run_experiments import backtest_signals
from backtest_framework import _compute_metrics


# ========= CONFIG =========

INSTRUMENT = "USDJPY"
STRATEGY_CLASS = TrendEMAPullback

TRAIN_YEARS = 2      # length of each training window
TEST_MONTHS = 6      # length of each test window
STEP_MONTHS = 6      # step between successive windows

OUTPUT_PREFIX = f"walkforward_{INSTRUMENT}_{STRATEGY_CLASS.__name__}"


# ========= HELPERS =========

def make_walkforward_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_years: int,
    test_months: int,
    step_months: int,
) -> List[Dict]:
    """
    Build rolling train/test windows like:

    [start        train_end] [test_start    test_end]
    [start+step   train_end+step] [test_start+step  test_end+step] ...
    """
    windows: List[Dict] = []

    # first train end = start + TRAIN_YEARS
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

        # slide forward
        train_end = train_end + relativedelta(months=step_months)

    return windows


# ========= MAIN WF LOGIC =========

def main() -> None:
    print(f"{INSTRUMENT} walk-forward for {STRATEGY_CLASS.__name__}")

    # 1) Build multi-TF frame once
    df_all = build_multi_tf_frame(INSTRUMENT)
    df_all = df_all.sort_index()
    start, end = df_all.index[0], df_all.index[-1]
    print(f"Data range: {start} -> {end}")

    # 2) Define windows
    windows = make_walkforward_windows(
        start=start,
        end=end,
        train_years=TRAIN_YEARS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
    )

    if not windows:
        print("No valid walk-forward windows (data too short?).")
        return

    for i, w in enumerate(windows, 1):
        print(
            f"Window {i}: "
            f"TRAIN [{w['train_start']} -> {w['train_end']}], "
            f"TEST [{w['test_start']} -> {w['test_end']}]"
        )

    # 3) Run WF
    window_stats = []
    all_test_trades = []

    for w in windows:
        train_start = w["train_start"]
        train_end = w["train_end"]
        test_start = w["test_start"]
        test_end = w["test_end"]

        df_train = df_all.loc[train_start:train_end]
        df_test = df_all.loc[test_start:test_end]

        if len(df_train) == 0 or len(df_test) == 0:
            continue

        # Strategy instance (no constructor args)
        strategy = STRATEGY_CLASS()

        # --- TRAIN ---
        sig_train = strategy.generate_signals(df_train, instrument=INSTRUMENT)
        trades_train = backtest_signals(df_train, sig_train)
        m_train = _compute_metrics(trades_train, instrument=INSTRUMENT)

        # --- TEST (OOS) ---
        sig_test = strategy.generate_signals(df_test, instrument=INSTRUMENT)
        trades_test = backtest_signals(df_test, sig_test)
        m_test = _compute_metrics(trades_test, instrument=INSTRUMENT)

        # record per-window stats
        window_stats.append(
            {
                "instrument": INSTRUMENT,
                "strategy": STRATEGY_CLASS.__name__,
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

        # keep all OOS trades
        if not trades_test.empty:
            df_trades_test = trades_test.copy()
            # Add metadata for Monte Carlo analysis
            df_trades_test.insert(0, "strategy_name", STRATEGY_CLASS.__name__)
            df_trades_test.insert(0, "instrument", INSTRUMENT)
            df_trades_test["window_train_start"] = train_start
            df_trades_test["window_train_end"] = train_end
            df_trades_test["window_test_start"] = test_start
            df_trades_test["window_test_end"] = test_end
            all_test_trades.append(df_trades_test)

    if not window_stats:
        print("No trades produced in any WF window.")
        return

    # 4) Save per-window stats
    df_win = pd.DataFrame(window_stats)
    win_path = f"{OUTPUT_PREFIX}_windows.csv"
    df_win.to_csv(win_path, index=False)
    print(f"\nPer-window stats saved to {win_path}")
    print(df_win)

    # 5) Aggregate OOS across all windows
    if all_test_trades:
        df_oos = pd.concat(all_test_trades, ignore_index=True)
        oos_path = f"{OUTPUT_PREFIX}_OOS_trades.csv"
        df_oos.to_csv(oos_path, index=False)
        print(f"\nAll OOS trades saved to {oos_path}")

        # compute overall OOS metrics with costs
        # trades_test is already a DataFrame from backtest_signals
        m_oos = _compute_metrics(df_oos, instrument=INSTRUMENT)
        print("\nAggregated OOS metrics across all windows (with costs):")
        for k, v in m_oos.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo OOS trades collected; nothing to aggregate.")


if __name__ == "__main__":
    main()
