from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from cost_model import get_cost_config
from multi_tf_builder import build_multi_tf_frame
from strategies import Signal


class Strategy(Protocol):
    name: str

    def generate_signals(self, df_m15: pd.DataFrame, instrument: str) -> List[Signal]:
        ...


@dataclass
class BacktestResult:
    instrument: str
    strategy_name: str
    split_time: pd.Timestamp
    trades_train: int
    trades_test: int
    winrate_train: float
    winrate_test: float
    avgR_train: float
    avgR_test: float
    pf_train: float
    pf_test: float
    maxDD_train: float
    maxDD_test: float


def simulate_trades(df: pd.DataFrame, signals: List[Signal], entry_on_next_bar: bool = True) -> pd.DataFrame:
    """
    Simple trade simulation with conservative assumptions:

    Entry/Exit Logic:
    - Independent trades (no overlap management)
    - Entry at next bar open (if entry_on_next_bar=True)
    - Exit at SL/TP or time stop from signal.meta["time_stop_bars"] (default 20)

    IMPORTANT - SL Priority Assumption (Conservative Bias):
    When both SL and TP are hit within the same bar, we assume SL is hit first.
    This creates a conservative bias in backtest results - actual performance may be better
    if TP was actually hit first. This is safer for prop firm risk management.

    To model this more realistically, you could:
    - Use a random 50/50 choice when both are hit
    - Model intra-bar price path (open→high/low→close sequence)
    - Use tick data instead of OHLC bars
    """
    trades: List[Dict[str, Any]] = []

    df = df.sort_index()
    default_time_stop_bars = 20

    for sig in signals:
        if sig.time not in df.index:
            continue

        loc = df.index.get_loc(sig.time)
        if isinstance(loc, slice):
            start_idx = loc.stop - 1
        else:
            start_idx = loc

        entry_idx = start_idx + 1 if entry_on_next_bar else start_idx
        if entry_idx >= len(df):
            continue

        entry_row = df.iloc[entry_idx]
        entry_price = float(entry_row["open"])
        sl = sig.stop_loss
        tp = sig.take_profit

        risk_per_unit = abs(entry_price - sl)
        if risk_per_unit <= 0:
            continue

        direction = sig.direction
        max_bars_hold = int(sig.meta.get("time_stop_bars", default_time_stop_bars))

        exit_price = None
        exit_time = None
        outcome = None  # TP, SL, TS

        for offset in range(0, max_bars_hold):
            idx = entry_idx + offset
            if idx >= len(df):
                break

            bar = df.iloc[idx]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            t = df.index[idx]

            if direction == "LONG":
                hit_sl = low <= sl
                hit_tp = high >= tp
            else:
                hit_sl = high >= sl
                hit_tp = low <= tp

            if hit_sl:
                exit_price = sl
                exit_time = t
                outcome = "SL"
                break
            if hit_tp:
                exit_price = tp
                exit_time = t
                outcome = "TP"
                break

            if offset == max_bars_hold - 1:
                exit_price = close
                exit_time = t
                outcome = "TS"

        if exit_price is None:
            continue

        if direction == "LONG":
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        R = pnl / risk_per_unit

        trades.append(
            {
                "entry_time": df.index[entry_idx],
                "exit_time": exit_time,
                "direction": direction,
                "entry": entry_price,
                "exit": exit_price,
                "sl": sl,
                "tp": tp,
                "outcome": outcome,
                "R": R,
            }
        )

    return pd.DataFrame(trades)


def _compute_metrics(trades: pd.DataFrame, instrument: str | None = None) -> dict:
    """
    Compute performance metrics from a trades DataFrame.

    Expects at least columns:
      - 'R'    : gross R per trade (before costs)
      - 'entry': entry price
      - 'sl'   : stop loss price

    If `instrument` is provided, applies a simple FTMO-style cost model:
      - spread cost converted to R using per-trade SL distance
      - fixed_R cost per trade (commission + residual slippage)

    Metrics are computed on R_net (after costs), but the original R
    column is left unchanged in the DataFrame.
    """
    if trades is None or trades.empty:
        return {
            "trades": 0,
            "winrate": 0.0,
            "avgR": 0.0,
            "pf": 0.0,
            "maxDD": 0.0,
        }

    df = trades.copy()

    # ---- Apply trading costs if we know the instrument ----
    if instrument is not None:
        cfg = get_cost_config(instrument)
        spread = cfg["spread"]
        fixed_R = cfg["fixed_R"]

        # Distance to SL in price units
        if "entry" not in df.columns or "sl" not in df.columns:
            # fallback: no SL info -> just subtract fixed_R
            cost_R = fixed_R
        else:
            dist = (df["entry"] - df["sl"]).abs()
            # protect against zero-distance stops
            dist = dist.replace(0.0, pd.NA)
            spread_R = (spread / dist).fillna(0.0)
            cost_R = spread_R + fixed_R

        df["R_net"] = df["R"] - cost_R
    else:
        df["R_net"] = df["R"]

    # ---- Compute stats on net R ----
    n_trades = len(df)
    wins = df[df["R_net"] > 0]
    losses = df[df["R_net"] < 0]

    total_R = df["R_net"].sum()
    winrate = len(wins) / n_trades * 100.0 if n_trades > 0 else 0.0

    gross_profit = wins["R_net"].sum()
    gross_loss = -losses["R_net"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    avgR = total_R / n_trades if n_trades > 0 else 0.0

    # Max drawdown in R using R_net
    equity = df["R_net"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    maxDD = dd.min() if not dd.empty else 0.0

    return {
        "trades": n_trades,
        "winrate": winrate,
        "avgR": avgR,
        "pf": pf,
        "maxDD": maxDD,
    }


# ---------- Utility helpers ----------

def build_df_for_instrument(
    instrument: str,
    m15_path_tpl: str = "./15m/{inst}.csv",
    h1_path_tpl: str = "./1h/{inst}_labelled.csv",
    h4_path_tpl: str = "./4h/{inst}_labelled.csv",
    d1_path_tpl: str = "./1d/{inst}_labelled.csv",
) -> pd.DataFrame:
    """
    Build the combined multi-TF dataframe for an instrument.
    """
    df = build_multi_tf_frame(
        m15_path=m15_path_tpl.format(inst=instrument),
        h1_path=h1_path_tpl.format(inst=instrument),
        h4_path=h4_path_tpl.format(inst=instrument),
        d1_path=d1_path_tpl.format(inst=instrument),
    )
    return df


def infer_split_time_from_df(df: pd.DataFrame, train_fraction: float = 0.6) -> pd.Timestamp:
    """
    Infer a sensible train/test split based on the index range and length.
    Uses train_fraction of the data for training, rest for test.
    """
    idx = df.index
    if len(idx) < 10:
        # trivial; just pick middle
        return idx[int(len(idx) / 2)]

    # make sure it's sorted and unique
    idx = idx.sort_values().unique()
    split_pos = int(len(idx) * train_fraction)
    if split_pos <= 0:
        split_pos = 1
    if split_pos >= len(idx):
        split_pos = len(idx) - 1

    split_ts = idx[split_pos]
    return split_ts


def get_data_range(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    idx = df.index
    return idx.min(), idx.max()


def backtest_on_df(
    df_m15_all: pd.DataFrame,
    instrument: str,
    strategy: Strategy,
    split_ts: pd.Timestamp,
) -> BacktestResult:
    """
    Backtest a strategy on a pre-built dataframe with a given split timestamp.
    Avoids rebuilding data repeatedly and avoids tz issues (split_ts taken from df index).
    """

    # ensure split_ts is aligned with df index tz
    idx_tz = df_m15_all.index.tz
    if idx_tz is not None and split_ts.tzinfo is None:
        split_ts = split_ts.tz_localize(idx_tz)
    elif idx_tz is None and split_ts.tzinfo is not None:
        split_ts = split_ts.tz_convert("UTC").tz_localize(None)

    signals = strategy.generate_signals(df_m15_all, instrument=instrument)
    trades_df = simulate_trades(df_m15_all, signals, entry_on_next_bar=True)

    trades_train = trades_df[trades_df["entry_time"] < split_ts]
    trades_test = trades_df[trades_df["entry_time"] >= split_ts]

    m_train = _compute_metrics(trades_train, instrument=instrument)
    m_test = _compute_metrics(trades_test, instrument=instrument)

    return BacktestResult(
        instrument=instrument,
        strategy_name=strategy.name,
        split_time=split_ts,
        trades_train=m_train["trades"],
        trades_test=m_test["trades"],
        winrate_train=m_train["winrate"],
        winrate_test=m_test["winrate"],
        avgR_train=m_train["avgR"],
        avgR_test=m_test["avgR"],
        pf_train=m_train["pf"],
        pf_test=m_test["pf"],
        maxDD_train=m_train["maxDD"],
        maxDD_test=m_test["maxDD"],
    )
