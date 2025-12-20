# live_engine.py

from typing import Optional

from strategies import TrendEMAPullback, Signal
from multi_tf_builder import build_multi_tf_frame


def generate_latest_trend_signal(
    instrument: str,
    data_root: str = "./",
    min_bars: int = 200,
) -> Optional[Signal]:
    """
    Loads the multi-TF history, runs Trend_EMA_Pullback, and returns the *latest* signal
    (if any) for this instrument. This is what live_main.py will call each bar.

    Args:
        instrument: Symbol name (e.g., "EURUSD", "US30")
        data_root: Root directory containing data folders
        min_bars: Minimum number of bars required for valid signal generation

    Returns:
        Signal instance or None if no fresh signal.
    """
    df_m15 = build_multi_tf_frame(instrument=instrument, data_root=data_root)

    if df_m15 is None or len(df_m15) < min_bars:
        print(f"[live_engine] Not enough data for {instrument}, got {len(df_m15)} bars.")
        return None

    strat = TrendEMAPullback()

    # Important: TrendEMAPullback already checks the H1 regime field inside df_m15
    # and only emits signals in trending regimes.
    signals = strat.generate_signals(df_m15, instrument=instrument)

    if not signals:
        return None

    latest = signals[-1]

    # Sanity: ensure this signal is at/near the last bar.
    last_ts = df_m15.index[-1]
    if latest.time < last_ts:
        # This is an old signal; in live mode you'd probably ignore it.
        return None

    return latest
