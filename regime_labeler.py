
import argparse
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Regime = Literal["RANGING", "TRENDING", "TRANSITION", "NO_DATA"]


@dataclass
class RegimeParams:
    # core lookbacks
    er_lookback: int = 20
    adx_lookback: int = 14
    ma_len: int = 50
    ma_slope_lookback: int = 10
    atr_lookback: int = 20
    vol_pct_lookback: int = 252
    bw_lookback: int = 20
    bw_pct_lookback: int = 252

    # if True: use quantiles for ER/ADX instead of fixed magic numbers
    use_quantiles: bool = True
    er_low_quantile: float = 0.40
    er_high_quantile: float = 0.60
    adx_low_quantile: float = 0.40
    adx_high_quantile: float = 0.60

    # fallback / non-quantile thresholds (still used for slope/ATR/BW)
    er_range_max: float = 0.25
    er_trend_min: float = 0.35

    adx_range_max: float = 20.0
    adx_trend_min: float = 25.0

    # slope in percent
    ma_slope_neutral_max: float = 0.10

    atr_low_pct: float = 20.0
    atr_high_pct: float = 90.0

    bw_low_pct: float = 10.0
    bw_high_pct: float = 95.0

    # smoothing: number of consecutive candidate bars needed to switch
    min_persist_bars: int = 3


def compute_efficiency_ratio(close: pd.Series, lookback: int) -> pd.Series:
    price_change = close.diff(lookback).abs()
    volatility = close.diff().abs().rolling(lookback).sum()
    er = price_change / volatility
    return er


def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    tr = compute_true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / lookback, adjust=False).mean()
    return atr


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = compute_true_range(high, low, close)

    atr = tr.ewm(alpha=1.0 / lookback, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / lookback, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / lookback, adjust=False).mean() / atr

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / lookback, adjust=False).mean()
    return adx


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling percentile rank of current value within the window.
    Optimized using vectorized numpy operations instead of slow .apply().
    """
    from numpy.lib.stride_tricks import sliding_window_view

    arr = series.values.astype(float)
    n = len(arr)

    if n < window:
        return pd.Series(np.full(n, np.nan), index=series.index)

    # Create sliding windows - shape (n - window + 1, window)
    windows = sliding_window_view(arr, window)

    # Current values are the last element of each window
    current_vals = windows[:, -1].reshape(-1, 1)

    # Count percentage of window values <= current value
    percentiles = (windows <= current_vals).mean(axis=1) * 100.0

    # Prepend NaNs for initial window
    result = np.concatenate([np.full(window - 1, np.nan), percentiles])

    return pd.Series(result, index=series.index)


def smooth_regimes_state_machine(
    candidate: pd.Series,
    min_persist: int,
) -> pd.Series:
    current_state: Regime = "TRANSITION"
    counter = 0
    out: list[Regime] = []

    for r in candidate:
        if r not in ("RANGING", "TRENDING", "TRANSITION"):
            out.append("NO_DATA")
            continue

        if r == current_state:
            counter = 0
            out.append(current_state)
            continue

        if r == "TRANSITION":
            out.append("TRANSITION")
            continue

        counter += 1
        if counter >= min_persist:
            current_state = r
            counter = 0

        out.append(current_state)

    return pd.Series(out, index=candidate.index)


def label_regimes(
    df: pd.DataFrame,
    params: RegimeParams,
) -> pd.DataFrame:
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["er"] = compute_efficiency_ratio(close, params.er_lookback)
    df["adx"] = compute_adx(high, low, close, params.adx_lookback)

    df["ema50"] = close.ewm(span=params.ma_len, adjust=False).mean()
    ema_shifted = df["ema50"].shift(params.ma_slope_lookback)
    df["ema_slope_pct"] = (df["ema50"] - ema_shifted) / ema_shifted * 100.0

    df["atr"] = compute_atr(high, low, close, params.atr_lookback)
    df["atr_pct"] = rolling_percentile(df["atr"], params.vol_pct_lookback)

    mid = close.rolling(params.bw_lookback).mean()
    std = close.rolling(params.bw_lookback).std()
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_width"] = upper - lower
    df["bb_width_pct"] = rolling_percentile(df["bb_width"], params.bw_pct_lookback)

    valid = df[["er", "adx"]].dropna()

    if params.use_quantiles and not valid.empty:
        er_q_low = valid["er"].quantile(params.er_low_quantile)
        er_q_high = valid["er"].quantile(params.er_high_quantile)
        adx_q_low = valid["adx"].quantile(params.adx_low_quantile)
        adx_q_high = valid["adx"].quantile(params.adx_high_quantile)

        er_range_max = er_q_low
        er_trend_min = er_q_high
        adx_range_max = adx_q_low
        adx_trend_min = adx_q_high

        # Quantile thresholds computed (verbose logging disabled for live trading)
    else:
        er_range_max = params.er_range_max
        er_trend_min = params.er_trend_min
        adx_range_max = params.adx_range_max
        adx_trend_min = params.adx_trend_min

    # Vectorized regime classification (much faster than iterrows)
    er = df["er"].values
    adx = df["adx"].values
    slope = np.abs(df["ema_slope_pct"].values)
    atr_pct = df["atr_pct"].values
    bw_pct = df["bb_width_pct"].values

    # Check for missing data
    has_data = ~(
        pd.isna(df["er"]) | pd.isna(df["adx"]) | pd.isna(df["ema_slope_pct"]) |
        pd.isna(df["atr_pct"]) | pd.isna(df["bb_width_pct"])
    ).values

    # Compute scores
    if er_trend_min != er_range_max:
        er_score = np.clip((er_trend_min - er) / (er_trend_min - er_range_max), 0.0, 1.0)
    else:
        er_score = np.full(len(er), 0.5)

    if adx_trend_min != adx_range_max:
        adx_score = np.clip((adx_trend_min - adx) / (adx_trend_min - adx_range_max), 0.0, 1.0)
    else:
        adx_score = np.full(len(adx), 0.5)

    range_score_arr = (er_score + adx_score) / 2.0

    # Condition checks
    slope_ok = slope < params.ma_slope_neutral_max
    atr_ok = (params.atr_low_pct <= atr_pct) & (atr_pct <= params.atr_high_pct)
    bw_ok = (params.bw_low_pct <= bw_pct) & (bw_pct <= params.bw_high_pct)

    is_range = (er <= er_range_max) & (adx <= adx_range_max) & slope_ok & atr_ok & bw_ok
    is_trend = (er >= er_trend_min) & (adx >= adx_trend_min) & (~slope_ok | ~atr_ok | ~bw_ok)

    # Build regime array
    candidate = np.where(~has_data, "NO_DATA",
                np.where(is_range, "RANGING",
                np.where(is_trend, "TRENDING", "TRANSITION")))

    # Set range_score to NaN where no data
    range_score_arr = np.where(has_data, range_score_arr, np.nan)

    df["regime_raw"] = candidate
    df["range_score"] = range_score_arr

    df["regime"] = smooth_regimes_state_machine(
        df["regime_raw"],
        min_persist=params.min_persist_bars,
    )

    return df


def plot_regimes(
    df: pd.DataFrame,
    n_bars: int = 1000,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    df_plot = df.tail(n_bars).copy()

    if "timestamp" in df_plot.columns:
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
        df_plot = df_plot.set_index("timestamp")

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(df_plot.index, df_plot["close"], linewidth=1.0, label="Close")

    # Define regime colours
    regime_colors = {
        "RANGING": "green",
        "TRENDING": "red",
        "TRANSITION": "gray",
        "NO_DATA": "white",
    }

    # Draw background colour bands
    current_regime = None
    block_start = None
    prev_t = None

    for t, r in zip(df_plot.index, df_plot["regime"]):
        if r != current_regime:
            if current_regime is not None and block_start is not None and prev_t is not None:
                color = regime_colors.get(current_regime, "white")
                ax.axvspan(block_start, prev_t, facecolor=color, alpha=0.15, linewidth=0)
            current_regime = r
            block_start = t
        prev_t = t

    if current_regime is not None and block_start is not None and prev_t is not None:
        color = regime_colors.get(current_regime, "white")
        ax.axvspan(block_start, prev_t, facecolor=color, alpha=0.15, linewidth=0)

    # Add a legend/key for background colours
    legend_handles = []
    for regime, color in regime_colors.items():
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, edgecolor="black")
        legend_handles.append(patch)

    ax.legend(
        legend_handles,
        list(regime_colors.keys()),
        title="Regimes",
        loc="upper left",
        framealpha=1.0
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    if title is not None:
        ax.set_title(title)

    fig.autofmt_xdate()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved regime plot to: {save_path}")

    plt.show()



def main():
    parser = argparse.ArgumentParser(
        description="Label regimes (RANGING / TRENDING / TRANSITION) for an OHLC CSV."
    )
    parser.add_argument("--input", required=True, help="Path to input OHLC CSV.")
    parser.add_argument("--output", required=True, help="Path to output CSV with regimes.")
    parser.add_argument("--time-col", default="timestamp", help="Name of the timestamp column.")
    parser.add_argument("--open-col", default="open")
    parser.add_argument("--high-col", default="high")
    parser.add_argument("--low-col", default="low")
    parser.add_argument("--close-col", default="close")

    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, show a plot of price with regime-colored background.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional path to save the plot image (e.g. regimes.png).",
    )
    parser.add_argument(
        "--plot-bars",
        type=int,
        default=1000,
        help="Number of most recent bars to include in the plot (default: 1000).",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df = df.rename(
        columns={
            args.time_col: "timestamp",
            args.open_col: "open",
            args.high_col: "high",
            args.low_col: "low",
            args.close_col: "close",
        }
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    params = RegimeParams()
    labeled = label_regimes(df, params)

    labeled.reset_index().to_csv(args.output, index=False)
    print(f"Saved labeled data with regimes to: {args.output}")

    if args.plot:
        plot_regimes(
            labeled.reset_index(),
            n_bars=args.plot_bars,
            title=f"Regimes for {args.input}",
            save_path=args.plot_output,
        )


if __name__ == "__main__":
    main()
