from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, NamedTuple, Dict, Any

import numpy as np
import pandas as pd


Direction = Literal["LONG", "SHORT"]


class Signal(NamedTuple):
    time: pd.Timestamp
    instrument: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    regime: str
    meta: Dict[str, Any]


# ============================================================
#                BASE MEAN REVERSION STRATEGY
# ============================================================

@dataclass
class MeanReversionParams:
    bb_length: int = 20
    bb_std: float = 2.0
    atr_length: int = 20
    rsi_length: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    min_stretch_atr: float = 0.75
    sl_atr: float = 1.0
    tp_buffer_atr: float = 0.1
    max_trades_per_day: int = 3
    max_loss_trades_per_day: int = 2
    time_stop_bars: int = 8
    min_range_score: float = 0.6


class MeanReversionStrategy:
    def __init__(self, params: MeanReversionParams | None = None):
        self.params = params or MeanReversionParams()
        self.name = "MR_Bollinger"

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Bollinger bands
        mid = df["close"].ewm(span=self.params.bb_length, adjust=False).mean()
        std = df["close"].rolling(self.params.bb_length).std()
        df["bb_mid"] = mid
        df["bb_upper"] = mid + self.params.bb_std * std
        df["bb_lower"] = mid - self.params.bb_std * std

        # ATR
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.ewm(alpha=1.0 / self.params.atr_length, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / self.params.rsi_length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.params.rsi_length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        return df

    def generate_signals(
        self,
        df_m15: pd.DataFrame,
        instrument: str,
    ) -> List[Signal]:
        """
        Regime-gated mean reversion on M15, using higher-TF regimes:
        - MR allowed when H1 is RANGING/TRANSITION
        - Blocked when H4 or D1 are TRENDING
        """
        p = self.params
        df = self._compute_indicators(df_m15)

        signals: List[Signal] = []
        daily_trade_count: dict[pd.Timestamp, int] = {}

        for t, row in df.iterrows():
            date_key = t.normalize()

            regime_h1 = str(row.get("regime_h1", "UNKNOWN"))
            regime_h4 = str(row.get("regime_h4", "UNKNOWN"))
            regime_d1 = str(row.get("regime_d1", "UNKNOWN"))
            range_score_h1 = float(row.get("range_score_h1", 0.0))

            # Regime filter (slightly relaxed)
            if regime_h1 not in ("RANGING", "TRANSITION"):
                continue
            if range_score_h1 < p.min_range_score:
                continue
            if regime_h4 == "TRENDING":
                continue
            if regime_d1 == "TRENDING":
                continue

            if any(pd.isna(row.get(k)) for k in ["bb_mid", "bb_upper", "bb_lower", "atr", "rsi"]):
                continue

            if daily_trade_count.get(date_key, 0) >= p.max_trades_per_day:
                continue

            close = float(row["close"])
            mid = float(row["bb_mid"])
            upper = float(row["bb_upper"])
            lower = float(row["bb_lower"])
            atr = float(row["atr"])
            rsi = float(row["rsi"])

            # distance from mid
            d_long = mid - close
            d_short = close - mid

            long_cond = (
                (close < lower)
                and (rsi < p.rsi_oversold)
                and (d_long >= p.min_stretch_atr * atr)
            )

            short_cond = (
                (close > upper)
                and (rsi > p.rsi_overbought)
                and (d_short >= p.min_stretch_atr * atr)
            )

            if not (long_cond or short_cond):
                continue

            if long_cond:
                direction: Direction = "LONG"
                sl_dist = max(d_long, p.sl_atr * atr)
                entry = close
                sl = entry - sl_dist
                # revert towards mean, leaving small buffer
                tp = mid - p.tp_buffer_atr * atr
            else:
                direction = "SHORT"
                sl_dist = max(d_short, p.sl_atr * atr)
                entry = close
                sl = entry + sl_dist
                tp = mid + p.tp_buffer_atr * atr

            sig = Signal(
                time=t,
                instrument=instrument,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                strategy_name=self.name,
                regime=regime_h1,
                meta={
                    "range_score_h1": range_score_h1,
                    "atr": atr,
                    "rsi": rsi,
                    "regime_h4": regime_h4,
                    "regime_d1": regime_d1,
                    "time_stop_bars": p.time_stop_bars,
                },
            )
            signals.append(sig)
            daily_trade_count[date_key] = daily_trade_count.get(date_key, 0) + 1

        return signals


# ============================================================
#           FIBONACCI PULLBACK (TREND STRATEGY)
# ============================================================

@dataclass
class FibPullbackParams:
    rsi_length: int = 14
    rsi_trend_threshold: float = 50.0
    swing_lookback: int = 3
    min_retrace: float = 0.382
    max_retrace: float = 0.618
    sl_beyond_level: float = 0.786
    tp_use_extension: bool = False
    max_trades_per_day: int = 2


class FibPullbackStrategy:
    def __init__(self, params: FibPullbackParams | None = None):
        self.params = params or FibPullbackParams()
        self.name = "Fib_Pullback"

    def generate_signals(
        self,
        df_m15: pd.DataFrame,
        instrument: str,
    ) -> List[Signal]:
        """
        INCOMPLETE: This strategy requires fib_* and trend_dir_* columns that are not yet implemented.

        Expected columns:
        - fib_low, fib_high, fib_382, fib_500, fib_618, fib_786
        - trend_dir_h1, trend_dir_h4

        These columns need to be computed from swing high/low detection and Fibonacci retracement
        calculation before this strategy can be used.
        """
        raise NotImplementedError(
            "FibPullbackStrategy requires Fibonacci levels to be precomputed. "
            "The feature engineering pipeline for fib_* columns is not yet implemented. "
            "Use MeanReversionStrategy or TrendEMAPullback instead."
        )

        # Original implementation below - kept for reference but unreachable
        p = self.params
        df = df_m15.copy()

        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / p.rsi_length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / p.rsi_length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        signals: List[Signal] = []
        daily_trade_count: dict[pd.Timestamp, int] = {}

        required_cols = ["fib_low", "fib_high", "fib_382", "fib_500", "fib_618", "fib_786"]

        for t, row in df.iterrows():
            date_key = t.normalize()

            regime_h1 = str(row.get("regime_h1", "UNKNOWN"))
            regime_h4 = str(row.get("regime_h4", "UNKNOWN"))
            regime_d1 = str(row.get("regime_d1", "UNKNOWN"))

            if regime_h1 != "TRENDING" or regime_h4 != "TRENDING":
                continue

            trend_dir_h1 = str(row.get("trend_dir_h1", "UNKNOWN"))
            trend_dir_h4 = str(row.get("trend_dir_h4", "UNKNOWN"))

            if trend_dir_h1 not in ("UP", "DOWN"):
                continue
            if trend_dir_h4 not in ("UP", "DOWN"):
                continue
            if trend_dir_h1 != trend_dir_h4:
                continue

            if daily_trade_count.get(date_key, 0) >= p.max_trades_per_day:
                continue

            if any(pd.isna(row.get(c, np.nan)) for c in required_cols):
                continue

            fib_low = float(row["fib_low"])
            fib_high = float(row["fib_high"])
            fib_382 = float(row["fib_382"])
            fib_500 = float(row["fib_500"])
            fib_618 = float(row["fib_618"])
            fib_786 = float(row["fib_786"])
            c = float(row["close"])
            rsi = float(row["rsi"])

            if trend_dir_h1 == "UP":
                in_zone = (fib_382 <= c <= fib_618)
                confirm = (rsi > p.rsi_trend_threshold) and (c >= fib_500)
                if not (in_zone and confirm):
                    continue

                direction: Direction = "LONG"
                entry = c
                sl_level = min(fib_low, fib_786)
                sl = sl_level
                tp = fib_high
            else:
                if fib_618 < fib_382:
                    in_zone = (fib_618 <= c <= fib_382)
                else:
                    in_zone = (fib_382 <= c <= fib_618)
                confirm = (rsi < (100.0 - p.rsi_trend_threshold)) and (c <= fib_500)
                if not (in_zone and confirm):
                    continue

                direction = "SHORT"
                entry = c
                sl_level = max(fib_high, fib_786)
                sl = sl_level
                tp = fib_low

            sig = Signal(
                time=t,
                instrument=instrument,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                strategy_name=self.name,
                regime=regime_h1,
                meta={
                    "trend_dir_h1": trend_dir_h1,
                    "trend_dir_h4": trend_dir_h4,
                    "fib_low": fib_low,
                    "fib_high": fib_high,
                    "fib_382": fib_382,
                    "fib_500": fib_500,
                    "fib_618": fib_618,
                    "fib_786": fib_786,
                    "rsi": rsi,
                    "regime_h4": regime_h4,
                    "regime_d1": regime_d1,
                    "time_stop_bars": 32,
                },
            )
            signals.append(sig)
            daily_trade_count[date_key] = daily_trade_count.get(date_key, 0) + 1

        return signals


# ============================================================
#        MEAN REVERSION VARIANTS (3 FLAVOURS)
# ============================================================

class MRConservative(MeanReversionStrategy):
    """Stricter MR: larger stretch, fewer but higher quality trades."""
    def __init__(self):
        params = MeanReversionParams(
            bb_length=20,
            bb_std=2.0,
            atr_length=20,
            rsi_length=14,
            rsi_oversold=25.0,
            rsi_overbought=75.0,
            min_stretch_atr=1.2,
            sl_atr=1.2,
            tp_buffer_atr=0.0,
            max_trades_per_day=2,
            max_loss_trades_per_day=2,
            time_stop_bars=32,
            min_range_score=0.6,
        )
        super().__init__(params)
        self.name = "MR_Conservative"


class MRAggressive(MeanReversionStrategy):
    """Looser MR: more trades, still regime-gated."""
    def __init__(self):
        params = MeanReversionParams(
            bb_length=20,
            bb_std=2.0,
            atr_length=20,
            rsi_length=14,
            rsi_oversold=35.0,
            rsi_overbought=65.0,
            min_stretch_atr=0.7,
            sl_atr=1.0,
            tp_buffer_atr=0.2,
            max_trades_per_day=5,
            max_loss_trades_per_day=3,
            time_stop_bars=24,
            min_range_score=0.4,
        )
        super().__init__(params)
        self.name = "MR_Aggressive"


class MRRSI2(MeanReversionStrategy):
    """RSI(2) style MR for higher win-rate."""
    def __init__(self):
        params = MeanReversionParams(
            bb_length=20,
            bb_std=2.5,
            atr_length=20,
            rsi_length=2,
            rsi_oversold=10.0,
            rsi_overbought=90.0,
            min_stretch_atr=1.0,
            sl_atr=1.2,
            tp_buffer_atr=0.0,
            max_trades_per_day=3,
            max_loss_trades_per_day=2,
            time_stop_bars=32,
            min_range_score=0.5,
        )
        super().__init__(params)
        self.name = "MR_RSI2"


# ============================================================
#          TREND EMA PULLBACK STRATEGY (TREND-FOL)
# ============================================================

@dataclass
class TrendEMAPullbackParams:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_length: int = 14
    rsi_pullback_buy: float = 40.0
    rsi_pullback_sell: float = 60.0
    max_trades_per_day: int = 3
    time_stop_bars: int = 32
    sl_atr_mult: float = 1.5
    tp_rr: float = 1.2   # >1R reward in trends


class TrendEMAPullback:
    """
    Simple trend-following pullback strategy:
    - Uses H1/H4 TRENDING regimes for context
    - Uses EMA20/EMA50 on M15 for local trend
    - Buys/sells pullbacks with ATR-based SL and RR>1 TP
    """
    def __init__(self, params: TrendEMAPullbackParams | None = None):
        self.params = params or TrendEMAPullbackParams()
        self.name = "Trend_EMA_Pullback"

    def generate_signals(self, df_m15: pd.DataFrame, instrument: str) -> List[Signal]:
        p = self.params
        df = df_m15.copy()

        # EMAs on M15
        df["ema_fast"] = df["close"].ewm(span=p.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=p.ema_slow, adjust=False).mean()

        # ATR
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.ewm(alpha=1.0 / 20, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / p.rsi_length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / p.rsi_length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        signals: List[Signal] = []
        daily_trade_count: dict[pd.Timestamp, int] = {}

        for t, row in df.iterrows():
            date_key = t.normalize()

            regime_h1 = str(row.get("regime_h1", "UNKNOWN"))
            regime_h4 = str(row.get("regime_h4", "UNKNOWN"))

            if regime_h1 != "TRENDING" or regime_h4 != "TRENDING":
                continue

            if any(pd.isna(row.get(k)) for k in ["ema_fast", "ema_slow", "atr", "rsi"]):
                continue

            if daily_trade_count.get(date_key, 0) >= p.max_trades_per_day:
                continue

            ema_fast = float(row["ema_fast"])
            ema_slow = float(row["ema_slow"])
            atr = float(row["atr"])
            rsi = float(row["rsi"])
            price = float(row["close"])

            long_trend = ema_fast > ema_slow
            short_trend = ema_fast < ema_slow

            long_pullback = long_trend and (rsi <= p.rsi_pullback_buy) and (price <= ema_fast)
            short_pullback = short_trend and (rsi >= p.rsi_pullback_sell) and (price >= ema_fast)

            if not (long_pullback or short_pullback):
                continue

            if long_pullback:
                direction: Direction = "LONG"
                entry = price
                sl = entry - p.sl_atr_mult * atr
                tp = entry + p.sl_atr_mult * atr * p.tp_rr
            else:
                direction = "SHORT"
                entry = price
                sl = entry + p.sl_atr_mult * atr
                tp = entry - p.sl_atr_mult * atr * p.tp_rr

            sig = Signal(
                time=t,
                instrument=instrument,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                strategy_name=self.name,
                regime=regime_h1,
                meta={
                    "atr": atr,
                    "rsi": rsi,
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "regime_h4": regime_h4,
                    "time_stop_bars": p.time_stop_bars,
                },
            )
            signals.append(sig)
            daily_trade_count[date_key] = daily_trade_count.get(date_key, 0) + 1

        return signals
