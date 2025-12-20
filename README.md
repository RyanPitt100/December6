# TrendEMA Pullback Trading Bot

A multi-timeframe trend-following trading system designed for FTMO evaluation challenges. The bot uses EMA pullback entries with regime filtering to trade forex pairs and indices via MetaTrader 5.

## Overview

This trading bot implements a systematic approach to catching pullbacks within larger trends. It combines:

- **Multi-timeframe regime analysis** (H1/H4/D1) to confirm trend direction
- **EMA pullback entries** on 15-minute charts for precise timing
- **ATR-based risk management** for adaptive position sizing
- **FTMO-compliant risk overlays** with soft/hard brakes

The system has been validated through walk-forward optimization across 22 instruments over 4+ years of historical data.

---

## Strategy: TrendEMA Pullback

### Entry Logic

The strategy looks for pullback opportunities within established trends:

**Long Entry Conditions:**
1. H1 regime = TRENDING
2. H4 regime = TRENDING
3. EMA20 > EMA50 (15m uptrend confirmed)
4. RSI ≤ 40 (oversold pullback)
5. Price ≤ EMA20 (price has pulled back to the EMA)

**Short Entry Conditions:**
1. H1 regime = TRENDING
2. H4 regime = TRENDING
3. EMA20 < EMA50 (15m downtrend confirmed)
4. RSI ≥ 60 (overbought pullback)
5. Price ≥ EMA20 (price has pulled back to the EMA)

### Risk Parameters

| Parameter | Value |
|-----------|-------|
| Stop Loss | 1.5x ATR from entry |
| Take Profit | 1.8x ATR from entry (1.2 R:R) |
| Risk per Trade | 0.5% of equity |
| Max Trades/Day | 3 per instrument |

### Regime Filtering

The bot uses a dual-indicator regime classification system:

- **Efficiency Ratio (ER):** Measures trend strength (0-1 scale)
- **ADX:** Confirms directional movement strength

Regimes are pre-computed from historical data using 40th/60th percentile thresholds, ensuring consistency between backtest and live execution.

---

## Risk Management

### FTMO Overlays

The bot includes built-in FTMO compliance with internal safety buffers:

| Limit | FTMO Raw | Internal Buffer |
|-------|----------|-----------------|
| Daily Loss | 5% | 4% |
| Total Loss | 10% | 8% |

### Brake System

**Soft Brake (50% of daily limit):**
- Blocks new trade entries
- Maintains existing positions
- Allows positions to hit TP/SL naturally

**Hard Brake (90% of daily limit):**
- Immediately flattens all positions
- Prevents any new trades
- Emergency protection mode

### Risk Envelope Guard

Prevents combined risk (realized DD + open position risk) from exceeding the daily limit minus a 0.5% safety margin. This is critical for prop firm compliance where unrealized losses count against limits.

### Position Sizing

Position size is calculated dynamically based on:

```
size_lots = (equity × 0.5%) / (SL_distance_pips × pip_value_per_lot)
```

Pip values are instrument-specific:
- JPY pairs: $10 per pip per lot (2-decimal)
- Standard forex: $10 per pip per lot (4-decimal)
- Indices: $1 per point per lot

---

## Portfolio

The current portfolio consists of 6 instruments selected through walk-forward optimization:

| Instrument | Weight | Type |
|------------|--------|------|
| EURUSD | 16.67% | Forex Major |
| USDJPY | 16.67% | Forex Major |
| USDCAD | 16.67% | Forex Major |
| JP225.cash | 16.67% | Index (Nikkei) |
| AUS200.cash | 16.67% | Index (ASX) |
| GER40.cash | 16.67% | Index (DAX) |

### Selection Criteria

Instruments must pass all walk-forward criteria:
- Minimum 50 out-of-sample trades
- Positive average R-multiple
- Maximum drawdown ≤ 15R
- Sharpe ratio ≥ 0
- Profit factor ≥ 1.0

---

## Walk-Forward Optimization

The bot uses rolling walk-forward validation to prevent overfitting:

```
Train Window: 2 years
Test Window:  6 months (out-of-sample)
Step Size:    6 months
```

For each window:
1. Train strategy on 2-year historical data
2. Test on next 6 months (completely unseen data)
3. Record out-of-sample metrics
4. Roll forward and repeat

This process validates that the strategy edge persists on unseen data.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Live MT5 Eval Runner                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  MT5 Bridge │──│Risk Manager │──│ Portfolio Controller│  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                │                    │              │
│         ▼                ▼                    ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Multi-TF    │  │   FTMO      │  │  Signal Generator   │  │
│  │ Data Builder│  │  Overlays   │  │  (TrendEMAPullback) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                                     │              │
│         ▼                                     ▼              │
│  ┌─────────────┐                    ┌─────────────────────┐  │
│  │Trade Logger │                    │  Discord Notifier   │  │
│  │  (SQLite)   │                    │    (Webhooks)       │  │
│  └─────────────┘                    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Modules

| File | Purpose |
|------|---------|
| `live_mt5_eval_runner.py` | Main execution loop (runs every 15 min) |
| `strategies.py` | TrendEMAPullback strategy logic |
| `risk_manager.py` | FTMO overlays and risk limits |
| `portfolio_controller.py` | Position sizing and order generation |
| `mt5_bridge.py` | MetaTrader 5 connection |
| `trade_logger.py` | SQLite trade logging |
| `live_signal_generator.py` | Canonical signal generation |
| `regime_labeler.py` | Multi-timeframe regime classification |

---

## Configuration

### instrument_portfolio.yml
Defines which instruments to trade and their weights.

### ftmo_overlays.yml
FTMO-specific risk limits and evaluation parameters:
- Starting equity
- Profit target (10% for eval)
- Daily/total loss limits
- Soft/hard brake thresholds

### portfolio_risk_settings.yml
Position sizing and portfolio constraints:
- Risk per trade (0.5%)
- Maximum open trades (10)
- Maximum risk at once (3%)

### regime_thresholds.yml
Pre-computed regime classification thresholds per instrument/timeframe. Ensures live regime labels match historical backtest.

### config/credentials.yml
MT5 login credentials (not tracked in git):
```yaml
mt5:
  login: YOUR_LOGIN
  password: YOUR_PASSWORD
  server: FTMO-Demo2
```

---

## Usage

### Running the Bot

```bash
# Live execution (paper trading)
python live_mt5_eval_runner.py --mode eval --poll-interval-seconds 60

# Dry-run mode (no orders sent)
python live_mt5_eval_runner.py --mode eval --dry-run

# With iteration limit (for testing)
python live_mt5_eval_runner.py --mode eval --max-iterations 10
```

### Running Walk-Forward Optimization

```bash
python walkforward_multi_instrument.py
```

### Replaying Historical Signals

```bash
# Replay last 14 days from MT5
python replay_last_2_weeks.py --days 14 --verbose
```

### Testing Backtest vs Live Consistency

```bash
python test_backtest_vs_live_consistency.py --all-instruments --days 60
```

---

## Trade Logging

All trades are logged to `trades.db` (SQLite) with:

**Execution Context:**
- ATR at entry
- Spread at entry
- Trading session (Asia/London/NY/Overlap)
- Entry hour and day of week

**Trade Outcomes:**
- P&L in currency and percentage
- R-multiple
- Duration
- Exit reason (TP/SL/Manual)
- MAE/MFE (for SL/TP optimization)

---

## Performance Metrics

From walk-forward validation (2020-2024):

| Metric | Value |
|--------|-------|
| Pass Rate | 87.5% (7/8 cycles) |
| Win Rate | ~62% |
| Avg R-Multiple | +0.15R |
| Max Cycle Drawdown | 4.7% |
| Trades per Day | ~0.29 |

---

## Requirements

- Python 3.8+
- MetaTrader 5 terminal (running)
- FTMO demo or funded account

### Python Dependencies

```
MetaTrader5
pandas
numpy
pyyaml
requests (for Discord)
```

---

## Disclaimer

This software is for educational purposes only. Trading forex and CFDs carries significant risk. Past performance does not guarantee future results. Use at your own risk.

---

## License

Private repository - all rights reserved.
