# cost_model.py
"""
Simple FTMO-style cost model: per-instrument spread (in price units)
and an approximate fixed cost in R (commission + residual slippage).

We use fixed_R ≈ 0.05 for FX pairs:
- $5 per 1-lot round turn
- typical SL ~10 pips => ~0.5 pip cost ≈ 0.05R

Indices & oil: no explicit commission at FTMO, so fixed_R = 0.0 for now.
Tune these numbers later once you have precise broker stats.
"""

INSTRUMENT_COSTS = {
    # ----- FX majors & crosses -----
    # spreads from your FTMO screenshot (price units)
    "EURUSD": {"spread": 0.00001, "fixed_R": 0.05},   # spread often 0 but add tiny buffer
    "GBPUSD": {"spread": 0.00008, "fixed_R": 0.05},
    "EURGBP": {"spread": 0.00002, "fixed_R": 0.05},
    "USDJPY": {"spread": 0.00500, "fixed_R": 0.05},
    "USDCAD": {"spread": 0.00003, "fixed_R": 0.05},
    "USDCHF": {"spread": 0.00004, "fixed_R": 0.05},
    "AUDUSD": {"spread": 0.00002, "fixed_R": 0.05},
    "NZDUSD": {"spread": 0.00007, "fixed_R": 0.05},
    "GBPJPY": {"spread": 0.02200, "fixed_R": 0.05},
    "AUDJPY": {"spread": 0.00800, "fixed_R": 0.05},

    # ----- Indices (points) -----
    "US30":   {"spread": 2.08, "fixed_R": 0.00},
    "US100":  {"spread": 1.60, "fixed_R": 0.00},
    "US500":  {"spread": 0.46, "fixed_R": 0.00},
    "US2000": {"spread": 1.00, "fixed_R": 0.00},  # '---' on screenshot -> conservative guess
    "GER40":  {"spread": 1.19, "fixed_R": 0.00},
    "UK100":  {"spread": 0.75, "fixed_R": 0.00},
    "EU50":   {"spread": 1.46, "fixed_R": 0.00},
    "JP225":  {"spread": 4.34, "fixed_R": 0.00},
    "HK50":   {"spread": 4.54, "fixed_R": 0.00},
    "AUS200": {"spread": 1.00, "fixed_R": 0.00},  # '---' -> conservative guess

    # ----- Oil (commodities) -----
    "USOIL":  {"spread": 0.022, "fixed_R": 0.00},
    "UKOIL":  {"spread": 0.014, "fixed_R": 0.00},
}


def get_cost_config(instrument: str) -> dict:
    """
    Returns a dict with keys:
      - 'spread': price units
      - 'fixed_R': R cost per trade
    Defaults to zero cost if instrument not found.
    """
    base = {"spread": 0.0, "fixed_R": 0.0}
    cfg = INSTRUMENT_COSTS.get(instrument)
    if cfg is None:
        return base
    out = base.copy()
    out.update(cfg)
    return out
