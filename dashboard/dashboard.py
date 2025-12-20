"""
ORBIT Dashboard - RegimeGate Trend Engine
Run with: streamlit run dashboard.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml
import sys
import base64

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# CONFIGURATION
# =============================================================================

VERSION = "0.9.0"
BUILD_NAME = "December-6"
PRODUCT_NAME = "ORBIT"
STRATEGY_NAME = "RegimeGate Trend Engine"

# Paths
PARENT_DIR = Path(__file__).parent.parent
DB_PATH = PARENT_DIR / "trades.db"
UNIVERSE_PATH = PARENT_DIR / "universe.yml"
LOG_FILE_PATH = PARENT_DIR / "bot_output.log"
LOGO_PATH = Path(__file__).parent / "orbit_logo.png"

# =============================================================================
# COLOR SYSTEM - Single accent (Cyan) + semantic colors
# =============================================================================

CYAN = '#00D4AA'        # Primary accent
CYAN_DIM = '#00A888'    # Muted accent
BG_DARK = '#0a0e14'     # Main background
BG_CARD = '#141a22'     # Card background
BG_ELEVATED = '#1a222c' # Elevated surfaces
BORDER = '#242d3a'      # Subtle borders
TEXT_PRIMARY = '#e6edf3'
TEXT_MUTED = '#7d8590'
TEXT_DIM = '#484f58'

# Semantic colors (only for data)
GREEN = '#3fb950'       # Positive P&L only
RED = '#f85149'         # Negative P&L / risk warnings only
YELLOW = '#d29922'      # Warnings

# Page config
st.set_page_config(
    page_title=f"{PRODUCT_NAME} Dashboard",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS STYLES - Modern, clean design
# =============================================================================

st.markdown(f"""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Main background with subtle gradient */
    .stApp {{
        background: linear-gradient(180deg, {BG_DARK} 0%, #0d1117 100%);
        color: {TEXT_PRIMARY};
    }}

    /* Hide default streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Typography */
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    code, pre, .mono {{
        font-family: 'JetBrains Mono', 'Consolas', monospace;
    }}

    /* Card styling - softer, more padding */
    .card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }}

    .card-elevated {{
        background: {BG_ELEVATED};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 24px;
        margin: 8px 0;
    }}

    /* Metric styling */
    .metric-label {{
        color: {TEXT_MUTED};
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }}

    .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: {TEXT_PRIMARY};
    }}

    .metric-value-lg {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
    }}

    /* Pills */
    .pill {{
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 8px;
    }}

    .pill-status {{
        background: rgba(0, 212, 170, 0.15);
        color: {CYAN};
        border: 1px solid rgba(0, 212, 170, 0.3);
    }}

    .pill-status-offline {{
        background: rgba(248, 81, 73, 0.15);
        color: {RED};
        border: 1px solid rgba(248, 81, 73, 0.3);
    }}

    .pill-status-warning {{
        background: rgba(210, 153, 34, 0.15);
        color: {YELLOW};
        border: 1px solid rgba(210, 153, 34, 0.3);
    }}

    .pill-info {{
        background: {BG_ELEVATED};
        color: {TEXT_MUTED};
        border: 1px solid {BORDER};
    }}

    .pill-env {{
        background: rgba(0, 212, 170, 0.1);
        color: {CYAN};
        border: 1px solid rgba(0, 212, 170, 0.2);
        font-weight: 600;
    }}

    /* Section headers */
    .section-title {{
        color: {TEXT_PRIMARY};
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid {BORDER};
    }}

    /* Tables */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }}

    .data-table th {{
        padding: 12px 16px;
        text-align: left;
        color: {TEXT_MUTED};
        font-weight: 500;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background: {BG_ELEVATED};
        border-bottom: 1px solid {BORDER};
    }}

    .data-table td {{
        padding: 10px 16px;
        color: {TEXT_PRIMARY};
        border-bottom: 1px solid {BORDER};
    }}

    .data-table tr:last-child td {{
        border-bottom: none;
    }}

    .data-table tr:hover td {{
        background: rgba(0, 212, 170, 0.03);
    }}

    /* Progress bars */
    .progress-bar {{
        height: 6px;
        background: {BG_ELEVATED};
        border-radius: 3px;
        overflow: hidden;
    }}

    .progress-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }}

    /* Streamlit overrides */
    .stButton > button {{
        background: transparent;
        color: {TEXT_MUTED};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s;
    }}

    .stButton > button:hover {{
        background: {BG_ELEVATED};
        color: {TEXT_PRIMARY};
        border-color: {CYAN};
    }}

    .stSelectbox > div > div {{
        background: {BG_ELEVATED};
        border-color: {BORDER};
        border-radius: 8px;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

    .stCode {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}

    pre {{
        background: {BG_CARD} !important;
        color: {TEXT_PRIMARY} !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        color: {TEXT_MUTED};
        padding: 8px 16px;
    }}

    .stTabs [aria-selected="true"] {{
        background: {BG_ELEVATED};
        border-color: {CYAN};
        color: {CYAN};
    }}

    /* Empty states */
    .empty-state {{
        text-align: center;
        padding: 48px 24px;
        color: {TEXT_MUTED};
    }}

    .empty-state-title {{
        font-size: 1rem;
        margin-bottom: 8px;
        color: {TEXT_PRIMARY};
    }}

    .empty-state-subtitle {{
        font-size: 0.85rem;
        color: {TEXT_DIM};
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def get_connection():
    """Get database connection."""
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH)

def load_portfolio_instruments():
    """Load instruments from universe.yml config."""
    default_instruments = ["USDCAD", "JP225.cash", "AUS200.cash", "GER40.cash", "EURUSD", "USDJPY"]

    if not UNIVERSE_PATH.exists():
        return default_instruments

    with open(UNIVERSE_PATH, 'r') as f:
        config = yaml.safe_load(f)

    instruments = []
    for item in config.get('instruments', []):
        if item.get('enabled', True):
            instruments.append(item['symbol'])

    return instruments if instruments else default_instruments

def load_instrument_ohlc(instrument: str, limit: int = 100):
    """Load recent OHLC data for an instrument from MT5."""
    try:
        import mt5_bridge
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None
        df = mt5_bridge.fetch_ohlc(instrument, "15m", n_bars=limit)
        if df is not None:
            df = df.reset_index()
        return df
    except:
        return None

def load_instrument_trades(instrument: str):
    """Load trades for a specific instrument."""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame(), pd.DataFrame()

    df_entries = pd.read_sql("""
        SELECT timestamp, instrument, direction, entry_price, sl_price, tp_price
        FROM executions WHERE instrument = ?
        ORDER BY timestamp DESC LIMIT 50
    """, conn, params=(instrument,))

    df_exits = pd.read_sql("""
        SELECT timestamp, instrument, direction, entry_price, exit_price, pnl_currency
        FROM trade_closes WHERE instrument = ?
        ORDER BY timestamp DESC LIMIT 50
    """, conn, params=(instrument,))

    conn.close()

    if not df_entries.empty:
        df_entries['timestamp'] = pd.to_datetime(df_entries['timestamp'])
    if not df_exits.empty:
        df_exits['timestamp'] = pd.to_datetime(df_exits['timestamp'])

    return df_entries, df_exits

def load_console_output(lines: int = 100):
    """Load the most recent console output from the log file."""
    if not LOG_FILE_PATH.exists():
        return ["[Log file not found]"]

    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            if not all_lines:
                return ["[Log file is empty]"]
            return all_lines[-lines:] if len(all_lines) > lines else all_lines
    except Exception as e:
        return [f"[Error reading log: {e}]"]

def load_equity_curve():
    """Load equity curve from account snapshots."""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()

    df = pd.read_sql("""
        SELECT timestamp, equity, balance, total_dd_pct, open_positions
        FROM account_snapshots ORDER BY timestamp
    """, conn)
    conn.close()

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def load_trade_stats():
    """Load trade statistics."""
    conn = get_connection()
    if conn is None:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "profit_factor": 0.0, "avg_r": 0.0}

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trade_closes")
    total_trades = cursor.fetchone()[0]

    if total_trades == 0:
        conn.close()
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "profit_factor": 0.0, "avg_r": 0.0}

    cursor.execute("SELECT COUNT(*) FROM trade_closes WHERE pnl_currency > 0")
    wins = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(pnl_currency) FROM trade_closes")
    total_pnl = cursor.fetchone()[0] or 0.0

    cursor.execute("SELECT SUM(pnl_currency) FROM trade_closes WHERE pnl_currency > 0")
    gross_profit = cursor.fetchone()[0] or 0.0
    cursor.execute("SELECT SUM(ABS(pnl_currency)) FROM trade_closes WHERE pnl_currency < 0")
    gross_loss = cursor.fetchone()[0] or 1.0

    cursor.execute("SELECT AVG(r_multiple) FROM trade_closes WHERE r_multiple IS NOT NULL")
    avg_r = cursor.fetchone()[0] or 0.0

    conn.close()

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": total_trades - wins,
        "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0,
        "total_pnl": total_pnl,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0,
        "avg_r": avg_r,
    }

def load_recent_signals(limit=20):
    """Load recent signals."""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()

    df = pd.read_sql(f"""
        SELECT timestamp, instrument, direction, entry_price, sl_price, tp_price,
               reason, executed, regime_h1, regime_h4
        FROM signals ORDER BY timestamp DESC LIMIT {limit}
    """, conn)
    conn.close()

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def load_recent_trades(limit=20):
    """Load recent closed trades."""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()

    df = pd.read_sql(f"""
        SELECT timestamp, instrument, direction, entry_price, exit_price,
               pnl_currency, r_multiple, exit_reason, duration_minutes
        FROM trade_closes ORDER BY timestamp DESC LIMIT {limit}
    """, conn)
    conn.close()

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def load_recent_events(limit=30):
    """Load recent system events."""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()

    df = pd.read_sql(f"""
        SELECT timestamp, level, category, message
        FROM system_events ORDER BY timestamp DESC LIMIT {limit}
    """, conn)
    conn.close()

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_bot_status():
    """Check if bot is running based on recent snapshots."""
    conn = get_connection()
    if conn is None:
        return "Unknown", None

    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM account_snapshots ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return "No Data", None

    last_update = datetime.fromisoformat(row[0])
    age_minutes = (datetime.now(timezone.utc).replace(tzinfo=None) - last_update).total_seconds() / 60

    if age_minutes < 20:
        return "Online", last_update
    elif age_minutes < 60:
        return "Stalled", last_update
    else:
        return "Offline", last_update

def load_risk_state():
    """Load latest risk state from account snapshots."""
    conn = get_connection()
    if conn is None:
        return None

    cursor = conn.cursor()
    cursor.execute("""
        SELECT daily_pnl_pct, total_dd_pct, open_positions, risk_state, risk_action,
               eval_profit_target_hit, breached_daily_limit, breached_total_limit,
               equity, balance, start_of_day_equity
        FROM account_snapshots
        ORDER BY timestamp DESC LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "daily_pnl_pct": row[0] or 0.0,
        "total_dd_pct": row[1] or 0.0,
        "open_positions": row[2] or 0,
        "risk_state": row[3] or "UNKNOWN",
        "risk_action": row[4] or "NONE",
        "eval_profit_target_hit": row[5] or 0,
        "breached_daily_limit": row[6] or 0,
        "breached_total_limit": row[7] or 0,
        "equity": row[8] or 0.0,
        "balance": row[9] or 0.0,
        "start_of_day_equity": row[10] or 0.0,
    }

def get_next_bar_time():
    """Calculate time until next 15m bar."""
    now = datetime.now(timezone.utc)
    minutes_past = now.minute % 15
    seconds_past = now.second
    seconds_until = (15 - minutes_past) * 60 - seconds_past
    return seconds_until

def get_logo_base64():
    """Load logo as base64 if exists."""
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# =============================================================================
# HEADER BAR
# =============================================================================

status, last_update = get_bot_status()
instruments = load_portfolio_instruments()
logo_b64 = get_logo_base64()

# Status pill styling
if status == "Online":
    status_pill = f'<span class="pill pill-status">● Online</span>'
elif status == "Stalled":
    status_pill = f'<span class="pill pill-status-warning">● Stalled</span>'
elif status == "Offline":
    status_pill = f'<span class="pill pill-status-offline">● Offline</span>'
else:
    status_pill = f'<span class="pill pill-info">○ No Data</span>'

# Logo HTML
logo_html = ""
if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height: 32px; margin-right: 12px; vertical-align: middle;">'
else:
    logo_html = f'<span style="font-size: 1.5rem; margin-right: 8px; color: {CYAN};">◎</span>'

# Header
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center;
            padding: 16px 0; margin-bottom: 24px; border-bottom: 1px solid {BORDER};">
    <div style="display: flex; align-items: center;">
        {logo_html}
        <div>
            <span style="font-size: 1.4rem; font-weight: 700; color: {TEXT_PRIMARY};">{PRODUCT_NAME}</span>
            <span style="color: {TEXT_DIM}; margin-left: 8px;">|</span>
            <span style="color: {TEXT_MUTED}; margin-left: 8px; font-size: 0.9rem;">{STRATEGY_NAME}</span>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        {status_pill}
        <span class="pill pill-info" style="font-family: 'JetBrains Mono', monospace;">
            {last_update.strftime('%H:%M') if last_update else '--:--'} UTC
        </span>
        <span class="pill pill-info" style="font-family: 'JetBrains Mono', monospace;">
            {datetime.now().strftime('%H:%M')}
        </span>
        <span class="pill pill-env">SIM</span>
        <span class="pill pill-info">v{VERSION}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sub-header info chips
st.markdown(f"""
<div style="display: flex; gap: 16px; margin-bottom: 24px; font-size: 0.8rem; color: {TEXT_MUTED};">
    <span>Universe: <span style="color: {CYAN};">{len(instruments)}</span> instruments</span>
    <span>TFs: <span style="color: {CYAN};">15m / 1h / 4h / 1d</span></span>
    <span>Build: <span style="color: {TEXT_DIM};">{BUILD_NAME}</span></span>
</div>
""", unsafe_allow_html=True)

# Refresh button (small, top right)
col_spacer, col_refresh = st.columns([11, 1])
with col_refresh:
    if st.button("↻", help="Refresh dashboard"):
        st.rerun()

# =============================================================================
# MAIN CONTENT
# =============================================================================

stats = load_trade_stats()
df_equity = load_equity_curve()
risk_state = load_risk_state()

# Calculate time until next evaluation
seconds_until = get_next_bar_time()
minutes_until = seconds_until // 60
secs_until = seconds_until % 60

# =============================================================================
# PERFORMANCE METRICS (or Empty State)
# =============================================================================

if stats['total_trades'] == 0:
    # Empty state - show signal health instead
    st.markdown(f'<div class="section-title">System Status</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="empty-state">
                <div class="empty-state-title">No trades yet</div>
                <div class="empty-state-subtitle">Waiting for valid signals across {len(instruments)} instruments</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Next Evaluation</div>
            <div class="metric-value" style="color: {CYAN};">{minutes_until}m {secs_until}s</div>
            <div style="color: {TEXT_DIM}; font-size: 0.75rem; margin-top: 8px;">15m bar close</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Signal health metrics
        df_signals = load_recent_signals(50)
        signals_today = len(df_signals[df_signals['timestamp'].dt.date == datetime.now().date()]) if not df_signals.empty else 0

        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Signals Today</div>
            <div class="metric-value" style="color: {CYAN};">{signals_today}</div>
            <div style="color: {TEXT_DIM}; font-size: 0.75rem; margin-top: 8px;">Across all instruments</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Show actual performance metrics
    st.markdown(f'<div class="section-title">Performance</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    pnl_color = GREEN if stats['total_pnl'] >= 0 else RED
    pnl_sign = "+" if stats['total_pnl'] >= 0 else ""

    with col1:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Total P&L</div>
            <div class="metric-value-lg" style="color: {pnl_color};">{pnl_sign}${stats['total_pnl']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{stats['win_rate']:.1f}%</div>
            <div style="color: {TEXT_DIM}; font-size: 0.75rem; margin-top: 4px;">
                {stats['wins']}W / {stats['losses']}L
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        pf_color = GREEN if stats['profit_factor'] >= 1.5 else CYAN if stats['profit_factor'] >= 1.0 else RED
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value" style="color: {pf_color};">{stats['profit_factor']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_r_color = GREEN if stats['avg_r'] > 0 else RED
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Avg R</div>
            <div class="metric-value" style="color: {avg_r_color};">{stats['avg_r']:+.2f}R</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value" style="color: {CYAN};">{stats['total_trades']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# EQUITY CURVE (Hero section)
# =============================================================================

st.markdown(f'<div class="section-title">Equity</div>', unsafe_allow_html=True)

if df_equity.empty:
    st.markdown(f"""
    <div class="card">
        <div class="empty-state">
            <div class="empty-state-title">No equity data yet</div>
            <div class="empty-state-subtitle">Account snapshots will appear here every 15 minutes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Equity stats row
    current_equity = df_equity['equity'].iloc[-1]
    current_balance = df_equity['balance'].iloc[-1]
    current_dd = df_equity['total_dd_pct'].iloc[-1]
    open_pos = df_equity['open_positions'].iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Equity</div>
            <div class="metric-value" style="color: {CYAN};">${current_equity:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Balance</div>
            <div class="metric-value">${current_balance:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        dd_color = GREEN if current_dd < 2 else YELLOW if current_dd < 4 else RED
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Drawdown</div>
            <div class="metric-value" style="color: {dd_color};">{current_dd:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Open Positions</div>
            <div class="metric-value" style="color: {CYAN};">{int(open_pos)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        # Calculate exposure
        exposure = open_pos * 0.5  # Assuming 0.5% per position
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Open Risk</div>
            <div class="metric-value">{exposure:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Time filter pills
    time_col1, time_col2 = st.columns([1, 5])
    with time_col1:
        time_range = st.selectbox(
            "Range",
            ["1D", "1W", "1M", "All"],
            index=1,
            label_visibility="collapsed"
        )

    # Filter data
    df_plot = df_equity.copy()
    if time_range == "1D":
        cutoff = datetime.utcnow() - timedelta(hours=24)
        df_plot = df_plot[df_plot['timestamp'] >= cutoff]
    elif time_range == "1W":
        cutoff = datetime.utcnow() - timedelta(days=7)
        df_plot = df_plot[df_plot['timestamp'] >= cutoff]
    elif time_range == "1M":
        cutoff = datetime.utcnow() - timedelta(days=30)
        df_plot = df_plot[df_plot['timestamp'] >= cutoff]

    if not df_plot.empty:
        # Equity chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['timestamp'],
            y=df_plot['equity'],
            name='Equity',
            line=dict(color=CYAN, width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.08)'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['timestamp'],
            y=df_plot['balance'],
            name='Balance',
            line=dict(color=TEXT_MUTED, width=1, dash='dot')
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color=TEXT_MUTED, size=11)),
            paper_bgcolor='transparent',
            plot_bgcolor=BG_CARD,
            font=dict(color=TEXT_MUTED, family='Inter'),
            xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=False),
            yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickformat='$,.0f', side='right'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown sub-chart
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df_plot['timestamp'],
            y=df_plot['total_dd_pct'],
            fill='tozeroy',
            line=dict(color=RED, width=1),
            fillcolor='rgba(248, 81, 73, 0.2)'
        ))
        # Add limit lines
        fig_dd.add_hline(y=4, line_dash="dash", line_color=YELLOW, annotation_text="Daily Limit (4%)")
        fig_dd.add_hline(y=8, line_dash="dash", line_color=RED, annotation_text="Total Limit (8%)")
        fig_dd.update_layout(
            height=120,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='transparent',
            plot_bgcolor=BG_CARD,
            font=dict(color=TEXT_MUTED),
            xaxis=dict(gridcolor=BORDER, showticklabels=False, showgrid=False),
            yaxis=dict(gridcolor=BORDER, ticksuffix='%', side='right'),
            showlegend=False
        )
        st.plotly_chart(fig_dd, use_container_width=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# RISK & COMPLIANCE
# =============================================================================

st.markdown(f'<div class="section-title">Risk & Compliance</div>', unsafe_allow_html=True)

if risk_state:
    col1, col2, col3, col4, col5 = st.columns(5)

    # Daily DD
    daily_dd = abs(risk_state['daily_pnl_pct']) if risk_state['daily_pnl_pct'] < 0 else 0
    daily_limit = 4.0
    daily_pct = min(daily_dd / daily_limit * 100, 100)
    daily_color = GREEN if daily_dd < 2 else YELLOW if daily_dd < 3.5 else RED

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Daily DD Used</div>
            <div class="metric-value" style="color: {daily_color};">{daily_dd:.2f}%</div>
            <div style="margin-top: 8px;">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {daily_pct}%; background: {daily_color};"></div>
                </div>
            </div>
            <div style="color: {TEXT_DIM}; font-size: 0.7rem; margin-top: 4px;">of {daily_limit}% limit</div>
        </div>
        """, unsafe_allow_html=True)

    # Total DD
    total_dd = risk_state['total_dd_pct']
    total_limit = 8.0
    total_pct = min(total_dd / total_limit * 100, 100)
    total_color = GREEN if total_dd < 4 else YELLOW if total_dd < 7 else RED

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Total DD Used</div>
            <div class="metric-value" style="color: {total_color};">{total_dd:.2f}%</div>
            <div style="margin-top: 8px;">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {total_pct}%; background: {total_color};"></div>
                </div>
            </div>
            <div style="color: {TEXT_DIM}; font-size: 0.7rem; margin-top: 4px;">of {total_limit}% limit</div>
        </div>
        """, unsafe_allow_html=True)

    # Open Risk
    open_risk = risk_state['open_positions'] * 0.5  # 0.5% per position
    max_risk = 3.0
    risk_pct = min(open_risk / max_risk * 100, 100)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Open Risk</div>
            <div class="metric-value">{open_risk:.1f}%</div>
            <div style="margin-top: 8px;">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {risk_pct}%; background: {CYAN};"></div>
                </div>
            </div>
            <div style="color: {TEXT_DIM}; font-size: 0.7rem; margin-top: 4px;">{risk_state['open_positions']} positions @ 0.5%</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk State
    risk_action = risk_state['risk_action'] or "NORMAL"
    if risk_action in ["BLOCK_NEW_TRADES", "MUST_FLATTEN"]:
        state_color = RED
        state_text = "HARD BRAKE" if risk_action == "MUST_FLATTEN" else "SOFT BRAKE"
    elif risk_state['breached_daily_limit'] or risk_state['breached_total_limit']:
        state_color = RED
        state_text = "LIMIT BREACHED"
    elif risk_state['eval_profit_target_hit']:
        state_color = GREEN
        state_text = "TARGET HIT"
    else:
        state_color = GREEN
        state_text = "NORMAL"

    with col4:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Risk State</div>
            <div class="metric-value" style="color: {state_color};">{state_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # Next Reset
    now_utc = datetime.now(timezone.utc)
    next_reset = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    hours_until_reset = (next_reset - now_utc).total_seconds() / 3600

    with col5:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-label">Daily Reset</div>
            <div class="metric-value">{hours_until_reset:.1f}h</div>
            <div style="color: {TEXT_DIM}; font-size: 0.7rem; margin-top: 4px;">00:00 UTC</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class="card">
        <div class="empty-state">
            <div class="empty-state-title">No risk data available</div>
            <div class="empty-state-subtitle">Risk metrics will appear after the first account snapshot</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# SIGNALS & TRADES
# =============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown(f'<div class="section-title">Recent Signals</div>', unsafe_allow_html=True)
    df_signals = load_recent_signals(15)

    if df_signals.empty:
        st.markdown(f"""
        <div class="card">
            <div class="empty-state">
                <div class="empty-state-title">No signals yet</div>
                <div class="empty-state-subtitle">Signals will appear when regime conditions align</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        table_html = f"""
        <div class="card" style="padding: 0; overflow: hidden;">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Instrument</th>
                    <th>Direction</th>
                    <th style="text-align: right;">Entry</th>
                    <th style="text-align: center;">Exec</th>
                </tr>
            </thead>
            <tbody>
        """
        for _, row in df_signals.head(8).iterrows():
            dir_color = GREEN if row['direction'] == 'LONG' else RED
            dir_symbol = "↑" if row['direction'] == 'LONG' else "↓"
            exec_color = GREEN if row['executed'] == 1 else TEXT_DIM
            exec_text = "Yes" if row['executed'] == 1 else "No"
            time_str = row['timestamp'].strftime('%m-%d %H:%M')
            table_html += f"""
                <tr>
                    <td>{time_str}</td>
                    <td style="font-weight: 600;">{row['instrument']}</td>
                    <td style="color: {dir_color};">{dir_symbol} {row['direction']}</td>
                    <td style="text-align: right;">{row['entry_price']:.5f}</td>
                    <td style="text-align: center; color: {exec_color};">{exec_text}</td>
                </tr>
            """
        table_html += "</tbody></table></div>"
        st.markdown(table_html, unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="section-title">Recent Trades</div>', unsafe_allow_html=True)
    df_trades = load_recent_trades(15)

    if df_trades.empty:
        st.markdown(f"""
        <div class="card">
            <div class="empty-state">
                <div class="empty-state-title">No closed trades yet</div>
                <div class="empty-state-subtitle">Trade history will appear here</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        table_html = f"""
        <div class="card" style="padding: 0; overflow: hidden;">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Instrument</th>
                    <th>Dir</th>
                    <th style="text-align: right;">P&L</th>
                    <th style="text-align: right;">R</th>
                </tr>
            </thead>
            <tbody>
        """
        for _, row in df_trades.head(8).iterrows():
            dir_color = GREEN if row['direction'] == 'LONG' else RED
            dir_symbol = "↑" if row['direction'] == 'LONG' else "↓"
            pnl_color = GREEN if row['pnl_currency'] >= 0 else RED
            pnl_sign = "+" if row['pnl_currency'] >= 0 else ""
            r_val = f"{row['r_multiple']:+.2f}R" if pd.notna(row['r_multiple']) else "-"
            r_color = GREEN if pd.notna(row['r_multiple']) and row['r_multiple'] >= 0 else RED
            time_str = row['timestamp'].strftime('%m-%d %H:%M')
            table_html += f"""
                <tr>
                    <td>{time_str}</td>
                    <td style="font-weight: 600;">{row['instrument']}</td>
                    <td style="color: {dir_color};">{dir_symbol}</td>
                    <td style="text-align: right; color: {pnl_color}; font-weight: 600;">{pnl_sign}${row['pnl_currency']:,.2f}</td>
                    <td style="text-align: right; color: {r_color};">{r_val}</td>
                </tr>
            """
        table_html += "</tbody></table></div>"
        st.markdown(table_html, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# INSTRUMENT CHARTS
# =============================================================================

st.markdown(f'<div class="section-title">Instruments</div>', unsafe_allow_html=True)

if instruments:
    tabs = st.tabs(instruments)

    for i, instrument in enumerate(instruments):
        with tabs[i]:
            df_ohlc = load_instrument_ohlc(instrument, limit=100)
            df_entries, df_exits = load_instrument_trades(instrument)

            if df_ohlc is None or df_ohlc.empty:
                st.markdown(f"""
                <div class="card">
                    <div class="empty-state">
                        <div class="empty-state-title">Chart unavailable for {instrument}</div>
                        <div class="empty-state-subtitle">MT5 connection required for live charts</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_ohlc['timestamp'],
                    open=df_ohlc['open'],
                    high=df_ohlc['high'],
                    low=df_ohlc['low'],
                    close=df_ohlc['close'],
                    name=instrument,
                    increasing_line_color=GREEN,
                    decreasing_line_color=RED,
                    increasing_fillcolor=GREEN,
                    decreasing_fillcolor=RED,
                ))

                # Entry markers
                if not df_entries.empty:
                    for _, trade in df_entries.iterrows():
                        marker_color = GREEN if trade['direction'] == 'LONG' else RED
                        marker_symbol = 'triangle-up' if trade['direction'] == 'LONG' else 'triangle-down'
                        fig.add_trace(go.Scatter(
                            x=[trade['timestamp']], y=[trade['entry_price']],
                            mode='markers',
                            marker=dict(symbol=marker_symbol, size=10, color=marker_color),
                            showlegend=False,
                            hovertemplate=f"Entry: {trade['direction']}<br>Price: {trade['entry_price']:.5f}<extra></extra>"
                        ))

                # Exit markers
                if not df_exits.empty:
                    for _, trade in df_exits.iterrows():
                        marker_color = GREEN if trade['pnl_currency'] >= 0 else RED
                        fig.add_trace(go.Scatter(
                            x=[trade['timestamp']], y=[trade['exit_price']],
                            mode='markers',
                            marker=dict(symbol='x', size=8, color=marker_color),
                            showlegend=False,
                            hovertemplate=f"Exit<br>P&L: ${trade['pnl_currency']:+.2f}<extra></extra>"
                        ))

                fig.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=20, b=0),
                    paper_bgcolor='transparent',
                    plot_bgcolor=BG_CARD,
                    font=dict(color=TEXT_MUTED),
                    xaxis=dict(gridcolor=BORDER, rangeslider=dict(visible=False)),
                    yaxis=dict(gridcolor=BORDER, side='right'),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Instrument stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    open_trades = max(0, len(df_entries) - len(df_exits))
                    st.markdown(f"""
                    <div class="card" style="text-align: center;">
                        <div class="metric-label">Open</div>
                        <div class="metric-value" style="color: {CYAN};">{open_trades}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    total_pnl = df_exits['pnl_currency'].sum() if not df_exits.empty else 0
                    pnl_color = GREEN if total_pnl >= 0 else RED
                    st.markdown(f"""
                    <div class="card" style="text-align: center;">
                        <div class="metric-label">P&L</div>
                        <div class="metric-value" style="color: {pnl_color};">${total_pnl:+,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    wins = len(df_exits[df_exits['pnl_currency'] > 0]) if not df_exits.empty else 0
                    total = len(df_exits) if not df_exits.empty else 0
                    wr = (wins / total * 100) if total > 0 else 0
                    st.markdown(f"""
                    <div class="card" style="text-align: center;">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{wr:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# SYSTEM EVENTS & CONSOLE
# =============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown(f'<div class="section-title">System Events</div>', unsafe_allow_html=True)
    df_events = load_recent_events(20)

    if df_events.empty:
        st.markdown(f"""
        <div class="card">
            <div class="empty-state">
                <div class="empty-state-title">No events yet</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_display = df_events.head(10).copy()
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%m-%d %H:%M')
        df_display['message'] = df_display['message'].str[:60]
        df_display = df_display[['timestamp', 'level', 'message']]
        df_display.columns = ['Time', 'Level', 'Message']
        st.dataframe(df_display, use_container_width=True, hide_index=True, height=300)

with col2:
    st.markdown(f'<div class="section-title">Console Output</div>', unsafe_allow_html=True)
    console_lines = load_console_output(lines=30)
    console_text = ''.join(console_lines[-20:])
    st.code(console_text, language=None)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown(f"""
<div style="margin-top: 40px; padding-top: 16px; border-top: 1px solid {BORDER};
            text-align: center; color: {TEXT_DIM}; font-size: 0.75rem;">
    {PRODUCT_NAME} {STRATEGY_NAME} · v{VERSION} · Build {BUILD_NAME} ·
    Last refresh: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
