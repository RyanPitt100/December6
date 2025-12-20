import os
import pandas as pd


def load_ohlc(path: str, time_col: str = "timestamp") -> pd.DataFrame:
    """Load OHLC CSV file with timestamp index."""
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col).set_index(time_col)
    return df


def load_ohlc_15m(instrument: str, data_root: str = "./") -> pd.DataFrame:
    """
    Load 15m OHLC data for an instrument.
    Expects path: {data_root}/15m/{instrument}.csv with a 'timestamp' column.
    """
    path = os.path.join(data_root, "15m", f"{instrument}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing 15m data for {instrument}: {path}")

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} must contain a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Make sure required OHLC columns exist
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing '{col}' column")

    return df


def load_regime_tf(instrument: str, tf: str, data_root: str = "./") -> pd.DataFrame:
    """
    Load labelled regime CSV for a given timeframe.
    Expected:
        {data_root}/1h/{instrument}_labelled.csv
        {data_root}/4h/{instrument}_labelled.csv
        {data_root}/1d/{instrument}_labelled.csv
    """
    filename = f"{instrument}_labelled.csv"
    path = os.path.join(data_root, tf, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing labelled regime file for {instrument} {tf}: {path}")

    df = pd.read_csv(path)

    if "timestamp" not in df.columns or "regime" not in df.columns:
        raise ValueError(f"{path} must contain 'timestamp' and 'regime' columns")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    cols = ["regime"]
    if "range_score" in df.columns:
        cols.append("range_score")

    return df[cols]


def build_multi_tf_frame(
    instrument: str,
    data_root: str = "./",
    m15_dir: str = "15m",
    h1_dir: str = "1h",
    h4_dir: str = "4h",
    d1_dir: str = "1d",
) -> pd.DataFrame:
    """
    Build a 15m frame with H1/H4/D1 regime labels merged in.

    Args:
        instrument: Symbol name (e.g., "EURUSD", "US30")
        data_root: Root directory containing data folders
        m15_dir: Subdirectory for 15m data (default: "15m")
        h1_dir: Subdirectory for 1h labelled data (default: "1h")
        h4_dir: Subdirectory for 4h labelled data (default: "4h")
        d1_dir: Subdirectory for 1d labelled data (default: "1d")

    Returns:
        DataFrame with columns:
        - open, high, low, close (from 15m)
        - regime_h1, range_score_h1
        - regime_h4
        - regime_d1
    """
    # Base 15m
    df_m15 = load_ohlc_15m(instrument, data_root=data_root)
    df_m15 = df_m15.copy()
    df_m15.reset_index(inplace=True)  # keep timestamp column
    df_m15 = df_m15.sort_values("timestamp")

    # H1 regimes
    h1 = load_regime_tf(instrument, h1_dir, data_root=data_root)
    h1 = h1.reset_index().sort_values("timestamp")
    h1 = h1.rename(
        columns={
            "regime": "regime_h1",
            "range_score": "range_score_h1",
        }
    )

    # H4 regimes
    h4 = load_regime_tf(instrument, h4_dir, data_root=data_root)
    h4 = h4.reset_index().sort_values("timestamp")
    h4 = h4.rename(columns={"regime": "regime_h4"})

    # D1 regimes
    d1 = load_regime_tf(instrument, d1_dir, data_root=data_root)
    d1 = d1.reset_index().sort_values("timestamp")
    d1 = d1.rename(columns={"regime": "regime_d1"})

    # As-of merge: each 15m bar gets the latest regime from each TF
    df = pd.merge_asof(
        df_m15.sort_values("timestamp"),
        h1.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    df = pd.merge_asof(
        df.sort_values("timestamp"),
        h4.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    df = pd.merge_asof(
        df.sort_values("timestamp"),
        d1.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    # Set timestamp index again
    df = df.set_index("timestamp").sort_index()
    return df


# Legacy low-level function - kept for backward compatibility if needed
def _build_multi_tf_from_paths(
    m15_path: str,
    h1_path: str,
    h4_path: str,
    d1_path: str,
) -> pd.DataFrame:
    """
    DEPRECATED: Low-level function that takes explicit file paths.
    Use build_multi_tf_frame(instrument, ...) instead.

    Load 15m, 1h, 4h, 1d OHLC/regime files and forward-fill higher-TF
    features down to the 15m index.
    """
    def _suffix_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        df = df.copy()
        drop_cols = [c for c in ["open", "high", "low", "close", "volume", "timestamp"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        rename_map = {c: f"{c}_{suffix}" for c in df.columns}
        return df.rename(columns=rename_map)

    df_m15 = load_ohlc(m15_path)
    df_h1 = load_ohlc(h1_path)
    df_h4 = load_ohlc(h4_path)
    df_d1 = load_ohlc(d1_path)

    df_h1_s = _suffix_cols(df_h1, "h1")
    df_h4_s = _suffix_cols(df_h4, "h4")
    df_d1_s = _suffix_cols(df_d1, "d1")

    # Align to 15m index with forward-fill
    df_h1_ff = df_h1_s.reindex(df_m15.index, method="ffill")
    df_h4_ff = df_h4_s.reindex(df_m15.index, method="ffill")
    df_d1_ff = df_d1_s.reindex(df_m15.index, method="ffill")

    # Join all features onto 15m OHLC
    df_all = df_m15.copy()
    df_all = df_all.join(df_h1_ff, how="left")
    df_all = df_all.join(df_h4_ff, how="left")
    df_all = df_all.join(df_d1_ff, how="left")

    return df_all
