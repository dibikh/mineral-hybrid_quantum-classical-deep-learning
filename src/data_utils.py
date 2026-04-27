# src/data_utils.py
# =========================================================
# Multivariate data utilities
# Inputs: Open, High, Low, Close + engineered features
# Split: train / val / test (chronological)
# =========================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ------------------------------------------------------------------
# 1) Robust CSV loader  (handles UTF-8/UTF-16, OHLC or Date/Close)
# ------------------------------------------------------------------

def load_market_csv(path: str) -> pd.DataFrame:
    """
    Load a price CSV robustly.

    Supports:
      - Standard multi-column CSV (Symbol, Date, Open, High, Low, Close …)
      - Single-column CSV where all fields are pasted into one cell
      - UTF-8, UTF-16, UTF-8-sig, latin1 encodings
    Returns a clean DataFrame with columns: Date, Open, High, Low, Close
    (and optionally Symbol), sorted by Date.
    """
    raw = None
    last_err = None
    for enc in ["utf-16", "utf-8-sig", "utf-8", "latin1"]:
        try:
            raw = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            last_err = e

    if raw is None:
        raise RuntimeError(f"Could not read CSV. Last error: {last_err}")

    raw.columns = [str(c).strip() for c in raw.columns]

    # --- single-column case ---
    if raw.shape[1] == 1:
        only_col = raw.columns[0]
        lines = [only_col] + raw[only_col].astype(str).tolist()
        lines = [ln for ln in lines if str(ln).strip() != ""]
        rows = [str(ln).split(",") for ln in lines]
        max_len = max(len(r) for r in rows)
        rows = [r + [""] * (max_len - len(r)) for r in rows]
        temp = pd.DataFrame(rows)
        header_candidate = [str(x).strip() for x in temp.iloc[0].tolist()]
        header_lower = [x.lower() for x in header_candidate]
        needed = {"date", "open", "high", "low", "close"}
        if needed.issubset(set(header_lower)):
            temp.columns = header_candidate
            df = temp.iloc[1:].reset_index(drop=True)
        else:
            default_cols = ["Symbol", "Date", "Open", "High", "Low", "Close"]
            if max_len > len(default_cols):
                default_cols += [f"Extra_{i}" for i in range(max_len - len(default_cols))]
            temp.columns = default_cols[:max_len]
            df = temp.copy()
    else:
        df = raw.copy()

    # --- standardise column names ---
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "symbol":   rename_map[c] = "Symbol"
        elif cl == "date":   rename_map[c] = "Date"
        elif cl == "open":   rename_map[c] = "Open"
        elif cl == "high":   rename_map[c] = "High"
        elif cl == "low":    rename_map[c] = "Low"
        elif cl == "close":  rename_map[c] = "Close"
    df = df.rename(columns=rename_map)

    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_cols = ["Date", "Open", "High", "Low", "Close"]
    if "Symbol" in df.columns:
        keep_cols = ["Symbol"] + keep_cols
    df = df[keep_cols].copy()

    # remove duplicated header rows that sometimes appear inside data
    df = df[df["Date"].astype(str).str.lower() != "date"].copy()

    # parse types
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(subset=["Date", "Open", "High", "Low", "Close"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ------------------------------------------------------------------
# 2) Feature engineering
# ------------------------------------------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical / ratio features to an OHLC DataFrame."""
    df_ = df.copy()
    df_["HL_range"]   = (df_["High"] - df_["Low"]) / (df_["Close"].abs() + 1e-8)
    df_["OC_change"]  = (df_["Close"] - df_["Open"]) / (df_["Open"].abs() + 1e-8)
    df_["log_return"] = np.log(df_["Close"] / df_["Close"].shift(1).replace(0, np.nan))
    df_["close_diff"] = df_["Close"].diff()
    df_["ma_5"]       = df_["Close"].rolling(5).mean()
    df_["std_5"]      = df_["Close"].rolling(5).std()

    df_.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_.fillna(method="bfill", inplace=True)
    df_.fillna(method="ffill", inplace=True)
    df_.fillna(0.0, inplace=True)
    return df_


# ------------------------------------------------------------------
# 3) Scaling  (fit on TRAIN rows only)
# ------------------------------------------------------------------

def fit_scalers(df: pd.DataFrame, train_end_row: int,
                feature_cols: list, target_col: str = "Close"):
    """
    Fit MinMaxScaler on training rows only.
    Returns (x_scaler, y_scaler, X_all_scaled, y_all_scaled).
    """
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(df.loc[:train_end_row - 1, feature_cols])
    y_scaler.fit(df.loc[:train_end_row - 1, [target_col]])
    X_all_scaled = x_scaler.transform(df[feature_cols])
    y_all_scaled = y_scaler.transform(df[[target_col]])
    return x_scaler, y_scaler, X_all_scaled, y_all_scaled


# ------------------------------------------------------------------
# 4) Sequence creation (multivariate)
# ------------------------------------------------------------------

def create_multivariate_sequences(X_data: np.ndarray, y_data: np.ndarray,
                                   dates: pd.Series, window: int):
    """
    Build overlapping windows from scaled arrays.

    Returns
    -------
    X_seq        : (N, window, n_features)
    y_seq        : (N, 1)
    y_dates      : (N,)  – date of the target step
    target_indices: (N,) – row index of the target in the original DataFrame
    """
    X_seq, y_seq, y_dates, target_indices = [], [], [], []
    for i in range(window, len(X_data)):
        X_seq.append(X_data[i - window:i, :])
        y_seq.append(y_data[i, 0])
        y_dates.append(dates.iloc[i])
        target_indices.append(i)
    return (
        np.array(X_seq,          dtype=np.float32),
        np.array(y_seq,          dtype=np.float32).reshape(-1, 1),
        np.array(y_dates),
        np.array(target_indices)
    )


# ------------------------------------------------------------------
# 5) Chronological 3-way split
# ------------------------------------------------------------------

def chronological_split(X: np.ndarray, y: np.ndarray,
                         y_dates: np.ndarray, target_indices: np.ndarray,
                         train_end_row: int, val_end_row: int):
    """Split sequences into train / val / test by the target row index."""
    train_mask = target_indices < train_end_row
    val_mask   = (target_indices >= train_end_row) & (target_indices < val_end_row)
    test_mask  = target_indices >= val_end_row

    return (
        X[train_mask], y[train_mask], y_dates[train_mask],
        X[val_mask],   y[val_mask],   y_dates[val_mask],
        X[test_mask],  y[test_mask],  y_dates[test_mask],
    )


# ------------------------------------------------------------------
# 6) Legacy helpers kept for backward-compatibility
# ------------------------------------------------------------------

def load_series(csv_path: str) -> pd.DataFrame:
    """Legacy loader: returns a DataFrame with at least Date and Close."""
    try:
        df = load_market_csv(csv_path)
    except Exception:
        # fall back to simple 2-col CSV
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(columns={
            cols.get("date",  "Date"):  "Date",
            cols.get("close", "Close"): "Close"
        })
        df["Date"]  = pd.to_datetime(df["Date"],  errors="coerce", dayfirst=True)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return df


def make_sequences(values_scaled: np.ndarray, window: int):
    """Legacy univariate sequence builder."""
    X, y = [], []
    for i in range(window, len(values_scaled)):
        X.append(values_scaled[i - window:i, 0])
        y.append(values_scaled[i, 0])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def train_test_split_seq(X, y, train_frac: float = 0.7):
    split = int(len(X) * train_frac)
    return X[:split], X[split:], y[:split], y[split:], split


def scale_close(df: pd.DataFrame):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close"]].values)
    return scaler, scaled


def test_dates(df: pd.DataFrame, window: int, split: int):
    start_idx = window + split
    return df["Date"].iloc[start_idx:].reset_index(drop=True).values
