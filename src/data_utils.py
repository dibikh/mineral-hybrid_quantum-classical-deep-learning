import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_series(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns or "Close" not in df.columns:
        # allow case-insensitive
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(columns={cols.get("date","Date"):"Date", cols.get("close","Close"):"Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return df

def make_sequences(values_scaled: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(values_scaled)):
        X.append(values_scaled[i-window:i, 0])
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
    # Targets correspond to df indices [window, ..., N-1]; test starts at sample index=split
    start_idx = window + split
    return df["Date"].iloc[start_idx:].reset_index(drop=True).values
