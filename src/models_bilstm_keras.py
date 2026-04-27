# src/models_bilstm_keras.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input


def build_bilstm(window: int, n_features: int = 1, units: int = 64):
    """
    Multivariate Bidirectional LSTM.

    Parameters
    ----------
    window     : look-back window length
    n_features : number of input features per time step (default 1 for univariate)
    units      : number of units in each LSTM layer
    """
    model = Sequential([
        Input(shape=(window, n_features)),
        Bidirectional(LSTM(units, return_sequences=True)),
        Bidirectional(LSTM(units)),
        Dense(1)
    ])
    return model
