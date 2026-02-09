from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

def build_bilstm(window: int, units: int = 64):
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True), input_shape=(window, 1)),
        Bidirectional(LSTM(units)),
        Dense(1)
    ])
    return model
