import pandas as pd
import numpy as np


# ------------------------------------------------------------
# LAGGED RETURNS (Your X1 style)
# ------------------------------------------------------------

def add_lagged_returns(
    df: pd.DataFrame,
    price_column: str = "Close",
    lagged_return_periods: list = []
):
    X = pd.concat(
        [
            np.log(df[price_column]).diff(i)
            for i in lagged_return_periods
        ],
        axis=1
    )

    X.columns = [f"{i}DT" for i in lagged_return_periods]

    return X

# ------------------------------------------------------------
# VOLATILITY FEATURES
# ------------------------------------------------------------


def add_vol_features(
    df: pd.DataFrame,
    price_column: str = "Close",
    vol_periods: list = []
):
    X = pd.concat(
        [
            df[price_column].rolling(i).std()
            for i in vol_periods
        ],
        axis=1
    )
    X.columns = [f"VOL{i}" for i in vol_periods]
    return X



# -----------------------------------------------------------
# Trend Features (SMA, EMA)
# -----------------------------------------------------------

def add_trend_features(
    df: pd.DataFrame,
    sma_periods: list = [],
    ema_periods: list = [],
    price_column: str = "Close"
):
    df = df.copy()

    # ---- SMA ----
    if len(sma_periods) > 0:
        SMA = pd.concat(
            [
                df[price_column]
                .rolling(i)
                .mean()
                for i in sma_periods
            ],
            axis=1
        )
        SMA.columns = [f"SMA{i}" for i in sma_periods]
    else:
        SMA = pd.DataFrame(index=df.index)

    # ---- EMA ----
    if len(ema_periods) > 0:
        EMA = pd.concat(
            [
                df[price_column]
                .ewm(span=i, adjust=False)
                .mean()
                for i in ema_periods
            ],
            axis=1
        )
        EMA.columns = [f"EMA{i}" for i in ema_periods]
    else:
        EMA = pd.DataFrame(index=df.index)

    return pd.concat([SMA, EMA], axis=1)

# ------------------------------------------------------------
# RSI 
# ------------------------------------------------------------

def compute_rsi_strategy(close_df, window, price_col='Close'):
    df = close_df.copy().reset_index()

    df["passive_returns"] = np.log(
        df[price_col] / df[price_col].shift(1)
    )

    df['gain_or_loss'] = np.sign(df['passive_returns'])
    df['gain'] = np.where(df['gain_or_loss'] == 1, df['passive_returns'], 0)
    df['loss'] = np.where(df['gain_or_loss'] == -1, -df['passive_returns'], 0)

    df['avg_gain'] = df['gain'].rolling(window).mean()
    df['avg_loss'] = df['loss'].rolling(window).mean()

    df['RSI'] = 100 - 100 / (1 + df['avg_gain'] / df['avg_loss'])
    df = df.set_index('Date')

    return df['RSI']

def add_rsi(df, rsi_periods: list = [], price_column='Close'):
    X = pd.concat(
        [
            compute_rsi_strategy(
                df[price_column],
                window=i,
                price_col=price_column
            )
            for i in rsi_periods
        ],
        axis=1
    )
    X.columns = [f"RSI{i}" for i in rsi_periods]
    return X


# ------------------------------------------------------------
# STOCHASTICS (%K and %D)
# ------------------------------------------------------------

def STOK(close, low, high, n):
    return ((close - low.rolling(n).min()) /
            (high.rolling(n).max() - low.rolling(n).min())) * 100


def STOD(close, low, high, n):
    return STOK(close, low, high, n).rolling(3).mean()


def add_stochastics(df, stok_periods: list = [], stod_periods: list = []):
    close, low, high = df["Close"], df["Low"], df["High"]

    Xk = pd.concat(
        [STOK(close, low, high, n) for n in stok_periods],
        axis=1
    )
    Xk.columns = [f"%K{p}" for p in stok_periods]

    Xd = pd.concat(
        [STOD(close, low, high, n) for n in stod_periods],
        axis=1
    )
    Xd.columns = [f"%D{p}" for p in stod_periods]

    return pd.concat([Xk, Xd], axis=1)


# ------------------------------------------------------------
# 6. RATE OF CHANGE (ROC)
# ------------------------------------------------------------

def ROC(series, n):
    M = series.diff(n - 1)
    N = series.shift(n - 1)
    return pd.Series((M / N) * 100, name=f"ROC{n}")


def add_roc(df, roc_periods: list = [], price_column='Close'):
    X = pd.concat(
        [
            ROC(df[price_column], n)
            for n in roc_periods
        ],
        axis=1
    )
    X.columns = [f"ROC{i}" for i in roc_periods]
    return X

# ------------------------------------------------------------
# Average True Range
# ------------------------------------------------------------

def compute_ATR(df: pd.DataFrame, window: int = 14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR
    atr = true_range.rolling(window).mean()
    atr.name = f"ATR{window}"

    return atr


def add_atr(df: pd.DataFrame, atr_periods: list = []):
    X = pd.concat(
        [
            compute_ATR(df, window=p)
            for p in atr_periods
        ],
        axis=1
    )
    X.columns = [f"ATR{p}" for p in atr_periods]
    return X


# -----------------------------------------------------------
# Build Final Feature Matrix X
# -----------------------------------------------------------

def create_all_features(
    df: pd.DataFrame,
    price_column: str = "Close",
    lagged_return_periods: list = [],
    vol_periods: list = [],
    sma_periods: list = [],
    ema_periods: list = [],
    rsi_periods: list = [],
    stok_periods: list = [],
    stod_periods: list = [],
    roc_periods: list = [],
    atr_periods: list = []
):

    features = []

    # Lags
    features.append(
        add_lagged_returns(df, price_column, lagged_return_periods)
    )

    # Volatility
    features.append(
        add_vol_features(df, price_column, vol_periods)
    )

    # Trend
    features.append(
        add_trend_features(df, sma_periods, ema_periods, price_column)
    )

    # RSI
    features.append(
        add_rsi(df, rsi_periods, price_column)
    )

    # Stochastics
    features.append(
        add_stochastics(df, stok_periods, stod_periods)
    )

    # ROC
    features.append(
        add_roc(df, roc_periods, price_column)
    )

    features.append(
        add_atr(df, atr_periods)
    )

    feature_df = pd.concat(features, axis=1)

    return feature_df