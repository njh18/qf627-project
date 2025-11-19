import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# =====================================================================
#  PART A — ATR + VOLATILITY FEATURES
# =====================================================================
def compute_ATR(
    df: pd.DataFrame,
    window: int = 14,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close"
) -> pd.Series:
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    atr.name = f"ATR{window}"
    return atr


def build_regime_feature_df(
    df: pd.DataFrame,
    price_col: str = "Close",
    atr_window: int = 14,
    vol_windows: list[int] = [20, 60]
) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    log_ret = np.log(df[price_col] / df[price_col].shift(1))

    # Realized rolling volatility windows
    for w in vol_windows:
        features[f"VOL{w}"] = log_ret.rolling(w).std()

    # ATR
    atr = compute_ATR(df, window=atr_window)
    features[atr.name] = atr

    # Normalised ATR
    features[f"{atr.name}_norm"] = atr / df[price_col]

    return features.dropna()
    

# =====================================================================
#  PART B — CLUSTERING PIPELINE (TRAIN + TEST)
# =====================================================================

def compute_regime_clusters(
    df_train: pd.DataFrame,
    n_clusters: int | None = None,
    atr_window: int = 14,
    vol_windows: list[int] = [20, 60],
    plot_silhouette: bool = True
):
    """
    Step 1 — Build volatility features
    Step 2 — Determine ideal cluster count (if n_clusters=None)
    Step 3 — Fit StandardScaler + KMeans on TRAIN set only

    Returns:
        scaler, kmeans, train_regimes, regime_feature_df
    """

    # --------- Build features ---------
    train_feat = build_regime_feature_df(
        df_train,
        price_col="Close",
        atr_window=atr_window,
        vol_windows=vol_windows
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train_feat),
        columns=train_feat.columns,
        index=train_feat.index
    )

    # --------- Determine optimal K (silhouette) ---------
    if n_clusters is None:
        sil = []
        K_range = range(2, 10)

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=627, n_init=10)
            km.fit(X_train)
            sil.append(silhouette_score(X_train, km.labels_))

        if plot_silhouette:
            plt.figure(figsize=(10, 5))
            plt.plot(K_range, sil)
            plt.title("Silhouette Scores for K")
            plt.xticks([i for i in K_range], 
                    rotation=75)
            plt.xlabel("k")
            plt.ylabel("Silhouette Score")
            plt.grid(True)
            plt.show()

        # best K
        n_clusters = K_range[int(np.argmax(sil))]

    # --------- Final KMeans ---------
    kmeans = KMeans(n_clusters=n_clusters, random_state=2025, n_init=10)
    kmeans.fit(X_train)

    # Assign train regimes
    train_regimes = kmeans.predict(X_train)

    # Return everything needed for test later
    return scaler, kmeans, train_regimes, train_feat



def assign_regimes(
    df: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans,
    atr_window: int = 14,
    vol_windows: list[int] = [20, 60],
    price_col: str = "Close"
):

    feat = build_regime_feature_df(
        df,
        price_col=price_col,
        atr_window=atr_window,
        vol_windows=vol_windows
    )

    X = pd.DataFrame(
        scaler.fit_transform(feat),
        columns=feat.columns,
        index=feat.index
    )

    regimes = kmeans.predict(X)

    # Inject into dataframe
    df_result = df.copy()
    df_result.loc[X.index, "regime"] = regimes

    return df_result.dropna()
