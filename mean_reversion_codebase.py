
import os   
import pandas as pd
import yfinance as yf
import datetime as dt
from IPython.display import display
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import datetime as dt
from datetime import datetime, timedelta
import yfinance as yf
from IPython.display import display
from lets_plot import *

from IPython.display import display
import pandas_datareader.data as web
from pandas_datareader import data as pdr
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA #non-linear model, alternatively, t-stochastic neighborhood embedding, UMAP

from sklearn.decomposition import TruncatedSVD

from numpy.linalg import inv, eig, svd

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from pandas.plotting import scatter_matrix
from lets_plot import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import cluster


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)


# if nclust == 2:
#     regime_names = {0: 'Low Vol', 1: 'High Vol'}
# elif nclust == 3:
#     regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
# elif nclust == 4:
#     regime_names = {0: 'Very Low Vol', 1: 'Low Vol', 2: 'Med Vol', 3: 'High Vol'}
# elif nclust == 5:
#     regime_names = {
#         0: 'Very Low Vol (Calm)',
#         1: 'Low Vol (Stable)', 
#         2: 'Medium Vol (Normal)',
#         3: 'High Vol (Elevated)',
#         4: 'Very High Vol (Crisis)'
#     }
# else:
#     regime_names = {i: f'Regime {i}' for i in range(nclust)}
    
    
    
    
def mr_get_spy_raw(cache_file="spy_raw_cache.csv"):
    """Load SPY raw data from cache or download and cache it."""
    if os.path.exists(cache_file):
        spy_raw = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        spy_raw = (
            yf
            .download(
                ["SPY"],
                start=dt.datetime(2006, 11, 1),
                end=dt.datetime(2025, 11, 13),
                auto_adjust=True
            )
        )
        
        # Ensure index is datetime and is one-level
        spy_raw.index = pd.to_datetime(spy_raw.index)

        # Convert to 1-level columns if necessary
        if spy_raw.columns.nlevels > 1:
            spy_raw.columns = spy_raw.columns.droplevel(1)
        # print("columns: ", spy_raw.columns.nlevels)

        spy_raw.to_csv(cache_file)
    
    spy = spy_raw.copy()
    spy["Adj_Close"] = spy["Close"]
    train_size = int(len(spy) * 0.75)
    spy_train = spy.iloc[:train_size].copy()    
    spy_test = spy.iloc[train_size:].copy()

    return spy, spy_train, spy_test




def save_dataframe_to_csv(df, filename):
    try:
        df.to_csv(filename)
        # print(f"DataFrame saved to {filename}")
    except Exception as e:
        pass
        # print(f"Failed to save to {filename}: {e}")



# %%
def mr_add_clustering_features(df):
    """
    Add features with MULTIPLE volatility timeframes for exploration.
    This helps us understand which timeframes matter most.
    Includes both simple and log returns for comparison.
    """
    df = df.copy()
    
    # Returns (both types for comparison)
    df['returns_simple'] = df['Adj_Close'].pct_change()  # Simple: (P_t - P_{t-1}) / P_{t-1}
    df['returns_log'] = np.log(df['Adj_Close'] / df['Adj_Close'].shift(1))  # Log: ln(P_t / P_{t-1})
    
    # Use log returns for volatility (more theoretically sound)
    df['returns'] = df['returns_log']
    
    # Multiple volatility timeframes (short to long)
    volatility_windows = [3, 5, 10, 20, 30, 60, 90]
    for window in volatility_windows:
        df[f'volatility_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252) * 100
    
    # ATR with multiple windows
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    for window in [3, 5, 7, 14, 21, 30]:  # Multiple ATR windows
        df[f'ATR_{window}'] = true_range.rolling(window).mean()
        df[f'ATR_{window}_pct'] = (df[f'ATR_{window}'] / df['Close']) * 100
    
    # Volume features (multiple windows)
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close'] * 100  # Calculate range first
    
    volume_windows = [3, 5, 10, 20, 30, 60]
    for window in volume_windows:
        df[f'volume_ma{window}'] = df['Volume'].rolling(window).mean()
        df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_ma{window}']
        df[f'volume_std_{window}'] = df['Volume'].rolling(window).std()
    
    # Create alias for backward compatibility (default to 20-day)
    df['volume_ratio'] = df['volume_ratio_20']  # Used in feature_cols below
    
    # Price range features (multiple windows)
    range_windows = [3, 5, 10, 20, 30, 60]
    for window in range_windows:
        df[f'range_ma{window}'] = df['high_low_range'].rolling(window).mean()
        df[f'range_std{window}'] = df['high_low_range'].rolling(window).std()
    
    return df





def mr_atr_volatility_kmeans_clustering(spy):
        # Split into train/test (75/25)
    total_rows = len(spy)
    train_size = int(total_rows * 0.75)
    spy_train = spy.iloc[:train_size].copy()
    spy_test = spy.iloc[train_size:].copy()

    # print(f"Training set: {len(spy_train)} rows ({spy_train.index[0]} to {spy_train.index[-1]})")
    # print(f"Test set: {len(spy_test)} rows ({spy_test.index[0]} to {spy_test.index[-1]})")

    # Add features
    spy_train_features = mr_add_clustering_features(spy_train)
    # print("\n" + "="*70)
    # print("FEATURES ADDED FOR CLUSTERING")
    # print("="*70)
    feature_cols = ['volatility_20', 'volatility_60', 'ATR_14', 'ATR_14_pct',
                    'volume_ratio', 'high_low_range', 'range_ma20']
    # print("\nFeature Summary Statistics:")

    # Show all volatility windows
    # print("\nAll Volatility Windows:")
    vol_cols = [col for col in spy_train_features.columns if col.startswith('volatility_')]

    # Show all ATR windows
    # print("\nAll ATR Windows (%):")
    atr_cols = [col for col in spy_train_features.columns if col.startswith('ATR_') and col.endswith('_pct')]

    # Show all volume ratio windows
    # print("\nAll Volume Ratio Windows:")
    vol_ratio_cols = [col for col in spy_train_features.columns if col.startswith('volume_ratio_')]

    # Show all range MA windows
    # print("\nAll Range Moving Average Windows:")
    range_cols = [col for col in spy_train_features.columns if col.startswith('range_ma') and col != 'range_ma20']
    range_cols = ['range_ma5', 'range_ma10', 'range_ma20', 'range_ma30', 'range_ma60']

    # Compare simple vs log returns
    # print("\n" + "="*70)
    # print("SIMPLE vs LOG RETURNS COMPARISON")
    # print("="*70)
    comparison = pd.DataFrame({
        'Simple Returns': spy_train_features['returns_simple'].describe(),
        'Log Returns': spy_train_features['returns_log'].describe(),
        'Difference': spy_train_features['returns_simple'].describe() - spy_train_features['returns_log'].describe()
    })
    # print("\nKey Observations:")
    # print("• Differences are very small for daily returns (~0.01%)")
    # print("• Log returns are more symmetric (mean closer to median)")
    # print("• Using log returns for volatility calculations (theoretically correct)")
    # print("="*70)
    # ============================================================
    # K-MEANS CLUSTERING FOR SPY VOLATILITY REGIMES
    # Following Professor's Methodology
    # ============================================================

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    from sklearn import cluster

    # print("\n" + "="*70)
    # print("STEP 1: PREPARE FEATURES FOR CLUSTERING")
    # print("="*70)

    # Select features for clustering (4 key features like professor used Returns + Volatility)
    feature_cols = ['volatility_20', 'ATR_14_pct', 'volume_ratio_20', 'range_ma20']
    feature_cols = ['volatility_5', 'ATR_5_pct', 'volume_ratio_5', 'range_ma5']

    X = spy_train_features[feature_cols].dropna()

    # print(f"\nClustering {len(X)} trading days")
    # print(f"Date range: {X.index[0]} to {X.index[-1]}")
    # print(f"Features: {feature_cols}")

    # Standardize features (critical for K-means!)
    # print("\n" + "="*70)
    # print("STEP 2: STANDARDIZE FEATURES")
    # print("="*70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    # print("\nBefore Scaling:")

    # print("\nAfter Scaling (mean=0, std=1):")


    # ============================================================
    # ELBOW METHOD - Find optimal k (Professor's approach)
    # ============================================================

    # print("\n" + "="*70)
    # print("STEP 3: ELBOW METHOD - FINDING OPTIMAL K")
    # print("="*70)

    distortions = []  # Inertia values
    max_loop = 10

    for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k, random_state=627, n_init=10)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)
        
    fig = plt.figure(figsize=(16, 8))
    plt.plot(range(2, max_loop), distortions, 'bo-', linewidth=2, markersize=8)
    plt.xticks([i for i in range(2, max_loop)], rotation=0)
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=14)
    plt.title('Elbow Method: Finding Optimal k for SPY Volatility Regimes', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()

    # ============================================================
    # SILHOUETTE SCORE - Validate cluster quality (Professor's approach)
    # ============================================================

    # print("\n" + "="*70)
    # print("STEP 4: SILHOUETTE ANALYSIS")
    # print("="*70)

    silhouette_scores = []

    for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k, random_state=627, n_init=10)
        kmeans.fit(X_scaled)
        score = metrics.silhouette_score(X_scaled, kmeans.labels_, random_state=627)
        silhouette_scores.append(score)
        # print(f"k={k}: Silhouette Score = {score:.4f}")
            
    fig = plt.figure(figsize=(16, 8))
    plt.plot(range(2, max_loop), silhouette_scores, 'ro-', linewidth=2, markersize=8)
    plt.xticks([i for i in range(2, max_loop)], rotation=0)
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.title('Silhouette Analysis: Cluster Quality for SPY Volatility Regimes', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Good threshold (0.5)')
    plt.axhline(y=0.7, color='darkgreen', linestyle='--', linewidth=2, label='Excellent threshold (0.7)')
    plt.legend(fontsize=12)
    plt.show()

    # Find best k
    best_k = silhouette_scores.index(max(silhouette_scores)) + 2
    # print(f"\n{'='*70}")
    # print(f"RECOMMENDED k = {best_k} (highest silhouette score: {max(silhouette_scores):.4f})")
    # print(f"{'='*70}")

    # ============================================================
    # FIT FINAL K-MEANS MODEL (Professor's approach)
    # ============================================================

    # print("\n" + "="*70)
    # print("STEP 5: FIT FINAL K-MEANS MODEL")
    # print("="*70)

    # ============================================================
    # CHOOSE NUMBER OF CLUSTERS
    # ============================================================
    # Option 1: Use automatic selection (highest silhouette score)
    nclust = best_k  
    nclust = 5
    # Option 2: Override with manual selection (comment out line above, uncomment one below)
    # nclust = 2  # Use this for Low Vol vs High Vol (binary regime)
    # nclust = 3  # Use this for Low/Med/High Vol (most common choice)
    # nclust = 4  # Use this for more granular regimes
    # nclust = 5  # Use this if elbow shows 5 clusters
    # nclust = 6  # Use this if following professor's approach (elbow method)

    # print(f"\nSelected k = {nclust} (automatic selection based on silhouette score)")
    # print(f"To manually override, uncomment desired 'nclust = X' line above")
    # print(f"\nFitting K-Means with k={nclust} clusters...")

    k_means = cluster.KMeans(n_clusters=nclust, random_state=627, n_init=10)
    k_means.fit(X_scaled)

    # Extract labels and centroids
    target_labels = k_means.predict(X_scaled)
    centroids = k_means.cluster_centers_

    # print(f"✓ Clustering complete!")
    # print(f"  Cluster labels: {np.unique(target_labels)}")
    # print(f"  Centroids shape: {centroids.shape}")

    # Add regime labels to dataframe and sort by volatility
    spy_train_features.loc[X.index, 'regime'] = target_labels

    # Sort regimes by volatility (0=Low, 1=Med, 2=High, etc.)
    # Use first feature (volatility) from feature_cols for sorting
    volatility_col = feature_cols[0]  # e.g., 'volatility_5' or 'volatility_20'
    regime_volatility = spy_train_features.groupby('regime')[volatility_col].mean().sort_values()
    regime_mapping = {old: new for new, old in enumerate(regime_volatility.index)}
    spy_train_features['regime'] = spy_train_features['regime'].map(regime_mapping)

    # print(f"\nRegimes sorted by average volatility:")
    # for old_label, new_label in sorted(regime_mapping.items(), key=lambda x: x[1]):
    #     avg_vol = regime_volatility[old_label]
    #     print(f"  Regime {new_label} (was {old_label}): {avg_vol:.2f}% volatility")

    # ============================================================
    # VISUALIZE CLUSTERS IN 2D (Professor's exact approach)
    # Professor plotted (Returns, Volatility) in 2D
    # We plot (Volatility, ATR) in 2D - same methodology!
    # ============================================================

    # print("\n" + "="*70)
    # print("STEP 6: VISUALIZE CLUSTERS (2D SCATTER)")
    # print("="*70)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)

    # Plot using first 2 features (like professor used Returns vs Volatility)
    scatter = ax.scatter(X.iloc[:, 0],      # First feature (e.g., volatility_5)
                        X.iloc[:, 1],      # Second feature (e.g., ATR_5_pct)
                        c=target_labels,   # Color by cluster
                        cmap="rainbow", 
                        s=50, 
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5)

    ax.set_title(f"K-Means Clustering Results: SPY Volatility Regimes (k={nclust})", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel(f"{feature_cols[0]}", fontsize=14)
    ax.set_ylabel(f"{feature_cols[1]}", fontsize=14)

    # Plot centroids (convert back to original scale)
    centroids_original = scaler.inverse_transform(centroids)
    plt.scatter(centroids_original[:, 0],    # First feature centroid
            centroids_original[:, 1],     # Second feature centroid
            marker='*', 
            s=800, 
            c='black',
            edgecolors='white',
            linewidth=3,
            label='Centroids',
            zorder=10)

    plt.colorbar(scatter, label='Cluster Label')
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # print(f"\nCentroid Coordinates (original scale):")
    centroids_df = pd.DataFrame(centroids_original, columns=feature_cols)

    # ============================================================
    # CLUSTER MEMBER COUNTS (Professor's approach)
    # Professor showed "stocks per cluster"
    # We show "days per regime"
    # ============================================================

    # print("\n" + "="*70)
    # print("STEP 7: CLUSTER MEMBER COUNTS")
    # print("="*70)

    clustered_series = pd.Series(index=X.index, data=target_labels)
    clustered_series = clustered_series[clustered_series != -1]  # Remove any outliers

    # Apply regime mapping to ensure sorted by volatility
    clustered_series_sorted = clustered_series.map(regime_mapping)

    # Define regime names
    if nclust == 2:
        regime_names = {0: 'Low Vol', 1: 'High Vol'}
    elif nclust == 3:
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
    else:
        regime_names = {i: f'Regime {i}' for i in range(nclust)}

    plt.figure(figsize=(16, 8))
    counts = clustered_series_sorted.value_counts().sort_index()

    # Color mapping
    color_map = {0: 'green', 1: 'orange', 2: 'red', 3: 'purple', 4: 'brown'}
    colors = [color_map.get(i, 'gray') for i in counts.index]

    bars = plt.barh(counts.index, counts, color=colors)

    plt.title("Regime Distribution: Days per Volatility Regime", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Trading Days", fontsize=14)
    plt.ylabel("Regime", fontsize=14)
    plt.yticks(counts.index, [f"{regime_names.get(i, f'Regime {i}')}" for i in counts.index])

    # Add percentage labels on bars
    for i, (idx, count) in enumerate(counts.items()):
        pct = count / len(clustered_series_sorted) * 100
        plt.text(count + 5, idx, f'{count} days ({pct:.1f}%)', 
                va='center', fontsize=12, fontweight='bold')

    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

    # Print regime distribution
    # print(f"\nRegime Distribution Summary:")
    # for regime_id in sorted(counts.index):
    #     count = counts[regime_id]
    #     pct = count / len(clustered_series_sorted) * 100
    #     print(f"  {regime_names.get(regime_id, f'Regime {regime_id}')}: {count} days ({pct:.1f}%)")

    # ============================================================
    # SUMMARY AND FINAL OUTPUT
    # ============================================================

    # print("\n" + "="*70)
    # print("✅ K-MEANS CLUSTERING COMPLETE!")
    # print("="*70)
    # print(f"\nFinal Results:")
    # print(f"  - Optimal k: {nclust}")
    # print(f"  - Silhouette Score: {max(silhouette_scores):.4f}")
    # print(f"  - Features Used: {feature_cols}")
    # print(f"  - Training Days Clustered: {len(X)}")
    # print(f"\nRegime column added to spy_train_features DataFrame")
    # print("="*70)

    # Display sample of data with regime labels (using actual features used for clustering)
    # print("\nSample of clustered data:")
    display_cols = ['Adj_Close'] + feature_cols + ['regime']

    # Count days in each cluster
    # print("\n" + "="*70)
    # print("CLUSTER DISTRIBUTION")
    # print("="*70)

    regime_counts = spy_train_features['regime'].value_counts().sort_index()
    total_days = regime_counts.sum()

    # Define regime names based on nclust
    if nclust == 2:
        regime_names = {0: 'Low Vol', 1: 'High Vol'}
    elif nclust == 3:
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
    elif nclust == 4:
        regime_names = {0: 'Very Low Vol', 1: 'Low Vol', 2: 'Med Vol', 3: 'High Vol'}
    else:
        regime_names = {i: f'Regime {i}' for i in range(nclust)}

    # print(f"\nTotal Trading Days: {total_days}")
    # print(f"\nDays per Regime:")
    # print("-" * 70)
    # Dynamic column header based on first feature used
    vol_header = f"Avg {volatility_col}"
    # print(f"{'Regime':<20} {'Days':>15} {'Percentage':>15} {vol_header:>15}")
    # print("-" * 70)

    # for regime_id in sorted(regime_counts.index):
    #     count = regime_counts[regime_id]
    #     pct = count / total_days * 100
    #     # Use first feature (volatility) from feature_cols for display
    #     avg_vol = spy_train_features[spy_train_features['regime'] == regime_id][volatility_col].mean()
    #     regime_label = regime_names.get(int(regime_id), f'Regime {int(regime_id)}')
    #     print(f"{regime_label:<20} {count:>15} {pct:>14.1f}% {avg_vol:>14.2f}%")

    # print("-" * 70)
    # print(f"{'TOTAL':<20} {total_days:>15} {100.0:>14.1f}%")
    # print("="*70)

    # ============================================================
    # APPLY TRAINED MODEL TO TEST DATA (NO REFITTING!)
    # ============================================================

    # print("\n" + "="*70)
    # print("APPLYING TRAINED MODEL TO TEST DATA")
    # print("="*70)

    # Step 1: Add features to test data
    spy_test_features = mr_add_clustering_features(spy_test)

    # Step 2: Extract same features as training
    X_test = spy_test_features[feature_cols].dropna()

    # print(f"\nTest Data: {len(X_test)} trading days")
    # print(f"Date range: {X_test.index[0]} to {X_test.index[-1]}")

    # Step 3: Transform test data using TRAINING scaler (no refit!)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Predict clusters using TRAINING model (no refit!)
    test_labels = k_means.predict(X_test_scaled)

    # Step 5: Add regime labels to test dataframe
    spy_test_features.loc[X_test.index, 'regime'] = test_labels

    # Step 6: Apply SAME regime mapping as training (0=Low, 1=High, etc.)
    spy_test_features['regime'] = spy_test_features['regime'].map(regime_mapping)

    # print(f"✓ Test data clustered using trained model (no refitting)")
    # print(f"✓ Regime mapping consistent with training data")
    
    
    return spy_train_features, spy_test_features

def mr_compute_mean_reversion_sma_bb_strategy(df, window=20, position_size=150, initial_cash=100000,
                                   z_threshold_long=-2.0, z_threshold_short=2.0,
                                   position_sizing_method='capital_based', capital_allocation_pct=0.98,
                                   target_regime=None, signal_shift=0):
    df_signals = df.copy()
    # ============================================================
    # 1. Calculate Technical Indicators
    # ============================================================
    df_signals['dynamic_sma'] = df_signals['Adj_Close'].rolling(
        window=window, min_periods=1, center=False
    ).mean()

    df_signals['dynamic_std'] = df_signals['Adj_Close'].rolling(
        window=window, min_periods=1, center=False
    ).std()
    df_signals['dynamic_std'] = df_signals['dynamic_std'].replace(0, np.nan)

    df_signals['z_score'] = (
        (df_signals['Adj_Close'] - df_signals['dynamic_sma']) / df_signals['dynamic_std']
    )
    df_signals['z_score'] = df_signals['z_score'].replace([np.inf, -np.inf], np.nan).fillna(0)
    std_for_bands = df_signals['dynamic_std'].fillna(0)
    
    df_signals['range_upper'] = df_signals['dynamic_sma'] + z_threshold_short * std_for_bands
    df_signals['range_lower'] = df_signals['dynamic_sma'] + z_threshold_long * std_for_bands 
    
    # ============================================================
    # 2. Generate Entry/Exit Signals
    # ============================================================
    # Entry Long: z[t-1] > z_threshold_long AND z[t] <= z_threshold_long (crosses down through threshold)
    df_signals['entry_long'] = (
        (df_signals['z_score'].shift(1) > z_threshold_long) &
        (df_signals['z_score'] <= z_threshold_long)
    )

    # Exit Long: z[t-1] < 0 AND z[t] >= 0 (crosses up through 0)
    df_signals['exit_long'] = (
        (df_signals['z_score'].shift(1) < 0) &
        (df_signals['z_score'] >= 0)
    )

    # Entry Short: z[t-1] < z_threshold_short AND z[t] >= z_threshold_short (crosses up through threshold)
    df_signals['entry_short'] = (
        (df_signals['z_score'].shift(1) < z_threshold_short) &
        (df_signals['z_score'] >= z_threshold_short)
    )

    # Exit Short: z[t-1] > 0 AND z[t] <= 0 (crosses down through 0)
    df_signals['exit_short'] = (
        (df_signals['z_score'].shift(1) > 0) &
        (df_signals['z_score'] <= 0)
    )

    # Apply signal shift if specified (shift signals forward to delay execution)
    if signal_shift > 0:
        df_signals['entry_long'] = df_signals['entry_long'].shift(signal_shift).fillna(False)
        df_signals['exit_long'] = df_signals['exit_long'].shift(signal_shift).fillna(False)
        df_signals['entry_short'] = df_signals['entry_short'].shift(signal_shift).fillna(False)
        df_signals['exit_short'] = df_signals['exit_short'].shift(signal_shift).fillna(False)

    # ============================================================
    # 3. Calculate Position State and Portfolio Tracking
    # ============================================================
    # Position states: 0 = flat, 1 = long, -1 = short
    positions = []
    shares_held = []
    cash_balance = []
    equity_value = []
    total_portfolio = []
    
    current_position = 0
    current_shares = 0
    current_cash = initial_cash

    for i in range(len(df_signals)):
        price = df_signals.iloc[i]['Adj_Close']
        
        # First (window-1) days: no positions (indicator lookback period)
        if i < window - 1:
            positions.append(0)
            shares_held.append(0)
            cash_balance.append(current_cash)
            equity_value.append(0)
            total_portfolio.append(current_cash)
            continue

        # Process exit signals first (priority)
        if current_position == 1 and df_signals.iloc[i]['exit_long']:
            # Exit long: sell shares
            current_cash += current_shares * price
            current_shares = 0
            current_position = 0
            
        elif current_position == -1 and df_signals.iloc[i]['exit_short']:
            # Exit short: buy back shares to cover
            current_cash += current_shares * price  # current_shares is negative for short
            current_shares = 0
            current_position = 0
            
        # Process entry signals (only if flat)
        elif current_position == 0:
            # Check if we should trade in this regime (if target_regime is specified)
            regime_check = True
            if target_regime is not None and 'regime' in df_signals.columns:
                current_regime = df_signals.iloc[i]['regime']
                # Only trade if regime matches (skip NaN regimes too)
                regime_check = (not pd.isna(current_regime)) and (current_regime == target_regime)
            
            if regime_check and df_signals.iloc[i]['entry_long']:
                # Enter long position
                if position_sizing_method == 'capital_based':
                    # Calculate max shares based on available capital
                    available_capital = current_cash * capital_allocation_pct
                    current_shares = int(available_capital / price)
                else:
                    # Use fixed position size
                    current_shares = position_size
                
                # Execute trade if we can afford at least 1 share
                if current_shares > 0:
                    current_cash -= current_shares * price
                    current_position = 1
                else:
                    current_shares = 0
                    
            elif regime_check and df_signals.iloc[i]['entry_short']:
                # Enter short position
                if position_sizing_method == 'capital_based':
                    # Calculate max shares based on available capital
                    available_capital = current_cash * capital_allocation_pct
                    current_shares = -int(available_capital / price)  # Negative for short
                else:
                    # Use fixed position size (negative for short)
                    current_shares = -position_size
                
                # Execute trade if we can afford at least 1 share
                if current_shares < 0:
                    current_cash -= current_shares * price  # Subtracting negative adds to cash
                    current_position = -1
                else:
                    current_shares = 0

        # Record state
        positions.append(current_position)
        shares_held.append(current_shares)
        cash_balance.append(current_cash)
        
        
        # Calculate equity value (mark-to-market)
        equity = current_shares * price
        equity_value.append(equity)
        total_portfolio.append(current_cash + equity)

    df_signals['signal'] = positions
    df_signals['shares'] = shares_held
    df_signals['cash'] = cash_balance
    df_signals['equity'] = equity_value
    df_signals['portfolio_value'] = total_portfolio

    # ============================================================
    # 4. Calculate Returns and PnL
    # ============================================================
    df_signals['returns'] = df_signals['Adj_Close'].pct_change()
    df_signals['position'] = df_signals['signal'].shift(1).fillna(0)
    df_signals['strategy_returns'] = df_signals['returns'] * df_signals['position']       

    # Calculate daily PnL from portfolio value changes
    df_signals['pnl'] = df_signals['portfolio_value'].diff()
    
    # Calculate cumulative PnL
    df_signals['cumulative_pnl'] = df_signals['portfolio_value'] - initial_cash

    return df_signals



def mr_sma_bb_hyperparameter_test(df, 
                       window_range=[10, 20, 30, 50],
                       z_threshold_long_range=[-2.5, -2.0, -1.5, -1.0],
                       z_threshold_short_range=[1.0, 1.5, 2.0, 2.5],
                       position_sizing_methods=['fixed', 'capital_based'],
                       position_sizes=[150],
                       capital_allocation_pcts=[0.98],
                       initial_cash=100000,
                       sort_by='Sharpe Ratio',
                       top_n=20,
                       verbose=True,
                       n_jobs=-1,
                       target_regime=None):
    
    import itertools
    from datetime import datetime
    from joblib import Parallel, delayed
    import io
    import sys
    
    results = []
    
    # Generate all parameter combinations
    param_combinations = []
    
    for window in window_range:
        for z_long in z_threshold_long_range:
            for z_short in z_threshold_short_range:
                # Only test valid threshold combinations (short > long in absolute value makes sense)
                if z_short > abs(z_long):
                    for method in position_sizing_methods:
                        if method == 'fixed':
                            for pos_size in position_sizes:
                                param_combinations.append({
                                    'window': window,
                                    'z_threshold_long': z_long,
                                    'z_threshold_short': z_short,
                                    'position_sizing_method': method,
                                    'position_size': pos_size,
                                    'capital_allocation_pct': None
                                })
                        else:  # capital_based
                            for cap_alloc in capital_allocation_pcts:
                                param_combinations.append({
                                    'window': window,
                                    'z_threshold_long': z_long,
                                    'z_threshold_short': z_short,
                                    'position_sizing_method': method,
                                    'position_size': None,
                                    'capital_allocation_pct': cap_alloc
                                })
    
    # Helper function to test a single parameter combination
    def test_single_combination(params):
        try:
            # Run strategy
            if params['position_sizing_method'] == 'fixed':
                df_signals = mr_compute_mean_reversion_sma_bb_strategy(
                    df,
                    window=params['window'],
                    position_size=params['position_size'],
                    initial_cash=initial_cash,
                    z_threshold_long=params['z_threshold_long'],
                    z_threshold_short=params['z_threshold_short'],
                    position_sizing_method=params['position_sizing_method'],
                    target_regime=target_regime
                )
            else:  # capital_based
                df_signals = mr_compute_mean_reversion_sma_bb_strategy(
                    df,
                    window=params['window'],
                    initial_cash=initial_cash,
                    z_threshold_long=params['z_threshold_long'],
                    z_threshold_short=params['z_threshold_short'],
                    position_sizing_method=params['position_sizing_method'],
                    capital_allocation_pct=params['capital_allocation_pct'],
                    target_regime=target_regime
                )
            
            # Calculate metrics (suppress print output)
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            metrics = mr_sma_bb_calculate_performance_metrics(df_signals, initial_cash=initial_cash)
            
            sys.stdout = old_stdout
            
            # Store results
            result = {
                'Window': params['window'],
                'Z_Long': params['z_threshold_long'],
                'Z_Short': params['z_threshold_short'],
                'Method': params['position_sizing_method'],
                'Position_Size': params['position_size'],
                'Capital_Alloc_Pct': params['capital_allocation_pct'],
                **metrics  # Add all metrics
            }
            return result
        
        except Exception as e:
            return None
    
    total_tests = len(param_combinations)
    if verbose:
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TEST - PARALLEL PROCESSING")
        print(f"{'='*70}")
        print(f"Testing {total_tests} parameter combinations...")
        print(f"Using {n_jobs if n_jobs > 0 else 'ALL'} CPU cores")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")
    
    # Run tests in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(test_single_combination)(params) for params in param_combinations
    )
    
    # Filter out None results (failed tests)
    results = [r for r in results if r is not None]
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\nCompleted at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Successful tests: {len(results_df)}/{total_tests}")
    
    # Sort by specified metric (descending for most metrics)
    if sort_by in results_df.columns:
        ascending = True if 'Drawdown' in sort_by else False  # Drawdown should be ascending (less negative is better)
        results_df = results_df.sort_values(by=sort_by, ascending=ascending)
    
    # Return top N results
    return results_df.head(top_n) if top_n else results_df

def mr_sma_bb_calculate_performance_metrics(df_signals, initial_cash=100000):
    # Calculate signal changes to identify individual trades
    df_signals['signal_change'] = df_signals['signal'].diff()

    # Identify trade periods (when position is held)
    df_long = df_signals[df_signals['signal'] == 1].copy()
    df_short = df_signals[df_signals['signal'] == -1].copy()

    # Calculate trade-level P&L by grouping consecutive positions
    trades = []
    current_signal = 0
    entry_idx = None

    for idx, row in df_signals.iterrows():
        if row['signal'] != current_signal:
            # Position changed - close previous trade if exists
            if current_signal != 0 and entry_idx is not None:
                exit_idx = idx
                trade_pnl = df_signals.loc[entry_idx:idx, 'pnl'].sum()
                trades.append({
                    'entry_date': entry_idx,
                    'exit_date': exit_idx,
                    'direction': 'Long' if current_signal == 1 else 'Short',
                    'pnl': trade_pnl
                })

            # Start new trade if entering position
            if row['signal'] != 0:
                entry_idx = idx
            current_signal = row['signal']

    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)

    # Separate by direction and win/loss
    if len(trades_df) > 0:
        winning_longs = trades_df[(trades_df['direction'] == 'Long') & (trades_df['pnl'] > 0)]
        losing_longs = trades_df[(trades_df['direction'] == 'Long') & (trades_df['pnl'] <= 0)]
        winning_shorts = trades_df[(trades_df['direction'] == 'Short') & (trades_df['pnl'] > 0)]
        losing_shorts = trades_df[(trades_df['direction'] == 'Short') & (trades_df['pnl'] <= 0)]
    else:
        winning_longs = losing_longs = winning_shorts = losing_shorts = pd.DataFrame()

    # Calculate total returns
    # Use portfolio_value if available (more accurate), otherwise calculate from pnl
    if 'portfolio_value' in df_signals.columns:
        final_value = df_signals['portfolio_value'].iloc[-1]
        total_pnl = final_value - initial_cash
    else:
        total_pnl = df_signals['pnl'].sum()
        final_value = initial_cash + total_pnl
    
    total_return_pct = (total_pnl / initial_cash) * 100

    # Calculate CAGR
    years = (df_signals.index[-1] - df_signals.index[0]).days / 365.25
    cagr = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Calculate Sharpe Ratio (annualized)
    daily_returns = df_signals['strategy_returns'].dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Calculate Max Drawdown
    cumulative_returns = (1 + df_signals['strategy_returns']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Calculate volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0

    # Calculate final P&L by direction
    final_long_pnl = (winning_longs['pnl'].sum() if len(winning_longs) > 0 else 0) + \
                     (losing_longs['pnl'].sum() if len(losing_longs) > 0 else 0)
    final_short_pnl = (winning_shorts['pnl'].sum() if len(winning_shorts) > 0 else 0) + \
                      (losing_shorts['pnl'].sum() if len(losing_shorts) > 0 else 0)

    # Compile metrics
    metrics = {
        'Total P&L ($)': total_pnl,
        'Total Return (%)': total_return_pct,
        'CAGR (%)': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Volatility (% annualized)': volatility,
        'Total Trades': len(trades_df),
        'Winning Longs': len(winning_longs),
        'Winning Longs P&L ($)': winning_longs['pnl'].sum() if len(winning_longs) > 0 else 0,
        'Losing Longs': len(losing_longs),
        'Losing Longs P&L ($)': losing_longs['pnl'].sum() if len(losing_longs) > 0 else 0,
        'Final Long P&L ($)': final_long_pnl,
        'Long Win Rate (%)': (len(winning_longs) / (len(winning_longs) + len(losing_longs)) * 100) if (len(winning_longs) + len(losing_longs)) > 0 else 0,
        'Winning Shorts': len(winning_shorts),
        'Winning Shorts P&L ($)': winning_shorts['pnl'].sum() if len(winning_shorts) > 0 else 0,
        'Losing Shorts': len(losing_shorts),
        'Losing Shorts P&L ($)': losing_shorts['pnl'].sum() if len(losing_shorts) > 0 else 0,
        'Final Short P&L ($)': final_short_pnl,
        'Short Win Rate (%)': (len(winning_shorts) / (len(winning_shorts) + len(losing_shorts)) * 100) if (len(winning_shorts) + len(losing_shorts)) > 0 else 0,
    }

    # Print formatted summary
    print(f"\n{'='*70}")
    print(f"{'PERFORMANCE METRICS SUMMARY':^70}")
    print(f"{'='*70}\n")

    print(f"{'OVERALL PERFORMANCE':-^70}")
    print(f"Total P&L:                ${metrics['Total P&L ($)']:>15,.2f}")
    print(f"Total Return:             {metrics['Total Return (%)']:>15,.2f}%")
    print(f"CAGR:                     {metrics['CAGR (%)']:>15,.2f}%")
    print(f"Sharpe Ratio:             {metrics['Sharpe Ratio']:>15,.3f}")
    print(f"Max Drawdown:             {metrics['Max Drawdown (%)']:>15,.2f}%")
    print(f"Volatility (annualized):  {metrics['Volatility (% annualized)']:>15,.2f}%")
    print(f"\n{'TRADE STATISTICS':-^70}")
    print(f"Total Trades:             {metrics['Total Trades']:>15,}")
    print(f"\n{'LONG POSITIONS':-^70}")
    print(f"Winning Longs:            {metrics['Winning Longs']:>15,} trades")
    print(f"Winning Longs P&L:        ${metrics['Winning Longs P&L ($)']:>15,.2f}")
    print(f"Losing Longs:             {metrics['Losing Longs']:>15,} trades")
    print(f"Losing Longs P&L:         ${metrics['Losing Longs P&L ($)']:>15,.2f}")
    print(f"Final Long P&L:           ${metrics['Final Long P&L ($)']:>15,.2f}")
    print(f"Long Win Rate:            {metrics['Long Win Rate (%)']:>15,.2f}%")
    print(f"\n{'SHORT POSITIONS':-^70}")
    print(f"Winning Shorts:           {metrics['Winning Shorts']:>15,} trades")
    print(f"Winning Shorts P&L:       ${metrics['Winning Shorts P&L ($)']:>15,.2f}")
    print(f"Losing Shorts:            {metrics['Losing Shorts']:>15,} trades")
    print(f"Losing Shorts P&L:        ${metrics['Losing Shorts P&L ($)']:>15,.2f}")
    print(f"Final Short P&L:          ${metrics['Final Short P&L ($)']:>15,.2f}")
    print(f"Short Win Rate:           {metrics['Short Win Rate (%)']:>15,.2f}%")
    print(f"\n{'='*70}\n")

    return metrics


def mr_global_compute_performance_metrics(df_signals, initial_cash=100000, strategy='Mean_Reversion'):
    """
    Calculate performance metrics for mean reversion strategy.
    Returns a DataFrame with metrics in the same format as compute_performance_metrics.
    """
    # Calculate total returns
    if 'portfolio_value' in df_signals.columns:
        final_value = df_signals['portfolio_value'].iloc[-1]
        total_pnl = final_value - initial_cash
    else:
        total_pnl = df_signals['pnl'].sum()
        final_value = initial_cash + total_pnl

    # Calculate CAGR
    years = (df_signals.index[-1] - df_signals.index[0]).days / 365.25
    cagr = ((final_value / initial_cash) ** (1 / years) - 1) if years > 0 else 0

    # Calculate daily returns metrics
    daily_returns = df_signals['strategy_returns'].dropna()

    # Volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0

    # Sharpe Ratio (annualized)
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Sortino Ratio (annualized)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
    else:
        sortino_ratio = np.nan

    # Max Drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # Calmar Ratio
    calmar_ratio = cagr / max_drawdown if max_drawdown != 0 else np.nan

    # Cumulative Return
    cumulative_return = (1 + daily_returns).prod()

    metrics = {
        "CAGR": cagr,
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Final Portfolio Value": final_value,
        "Cumulative Return": cumulative_return
    }

    df = pd.DataFrame(metrics, index=[strategy])

    # Clean up all numeric columns to max 3 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].map(lambda x: round(x, 3))

    return df



# %%
def mr_sma_bb_viz_backtest_mean_reversion(df_signals, start_date=None, end_date=None):
    # Filter by date range if specified
    df_plot = df_signals.copy()
    if start_date is not None:
        df_plot = df_plot[df_plot.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot[df_plot.index <= pd.to_datetime(end_date)]

    plt.figure(figsize=[17, 6])

    # Plot price
    plt.plot(df_plot["Adj_Close"], label="Price = Adjusted Close")

    # Plot threshold bands
    plt.plot(df_plot["range_upper"], label="Upper Bound Threshold", ls="-.")
    plt.plot(df_plot["range_lower"], label="Lower Bound Threshold", ls="-.")

    # Plot moving average
    plt.plot(df_plot["dynamic_sma"], label="Dynamic SMA")

    # Calculate signal changes (actual trade executions)
    df_plot['signal_change'] = df_plot['signal'].diff()
    df_plot['prev_signal'] = df_plot['signal'].shift(1)

    # LONG Entry: signal changes to 1 (0→1 or -1→1)
    long_entries = df_plot[(df_plot['signal'] == 1) & (df_plot['prev_signal'] != 1)]
    if len(long_entries) > 0:
        long_entries["Adj_Close"].plot(label="Long Entry", style="g^", markersize=10)

    # LONG Exit: signal changes from 1 to something else
    long_exits = df_plot[(df_plot['prev_signal'] == 1) & (df_plot['signal'] != 1)]
    if len(long_exits) > 0:
        long_exits["Adj_Close"].plot(label="Long Exit", style="v", color="lightgreen", markersize=10)

    # SHORT Entry: signal changes to -1 (0→-1 or 1→-1)
    short_entries = df_plot[(df_plot['signal'] == -1) & (df_plot['prev_signal'] != -1)]
    if len(short_entries) > 0:
        short_entries["Adj_Close"].plot(label="Short Entry", style="rv", markersize=10)

    # SHORT Exit: signal changes from -1 to something else
    short_exits = df_plot[(df_plot['prev_signal'] == -1) & (df_plot['signal'] != -1)]
    if len(short_exits) > 0:
        short_exits["Adj_Close"].plot(label="Short Exit", style="^", color="lightcoral", markersize=10)

    plt.title("Mean Reversion Strategy - Backtesting with Buy & Sell Signals")
    plt.legend()
    plt.show()


# %%
def mr_sma_bb_viz_regime_aware_backtest_signals(df_results, start_date=None, end_date=None, regime_names_dict=None):
    """
    Visualize regime-aware mean reversion strategy with regime backgrounds and entry/exit signals.
    
    Parameters:
    -----------
    df_results : DataFrame from compute_regime_aware_strategy() 
    start_date : str or datetime, optional start date for zoom (e.g., '2020-01-01')
    end_date : str or datetime, optional end date for zoom (e.g., '2020-12-31')
    regime_names_dict : dict mapping regime IDs to names
    """
    # Filter by date range if specified
    df_plot = df_results.copy()
    if start_date is not None:
        df_plot = df_plot[df_plot.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot[df_plot.index <= pd.to_datetime(end_date)]
    
    # Define regime colors (gradient from calm to crisis)
    regime_colors = {
        0: 'lightgreen',    # Very Low Vol: Green (calm)
        1: 'lightblue',     # Low Vol: Blue (stable)
        2: 'lightyellow',   # Med Vol: Yellow (normal)
        3: 'orange',        # High Vol: Orange (elevated)
        4: 'lightcoral'     # Very High Vol: Red (crisis)
    }
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Add regime background first (behind everything)
    if 'regime' in df_plot.columns:
        for regime in df_plot['regime'].dropna().unique():
            regime_mask = df_plot['regime'] == regime
            regime_label = regime_names_dict.get(int(regime), f'Regime {int(regime)}') if regime_names_dict else f'Regime {int(regime)}'
            ax.fill_between(df_plot.index,
                           df_plot["Adj_Close"].min() * 0.9,
                           df_plot["Adj_Close"].max() * 1.1,
                           where=regime_mask,
                           alpha=0.2,
                           color=regime_colors.get(int(regime), 'lightgray'),
                           label=regime_label)
    
    # Plot price
    ax.plot(df_plot.index, df_plot["Adj_Close"], label="SPY Price", linewidth=2, color='black', zorder=3)
    
    # Plot threshold bands
    ax.plot(df_plot.index, df_plot["range_upper"], label="Upper Threshold", 
            ls="--", linewidth=1.5, color='red', alpha=0.7, zorder=2)
    ax.plot(df_plot.index, df_plot["range_lower"], label="Lower Threshold", 
            ls="--", linewidth=1.5, color='green', alpha=0.7, zorder=2)
    
    # Plot dynamic SMA
    ax.plot(df_plot.index, df_plot["dynamic_sma"], label="Dynamic SMA (Regime-Aware)", 
            linewidth=2, color='blue', alpha=0.8, zorder=2)
    
    # Calculate signal changes (actual trade executions)
    df_plot['signal_change'] = df_plot['signal'].diff()
    df_plot['prev_signal'] = df_plot['signal'].shift(1)
    
    # LONG Entry: signal changes to 1
    long_entries = df_plot[(df_plot['signal'] == 1) & (df_plot['prev_signal'] != 1)]
    if len(long_entries) > 0:
        ax.scatter(long_entries.index, long_entries["Adj_Close"], 
                  label="LONG Entry", marker="^", s=150, color="darkgreen", 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # LONG Exit: signal changes from 1 to something else
    long_exits = df_plot[(df_plot['prev_signal'] == 1) & (df_plot['signal'] != 1)]
    if len(long_exits) > 0:
        ax.scatter(long_exits.index, long_exits["Adj_Close"], 
                  label="LONG Exit", marker="v", s=150, color="lightgreen", 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # SHORT Entry: signal changes to -1
    short_entries = df_plot[(df_plot['signal'] == -1) & (df_plot['prev_signal'] != -1)]
    if len(short_entries) > 0:
        ax.scatter(short_entries.index, short_entries["Adj_Close"], 
                  label="SHORT Entry", marker="v", s=150, color="darkred", 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # SHORT Exit: signal changes from -1 to something else
    short_exits = df_plot[(df_plot['prev_signal'] == -1) & (df_plot['signal'] != -1)]
    if len(short_exits) > 0:
        ax.scatter(short_exits.index, short_exits["Adj_Close"], 
                  label="SHORT Exit", marker="^", s=150, color="lightcoral", 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # Formatting
    date_range = f"{df_plot.index[0].strftime('%Y-%m-%d')} to {df_plot.index[-1].strftime('%Y-%m-%d')}"
    ax.set_title(f"Regime-Aware Mean Reversion Strategy - Entry/Exit Signals\n{date_range}", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print signal summary
    print("\n" + "="*70)
    print("SIGNAL SUMMARY")
    print("="*70)
    print(f"Long Entries:  {len(long_entries)}")
    print(f"Long Exits:    {len(long_exits)}")
    print(f"Short Entries: {len(short_entries)}")
    print(f"Short Exits:   {len(short_exits)}")
    print(f"Total Signals: {len(long_entries) + len(long_exits) + len(short_entries) + len(short_exits)}")
    print("="*70)
    


# %%
def mr_sma_bb_viz_backtest_returns(df_signals, initial_cash=100000):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[17, 10])

    # Calculate cumulative returns
    df_plot = df_signals.copy()

    # Strategy cumulative returns
    df_plot['cum_strategy_returns'] = (1 + df_plot['strategy_returns']).cumprod() - 1

    # Buy-and-hold cumulative returns
    df_plot['cum_asset_returns'] = (1 + df_plot['returns']).cumprod() - 1

    # Calculate running maximum for drawdown
    strategy_cummax = (1 + df_plot['strategy_returns']).cumprod().expanding().max()
    df_plot['drawdown'] = ((1 + df_plot['strategy_returns']).cumprod() - strategy_cummax) / strategy_cummax

    # Plot 1: Cumulative Returns
    df_plot[['cum_asset_returns', 'cum_strategy_returns']].dropna().plot(ax=ax1, linewidth=1.5)

    # Format y-axis as percentage
    from matplotlib.ticker import PercentFormatter
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax1.legend(['Buy & Hold', 'Mean Reversion Strategy'], loc='best')
    ax1.set_title('Cumulative Returns: Strategy vs Buy-and-Hold', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(alpha=0.3)

    # Plot 2: Drawdown
    df_plot['drawdown'].plot(ax=ax2, color='red', linewidth=1.5)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax2.set_title('Strategy Drawdown', fontsize=14)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.grid(alpha=0.3)
    ax2.fill_between(df_plot.index, df_plot['drawdown'], 0, color='red', alpha=0.3)

    plt.tight_layout()
    plt.show()


# %%
def mr_sma_bb_viz_regime_aware_backtest_comparison(train_results, test_results, regime_names_dict=None):
    fig, axes = plt.subplots(3, 2, figsize=(24, 16))
    
    # Define regime colors (gradient from calm to crisis)
    regime_colors = {
        0: 'lightgreen',    # Very Low Vol: Green (calm)
        1: 'lightblue',     # Low Vol: Blue (stable)
        2: 'lightyellow',   # Med Vol: Yellow (normal)
        3: 'orange',        # High Vol: Orange (elevated)
        4: 'lightcoral'     # Very High Vol: Red (crisis)
    }
    
    # ============================================================
    # TRAINING DATA (Left Column)
    # ============================================================
    
    # 1. Training: Cumulative Returns with Regime Background
    ax_train_returns = axes[0, 0]
    train_cum_strategy = (1 + train_results['strategy_returns']).cumprod() - 1
    train_cum_bh = (1 + train_results['returns_simple']).cumprod() - 1 if 'returns_simple' in train_results.columns else None
    
    # Add regime background
    if 'regime' in train_results.columns:
        for regime in train_results['regime'].dropna().unique():
            regime_mask = train_results['regime'] == regime
            ax_train_returns.fill_between(train_results.index,
                                         train_cum_strategy.min() * 1.1,
                                         train_cum_strategy.max() * 1.1,
                                         where=regime_mask,
                                         alpha=0.15,
                                         color=regime_colors.get(int(regime), 'lightgray'))
    
    ax_train_returns.plot(train_results.index, train_cum_strategy * 100,
                         linewidth=2, color='blue', label='Strategy')
    if train_cum_bh is not None:
        ax_train_returns.plot(train_results.index, train_cum_bh * 100,
                             linewidth=2, color='gray', alpha=0.7, label='Buy & Hold')
    
    ax_train_returns.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_train_returns.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax_train_returns.set_title('TRAINING: Cumulative Returns by Regime', fontsize=14, fontweight='bold')
    ax_train_returns.legend(loc='upper left', fontsize=10)
    ax_train_returns.grid(alpha=0.3)
    
    # 2. Training: Drawdown
    ax_train_dd = axes[1, 0]
    train_cummax = (1 + train_results['strategy_returns']).cumprod().expanding().max()
    train_dd = ((1 + train_results['strategy_returns']).cumprod() - train_cummax) / train_cummax
    
    ax_train_dd.plot(train_results.index, train_dd * 100, linewidth=2, color='red')
    ax_train_dd.fill_between(train_results.index, train_dd * 100, 0, color='red', alpha=0.3)
    ax_train_dd.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_train_dd.set_ylabel('Drawdown (%)', fontsize=12)
    ax_train_dd.set_title('TRAINING: Strategy Drawdown', fontsize=14, fontweight='bold')
    ax_train_dd.grid(alpha=0.3)
    
    # 3. Training: Portfolio Value
    ax_train_pv = axes[2, 0]
    ax_train_pv.plot(train_results.index, train_results['portfolio_value'],
                    linewidth=2, color='green')
    ax_train_pv.axhline(y=100000, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax_train_pv.fill_between(train_results.index, 100000, train_results['portfolio_value'],
                            where=(train_results['portfolio_value'] >= 100000),
                            alpha=0.3, color='green')
    ax_train_pv.fill_between(train_results.index, 100000, train_results['portfolio_value'],
                            where=(train_results['portfolio_value'] < 100000),
                            alpha=0.3, color='red')
    ax_train_pv.set_xlabel('Date', fontsize=12)
    ax_train_pv.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax_train_pv.set_title('TRAINING: Portfolio Growth', fontsize=14, fontweight='bold')
    ax_train_pv.grid(alpha=0.3)
    
    # ============================================================
    # TEST DATA (Right Column)
    # ============================================================
    
    # 1. Test: Cumulative Returns with Regime Background
    ax_test_returns = axes[0, 1]
    test_cum_strategy = (1 + test_results['strategy_returns']).cumprod() - 1
    test_cum_bh = (1 + test_results['returns_simple']).cumprod() - 1 if 'returns_simple' in test_results.columns else None
    
    # Add regime background
    if 'regime' in test_results.columns:
        for regime in test_results['regime'].dropna().unique():
            regime_mask = test_results['regime'] == regime
            regime_label = regime_names_dict.get(int(regime), f'Regime {int(regime)}') if regime_names_dict else f'Regime {int(regime)}'
            ax_test_returns.fill_between(test_results.index,
                                        test_cum_strategy.min() * 1.1,
                                        test_cum_strategy.max() * 1.1,
                                        where=regime_mask,
                                        alpha=0.15,
                                        color=regime_colors.get(int(regime), 'lightgray'),
                                        label=regime_label)
    
    ax_test_returns.plot(test_results.index, test_cum_strategy * 100,
                        linewidth=2, color='blue', label='Strategy')
    if test_cum_bh is not None:
        ax_test_returns.plot(test_results.index, test_cum_bh * 100,
                            linewidth=2, color='gray', alpha=0.7, label='Buy & Hold')
    
    ax_test_returns.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_test_returns.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax_test_returns.set_title('TEST: Cumulative Returns by Regime (Out-of-Sample)', fontsize=14, fontweight='bold')
    ax_test_returns.legend(loc='upper left', fontsize=10)
    ax_test_returns.grid(alpha=0.3)
    
    # 2. Test: Drawdown
    ax_test_dd = axes[1, 1]
    test_cummax = (1 + test_results['strategy_returns']).cumprod().expanding().max()
    test_dd = ((1 + test_results['strategy_returns']).cumprod() - test_cummax) / test_cummax
    
    ax_test_dd.plot(test_results.index, test_dd * 100, linewidth=2, color='red')
    ax_test_dd.fill_between(test_results.index, test_dd * 100, 0, color='red', alpha=0.3)
    ax_test_dd.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_test_dd.set_ylabel('Drawdown (%)', fontsize=12)
    ax_test_dd.set_title('TEST: Strategy Drawdown', fontsize=14, fontweight='bold')
    ax_test_dd.grid(alpha=0.3)
    
    # 3. Test: Portfolio Value
    ax_test_pv = axes[2, 1]
    ax_test_pv.plot(test_results.index, test_results['portfolio_value'],
                   linewidth=2, color='green')
    ax_test_pv.axhline(y=100000, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax_test_pv.fill_between(test_results.index, 100000, test_results['portfolio_value'],
                           where=(test_results['portfolio_value'] >= 100000),
                           alpha=0.3, color='green')
    ax_test_pv.fill_between(test_results.index, 100000, test_results['portfolio_value'],
                           where=(test_results['portfolio_value'] < 100000),
                           alpha=0.3, color='red')
    ax_test_pv.set_xlabel('Date', fontsize=12)
    ax_test_pv.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax_test_pv.set_title('TEST: Portfolio Growth', fontsize=14, fontweight='bold')
    ax_test_pv.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Calculate SPY (buy-and-hold) drawdowns
    if train_cum_bh is not None:
        train_spy_cummax = (1 + train_results['returns_simple']).cumprod().expanding().max()
        train_spy_dd = ((1 + train_results['returns_simple']).cumprod() - train_spy_cummax) / train_spy_cummax
        train_spy_final_value = 100000 * (1 + train_cum_bh.iloc[-1])
    else:
        train_spy_dd = pd.Series([0])
        train_spy_final_value = 100000

    if test_cum_bh is not None:
        test_spy_cummax = (1 + test_results['returns_simple']).cumprod().expanding().max()
        test_spy_dd = ((1 + test_results['returns_simple']).cumprod() - test_spy_cummax) / test_spy_cummax
        test_spy_final_value = 100000 * (1 + test_cum_bh.iloc[-1])
    else:
        test_spy_dd = pd.Series([0])
        test_spy_final_value = 100000

    # Print summary statistics
    print("\n" + "="*100)
    print("SIDE-BY-SIDE COMPARISON SUMMARY")
    print("="*100)
    print(f"\n{'Metric':<30} {'Strategy (Train)':>18} {'Strategy (Test)':>18} {'SPY B&H (Train)':>18} {'SPY B&H (Test)':>18}")
    print("-"*100)
    print(f"{'Final Return (%)':<30} {train_cum_strategy.iloc[-1]*100:>18.2f} {test_cum_strategy.iloc[-1]*100:>18.2f} {train_cum_bh.iloc[-1]*100 if train_cum_bh is not None else 0:>18.2f} {test_cum_bh.iloc[-1]*100 if test_cum_bh is not None else 0:>18.2f}")
    print(f"{'Max Drawdown (%)':<30} {train_dd.min()*100:>18.2f} {test_dd.min()*100:>18.2f} {train_spy_dd.min()*100:>18.2f} {test_spy_dd.min()*100:>18.2f}")
    print(f"{'Final Portfolio ($)':<30} ${train_results['portfolio_value'].iloc[-1]:>17,.2f} ${test_results['portfolio_value'].iloc[-1]:>17,.2f} ${train_spy_final_value:>17,.2f} ${test_spy_final_value:>17,.2f}")
    print("="*100)




def mr_sma_bb_compute_regime_aware_strategy(df_with_regimes, regime_params_dict,
                                  initial_cash=100000, position_sizing_method='capital_based',
                                  capital_allocation_pct=0.98, signal_shift=0):

    df_signals = df_with_regimes.copy()
    
    # Initialize columns
    df_signals['dynamic_sma'] = np.nan
    df_signals['dynamic_std'] = np.nan
    df_signals['z_score'] = 0.0
    df_signals['range_upper'] = np.nan
    df_signals['range_lower'] = np.nan
    df_signals['entry_long'] = False
    df_signals['exit_long'] = False
    df_signals['entry_short'] = False
    df_signals['exit_short'] = False
    
    # Calculate indicators dynamically based on regime
    for i in range(len(df_signals)):
        # Get current regime (use regime 0 params if NaN)
        current_regime = df_signals['regime'].iloc[i]
        if pd.isna(current_regime):
            current_regime = 0  # Default to first regime if missing
        
        # Get regime-specific parameters
        params = regime_params_dict.get(int(current_regime), regime_params_dict[0])
        window = int(params['window'])
        z_long = float(params['z_long'])
        z_short = float(params['z_short'])
        
        # Calculate z-score with regime-specific window
        # Use min_periods=1 to match pandas rolling behavior (calculate from first bar)
        if i >= 0:  # Calculate for all bars (min_periods=1)
            # Get up to 'window' prices including current, but allow fewer for early bars
            start_idx = max(0, i - window + 1)
            prices = df_signals['Adj_Close'].iloc[start_idx:i+1]
            sma = prices.mean()
            std = prices.std()
            
            df_signals.loc[df_signals.index[i], 'dynamic_sma'] = sma
            df_signals.loc[df_signals.index[i], 'dynamic_std'] = std
            
            if std > 0:
                z_score = (df_signals['Adj_Close'].iloc[i] - sma) / std
                df_signals.loc[df_signals.index[i], 'z_score'] = z_score
            else:
                z_score = 0
                df_signals.loc[df_signals.index[i], 'z_score'] = 0
            
            # Calculate bands
            df_signals.loc[df_signals.index[i], 'range_upper'] = sma + z_short * std
            df_signals.loc[df_signals.index[i], 'range_lower'] = sma + z_long * std
            
            # Generate signals using regime-specific thresholds
            if i > 0:  # Need previous z-score for signal
                z_prev = df_signals['z_score'].iloc[i-1]
                z_curr = df_signals['z_score'].iloc[i]
                
                # Entry Long: crosses down through threshold
                df_signals.loc[df_signals.index[i], 'entry_long'] = (
                    (z_prev > z_long) and (z_curr <= z_long)
                )
                
                # Exit Long: crosses up through 0
                df_signals.loc[df_signals.index[i], 'exit_long'] = (
                    (z_prev < 0) and (z_curr >= 0)
                )
                
                # Entry Short: crosses up through threshold
                df_signals.loc[df_signals.index[i], 'entry_short'] = (
                    (z_prev < z_short) and (z_curr >= z_short)
                )
                
                # Exit Short: crosses down through 0
                df_signals.loc[df_signals.index[i], 'exit_short'] = (
                    (z_prev > 0) and (z_curr <= 0)
                )

    # Apply signal shift if specified (shift signals forward to delay execution)
    if signal_shift > 0:
        df_signals['entry_long'] = df_signals['entry_long'].shift(signal_shift).fillna(False)
        df_signals['exit_long'] = df_signals['exit_long'].shift(signal_shift).fillna(False)
        df_signals['entry_short'] = df_signals['entry_short'].shift(signal_shift).fillna(False)
        df_signals['exit_short'] = df_signals['exit_short'].shift(signal_shift).fillna(False)

    # ============================================================
    # Calculate Position State and Portfolio Tracking
    # ============================================================
    positions = []
    shares_held = []
    cash_balance = []
    portfolio_values = []
    pnl_daily = []
    
    position = 0  # 0 = flat, 1 = long, -1 = short
    shares = 0
    cash = initial_cash
    entry_price = 0
    
    for i in range(len(df_signals)):
        price = df_signals['Adj_Close'].iloc[i]
        
        # Default to previous state
        new_position = position
        new_shares = shares
        new_cash = cash
        daily_pnl = 0
        
        # Process signals
        if df_signals['entry_long'].iloc[i] and position <= 0:
            # Close short if exists
            cash_after_closing = cash
            if position == -1:
                cover_cost = shares * price
                daily_pnl = shares * (entry_price - price)  # Short P&L (for tracking)
                cash_after_closing = cash - cover_cost  # P&L already implicit in cash flow
                
            # Enter long (use cash after closing short if any)
            if position_sizing_method == 'capital_based':
                shares_to_buy = int((cash_after_closing * capital_allocation_pct) / price)
            else:
                shares_to_buy = 150  # Default position size
            
            cost = shares_to_buy * price
            if cost <= cash_after_closing:
                new_shares = shares_to_buy
                new_cash = cash_after_closing - cost
                new_position = 1
                entry_price = price
                if position != -1:  # Don't double count if closed short
                    daily_pnl = 0
                    
        elif df_signals['exit_long'].iloc[i] and position == 1:
            # Exit long
            proceeds = shares * price
            daily_pnl = shares * (price - entry_price)
            new_cash = cash + proceeds
            new_position = 0
            new_shares = 0
            entry_price = 0
            
        elif df_signals['entry_short'].iloc[i] and position >= 0:
            # Close long if exists
            cash_from_closing = 0
            if position == 1:
                proceeds = shares * price
                daily_pnl = shares * (price - entry_price)  # Long P&L
                cash_from_closing = proceeds
                
            # Enter short (use cash + proceeds from closing long if any)
            available_cash = cash + cash_from_closing
            if position_sizing_method == 'capital_based':
                shares_to_short = int((available_cash * capital_allocation_pct) / price)
            else:
                shares_to_short = 150
            
            if shares_to_short > 0:
                proceeds_from_short = shares_to_short * price
                new_shares = shares_to_short
                new_cash = available_cash + proceeds_from_short
                new_position = -1
                entry_price = price
                if position != 1:  # Don't double count if closed long
                    daily_pnl = 0
                    
        elif df_signals['exit_short'].iloc[i] and position == -1:
            # Exit short: buy back shares to cover
            cover_cost = shares * price
            daily_pnl = shares * (entry_price - price)  # For tracking only
            new_cash = cash - cover_cost  # P&L already implicit in cash flow
            new_position = 0
            new_shares = 0
            entry_price = 0
        
        # Update state
        position = new_position
        shares = new_shares
        cash = new_cash
        
        # Calculate portfolio value
        if position == 1:
            portfolio_value = cash + shares * price
        elif position == -1:
            portfolio_value = cash - shares * price
        else:
            portfolio_value = cash
        
        # Store results
        positions.append(position)
        shares_held.append(shares)
        cash_balance.append(cash)
        portfolio_values.append(portfolio_value)
        pnl_daily.append(daily_pnl)
    
    # Add to dataframe
    df_signals['signal'] = positions
    df_signals['shares'] = shares_held
    df_signals['cash'] = cash_balance
    df_signals['portfolio_value'] = portfolio_values
    df_signals['pnl'] = pnl_daily
    
    # Calculate strategy returns
    df_signals['strategy_returns'] = df_signals['portfolio_value'].pct_change().fillna(0)
    
    return df_signals



def backtest_mean_reversion_sma_bb_strategy(spy_train_features, spy_test_features):
    print("\n" + "="*70)
    print("LOADING REGIME PARAMETERS FROM CSV FILES (FROM TRAINING DATA)")
    print("="*70)

    print("\n" + "="*70)
    print("GENERATING TRAIN vs TEST VISUALIZATION WITH REGIME OVERLAY (USING TRAINING DATA)")
    print("="*70)

    # Create regime names dict based on actual number of clusters
    # Current setting: nclust = 5 (see line 316)
    nclust = 5
    if nclust == 2:
        regime_names = {0: 'Low Vol', 1: 'High Vol'}
    elif nclust == 3:
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
    elif nclust == 4:
        regime_names = {0: 'Very Low Vol', 1: 'Low Vol', 2: 'Med Vol', 3: 'High Vol'}
    elif nclust == 5:
        regime_names = {
            0: 'Very Low Vol (Calm)',
            1: 'Low Vol (Stable)', 
            2: 'Medium Vol (Normal)',
            3: 'High Vol (Elevated)',
            4: 'Very High Vol (Crisis)'
        }
    else:
        regime_names = {i: f'Regime {i}' for i in range(nclust)}

    # Initialize regime_best_params dictionary (in case optimization section was skipped)
    regime_best_params = {}

    for regime_id in range(5):  # Assuming 5 regimes (0-4)
        csv_path = os.path.join(f'mr_sma_bb_hyperparameter_results_regime_{regime_id}.csv')

        if os.path.exists(csv_path):
            # Read CSV and get best parameters (first row)
            regime_csv = pd.read_csv(csv_path)

            if not regime_csv.empty:
                best_row = regime_csv.iloc[0]
                regime_best_params[regime_id] = best_row.to_dict()

                regime_name = regime_names.get(regime_id, f'Regime {regime_id}')
                print(f"✅ Loaded {regime_name}: Window={best_row['Window']}, Z_Long={best_row['Z_Long']}, Z_Short={best_row['Z_Short']}")
            else:
                print(f"⚠️  Warning: Empty CSV for regime {regime_id}")
        else:
            print(f"❌ CSV not found: {csv_path}")

    print(f"\n✅ Loaded parameters for {len(regime_best_params)} regimes from CSV")
    print("="*70)


    print("\n" + "="*70)
    print("UNIFIED REGIME-AWARE BACKTEST - TRAINING DATA")
    print("="*70)
    print(f"\nTraining Data Period:")
    print(f"  Start Date: {spy_train_features.index[0].strftime('%Y-%m-%d')}")
    print(f"  End Date:   {spy_train_features.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Days: {len(spy_train_features)}")
    print("\nTesting strategy on FULL training data with regime transitions...")

    # Prepare regime parameters dictionary from optimization results
    regime_params_for_strategy = {}
    for regime_id, best_params in regime_best_params.items():
        regime_params_for_strategy[int(regime_id)] = {
            'window': best_params['Window'],
            'z_long': best_params['Z_Long'],
            'z_short': best_params['Z_Short']
        }

    print("\nRegime Parameters Being Used:")
    for regime_id, params in regime_params_for_strategy.items():
        regime_name = regime_names.get(regime_id, f'Regime {regime_id}')
        print(f"  {regime_name}: window={params['window']}, z_long={params['z_long']}, z_short={params['z_short']}")

    # Run unified backtest on full training data
    train_unified_results = mr_sma_bb_compute_regime_aware_strategy(
        spy_train_features,
        regime_params_for_strategy,
        initial_cash=100000,
        position_sizing_method='capital_based',
        capital_allocation_pct=0.98
    )

    # Calculate metrics
    train_unified_metrics = mr_sma_bb_calculate_performance_metrics(train_unified_results, initial_cash=100000)

    print("\n" + "="*70)
    print("UNIFIED BACKTEST RESULTS - TRAINING DATA")
    print("="*70)
    print(f"Total Return:     {train_unified_metrics['Total Return (%)']:.2f}%")
    print(f"CAGR:             {train_unified_metrics['CAGR (%)']:.2f}%")
    print(f"Sharpe Ratio:     {train_unified_metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:     {train_unified_metrics['Max Drawdown (%)']:.2f}%")
    print(f"Total Trades:     {train_unified_metrics['Total Trades']}")
    print("="*70)

    # %%
    # ============================================================
    # RUN UNIFIED REGIME-AWARE BACKTEST ON TEST DATA
    # ============================================================

    print("\n" + "="*70)
    print("UNIFIED REGIME-AWARE BACKTEST - TEST DATA")
    print("="*70)
    print(f"\nTest Data Period:")
    print(f"  Start Date: {spy_test_features.index[0].strftime('%Y-%m-%d')}")
    print(f"  End Date:   {spy_test_features.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Days: {len(spy_test_features)}")
    print("\nTesting strategy on FULL test data with regime transitions...")

    # Run unified backtest on full test data (using same params from training)
    test_unified_results = mr_sma_bb_compute_regime_aware_strategy(
        spy_test_features,
        regime_params_for_strategy,
        initial_cash=100000,
        position_sizing_method='capital_based',
        capital_allocation_pct=0.98
    )

    # Calculate metrics
    test_unified_metrics = mr_sma_bb_calculate_performance_metrics(test_unified_results, initial_cash=100000)

    print("\n" + "="*70)
    print("UNIFIED BACKTEST RESULTS - TEST DATA")
    print("="*70)
    print(f"Total Return:     {test_unified_metrics['Total Return (%)']:.2f}%")
    print(f"CAGR:             {test_unified_metrics['CAGR (%)']:.2f}%")
    print(f"Sharpe Ratio:     {test_unified_metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:     {test_unified_metrics['Max Drawdown (%)']:.2f}%")
    print(f"Total Trades:     {test_unified_metrics['Total Trades']}")
    print("="*70)

    # %%
    # ============================================================
    # COMPARE: REGIME-AWARE VS REGIME-AGNOSTIC
    # ============================================================

    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: TRAIN vs TEST")
    print("="*70)

    comparison_table = pd.DataFrame({
        'Metric': ['Total Return (%)', 'CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Total Trades'],
        'Train': [
            f"{train_unified_metrics['Total Return (%)']:.2f}",
            f"{train_unified_metrics['CAGR (%)']:.2f}",
            f"{train_unified_metrics['Sharpe Ratio']:.2f}",
            f"{train_unified_metrics['Max Drawdown (%)']:.2f}",
            f"{train_unified_metrics['Total Trades']}"
        ],
        'Test': [
            f"{test_unified_metrics['Total Return (%)']:.2f}",
            f"{test_unified_metrics['CAGR (%)']:.2f}",
            f"{test_unified_metrics['Sharpe Ratio']:.2f}",
            f"{test_unified_metrics['Max Drawdown (%)']:.2f}",
            f"{test_unified_metrics['Total Trades']}"
        ]
    })

    display(comparison_table)
    print("="*70)

    #%%
    # ============================================================
    # VISUALIZE TRAIN VS TEST COMPARISON WITH REGIME OVERLAY
    # ============================================================

    print("\n" + "="*70)
    print("GENERATING TRAIN vs TEST VISUALIZATION WITH REGIME OVERLAY")
    print("="*70)


    # Call the comparison visualization
    mr_sma_bb_viz_regime_aware_backtest_comparison(
        train_unified_results,
        test_unified_results,
        regime_names_dict=regime_names
    )

    return test_unified_results


def mr_calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI) using Wilder's smoothing method.

    Parameters:
    -----------
    prices : pandas Series, price data (typically adjusted close)
    period : int, RSI period (default=14)

    Returns:
    --------
    pandas Series with RSI values (0-100 scale)
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate Wilder's smoothed moving averages
    # First average uses simple mean, then uses Wilder's smoothing
    avg_gain = gains.ewm(com=period-1, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(com=period-1, min_periods=period, adjust=False).mean()

    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases (division by zero)
    rsi = rsi.fillna(50)  # When no price movement, RSI = 50 (neutral)

    return rsi


# %%
def mr_compute_mean_reversion_rsi_strategy(df, rsi_period=14, position_size=150, initial_cash=100000,
                                   rsi_oversold=30, rsi_overbought=70,
                                   rsi_exit_long=50, rsi_exit_short=50,
                                   position_sizing_method='capital_based', capital_allocation_pct=0.98,
                                   target_regime=None, signal_shift=0):
    """
    Compute mean reversion strategy based on RSI signals.

    Parameters:
    -----------
    df : DataFrame with 'Adj_Close' column
    rsi_period : int, RSI calculation period (default=14)
    position_size : int, number of shares per trade when using 'fixed' method (default=150)
    initial_cash : float, starting capital (default=100000)
    rsi_oversold : float, RSI threshold for long entry - buy when RSI crosses above this (default=30)
    rsi_overbought : float, RSI threshold for short entry - sell when RSI crosses below this (default=70)
    rsi_exit_long : float, RSI level to exit long positions - sell when RSI crosses above this (default=50)
    rsi_exit_short : float, RSI level to exit short positions - cover when RSI crosses below this (default=50)
    position_sizing_method : str, either 'fixed' or 'capital_based' (default='capital_based')
        - 'fixed': uses fixed position_size parameter
        - 'capital_based': calculates max shares based on available capital
    capital_allocation_pct : float, percentage of available capital to use per trade (default=0.98)
        Only applies when position_sizing_method='capital_based'
    target_regime : int or None, if specified, only trade when current regime matches this value (default=None)
        - None: trade in all regimes (standard behavior)
        - int: only generate entry signals when df['regime'] == target_regime
    signal_shift : int, number of days to delay signal execution (default=0)
        - 0: execute signals immediately
        - >0: delay signal execution by this many days (e.g., signal_shift=1 means trade 1 day after signal)

    Returns:
    --------
    DataFrame with signals, positions, PnL calculations, and portfolio tracking
    """
    df_signals = df.copy()
    # ============================================================
    # 1. Calculate Technical Indicators
    # ============================================================
    # Calculate RSI
    df_signals['rsi'] = mr_calculate_rsi(df_signals['Adj_Close'], period=rsi_period) 
    
    # ============================================================
    # 2. Generate Entry/Exit Signals
    # ============================================================
    # Entry Long: RSI crosses above oversold threshold (exits oversold zone - bullish reversal)
    # RSI[t-1] <= oversold AND RSI[t] > oversold
    df_signals['entry_long'] = (
        (df_signals['rsi'].shift(1) <= rsi_oversold) &
        (df_signals['rsi'] > rsi_oversold)
    )

    # Exit Long: RSI crosses above exit threshold (momentum normalizing/profit taking)
    # RSI[t-1] < rsi_exit_long AND RSI[t] >= rsi_exit_long
    df_signals['exit_long'] = (
        (df_signals['rsi'].shift(1) < rsi_exit_long) &
        (df_signals['rsi'] >= rsi_exit_long)
    )

    # Entry Short: RSI crosses below overbought threshold (exits overbought zone - bearish reversal)
    # RSI[t-1] >= overbought AND RSI[t] < overbought
    df_signals['entry_short'] = (
        (df_signals['rsi'].shift(1) >= rsi_overbought) &
        (df_signals['rsi'] < rsi_overbought)
    )

    # Exit Short: RSI crosses below exit threshold (momentum normalizing/profit taking)
    # RSI[t-1] > rsi_exit_short AND RSI[t] <= rsi_exit_short
    df_signals['exit_short'] = (
        (df_signals['rsi'].shift(1) > rsi_exit_short) &
        (df_signals['rsi'] <= rsi_exit_short)
    )

    # Apply signal shift if specified (shift signals forward to delay execution)
    if signal_shift > 0:
        df_signals['entry_long'] = df_signals['entry_long'].shift(signal_shift).fillna(False)
        df_signals['exit_long'] = df_signals['exit_long'].shift(signal_shift).fillna(False)
        df_signals['entry_short'] = df_signals['entry_short'].shift(signal_shift).fillna(False)
        df_signals['exit_short'] = df_signals['exit_short'].shift(signal_shift).fillna(False)

    # ============================================================
    # 3. Calculate Position State and Portfolio Tracking
    # ============================================================
    # Position states: 0 = flat, 1 = long, -1 = short
    positions = []
    shares_held = []
    cash_balance = []
    equity_value = []
    total_portfolio = []
    
    current_position = 0
    current_shares = 0
    current_cash = initial_cash

    for i in range(len(df_signals)):
        price = df_signals.iloc[i]['Adj_Close']

        # First (rsi_period-1) days: no positions (RSI lookback period)
        if i < rsi_period - 1:
            positions.append(0)
            shares_held.append(0)
            cash_balance.append(current_cash)
            equity_value.append(0)
            total_portfolio.append(current_cash)
            continue

        # Process exit signals first (priority)
        if current_position == 1 and df_signals.iloc[i]['exit_long']:
            # Exit long: sell shares
            current_cash += current_shares * price
            current_shares = 0
            current_position = 0
            
        elif current_position == -1 and df_signals.iloc[i]['exit_short']:
            # Exit short: buy back shares to cover
            current_cash += current_shares * price  # current_shares is negative for short
            current_shares = 0
            current_position = 0
            
        # Process entry signals (only if flat)
        elif current_position == 0:
            # Check if we should trade in this regime (if target_regime is specified)
            regime_check = True
            if target_regime is not None and 'regime' in df_signals.columns:
                current_regime = df_signals.iloc[i]['regime']
                # Only trade if regime matches (skip NaN regimes too)
                regime_check = (not pd.isna(current_regime)) and (current_regime == target_regime)
            
            if regime_check and df_signals.iloc[i]['entry_long']:
                # Enter long position
                if position_sizing_method == 'capital_based':
                    # Calculate max shares based on available capital
                    available_capital = current_cash * capital_allocation_pct
                    current_shares = int(available_capital / price)
                else:
                    # Use fixed position size
                    current_shares = position_size
                
                # Execute trade if we can afford at least 1 share
                if current_shares > 0:
                    current_cash -= current_shares * price
                    current_position = 1
                else:
                    current_shares = 0
                    
            elif regime_check and df_signals.iloc[i]['entry_short']:
                # Enter short position
                if position_sizing_method == 'capital_based':
                    # Calculate max shares based on available capital
                    available_capital = current_cash * capital_allocation_pct
                    current_shares = -int(available_capital / price)  # Negative for short
                else:
                    # Use fixed position size (negative for short)
                    current_shares = -position_size
                
                # Execute trade if we can afford at least 1 share
                if current_shares < 0:
                    current_cash -= current_shares * price  # Subtracting negative adds to cash
                    current_position = -1
                else:
                    current_shares = 0

        # Record state
        positions.append(current_position)
        shares_held.append(current_shares)
        cash_balance.append(current_cash)
        
        # Calculate equity value (mark-to-market)
        equity = current_shares * price
        equity_value.append(equity)
        total_portfolio.append(current_cash + equity)

    df_signals['signal'] = positions
    df_signals['shares'] = shares_held
    df_signals['cash'] = cash_balance
    df_signals['equity'] = equity_value
    df_signals['portfolio_value'] = total_portfolio

    # ============================================================
    # 4. Calculate Returns and PnL
    # ============================================================
    df_signals['returns'] = df_signals['Adj_Close'].pct_change()
    df_signals['position'] = df_signals['signal'].shift(1).fillna(0)
    df_signals['strategy_returns'] = df_signals['returns'] * df_signals['position']       

    # Calculate daily PnL from portfolio value changes
    df_signals['pnl'] = df_signals['portfolio_value'].diff()
    
    # Calculate cumulative PnL
    df_signals['cumulative_pnl'] = df_signals['portfolio_value'] - initial_cash

    return df_signals



#%%
# ============================================================
# RSI Visualization Function
# ============================================================
def mr_visualize_rsi_analysis(train_df, test_df, rsi_period=14,
                          rsi_oversold=30, rsi_overbought=70,
                          figsize=(16, 12), show_stats=True):
    """
    Comprehensive RSI visualization comparing train and test data.

    Parameters:
    -----------
    train_df : DataFrame with 'Adj_Close' column (training data)
    test_df : DataFrame with 'Adj_Close' column (test data)
    rsi_period : int, RSI calculation period (default=14)
    rsi_oversold : float, oversold threshold (default=30)
    rsi_overbought : float, overbought threshold (default=70)
    figsize : tuple, figure size (default=(16, 12))
    show_stats : bool, whether to print statistics (default=True)

    Returns:
    --------
    tuple: (train_df with RSI, test_df with RSI, figure)
    """
    import matplotlib.pyplot as plt

    # Calculate RSI for both datasets
    train_data = train_df.copy()
    test_data = test_df.copy()

    if show_stats:
        print("\n" + "="*70)
        print("CALCULATING RSI FOR TRAIN AND TEST DATA")
        print("="*70)

    train_data['rsi'] = mr_calculate_rsi(train_data['Adj_Close'], period=rsi_period)
    test_data['rsi'] = mr_calculate_rsi(test_data['Adj_Close'], period=rsi_period)

    if show_stats:
        print(f"Train RSI - Mean: {train_data['rsi'].mean():.2f}, Std: {train_data['rsi'].std():.2f}")
        print(f"Test RSI - Mean: {test_data['rsi'].mean():.2f}, Std: {test_data['rsi'].std():.2f}")

    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    fig.suptitle('RSI Analysis: Train vs Test Data', fontsize=16, fontweight='bold')

    # Plot 1: Train Data - Price and RSI
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(train_data.index, train_data['Adj_Close'],
             color='blue', linewidth=1.5, label='Price', alpha=0.7)
    ax1.set_ylabel('Price ($)', fontsize=10, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('TRAIN DATA: Price & RSI', fontsize=12, fontweight='bold')

    ax1_twin.plot(train_data.index, train_data['rsi'],
                  color='purple', linewidth=1, label='RSI', alpha=0.8)
    ax1_twin.axhline(y=rsi_overbought, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Overbought ({rsi_overbought})')
    ax1_twin.axhline(y=50, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, label='Neutral (50)')
    ax1_twin.axhline(y=rsi_oversold, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Oversold ({rsi_oversold})')
    ax1_twin.fill_between(train_data.index, rsi_overbought, 100, color='red', alpha=0.1)
    ax1_twin.fill_between(train_data.index, 0, rsi_oversold, color='green', alpha=0.1)
    ax1_twin.set_ylabel('RSI', fontsize=10, color='purple')
    ax1_twin.tick_params(axis='y', labelcolor='purple')
    ax1_twin.set_ylim(0, 100)
    ax1_twin.legend(loc='upper right', fontsize=8)

    # Plot 2: Test Data - Price and RSI
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.plot(test_data.index, test_data['Adj_Close'],
             color='blue', linewidth=1.5, label='Price', alpha=0.7)
    ax2.set_ylabel('Price ($)', fontsize=10, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('TEST DATA: Price & RSI', fontsize=12, fontweight='bold')

    ax2_twin.plot(test_data.index, test_data['rsi'],
                  color='purple', linewidth=1, label='RSI', alpha=0.8)
    ax2_twin.axhline(y=rsi_overbought, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Overbought ({rsi_overbought})')
    ax2_twin.axhline(y=50, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, label='Neutral (50)')
    ax2_twin.axhline(y=rsi_oversold, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Oversold ({rsi_oversold})')
    ax2_twin.fill_between(test_data.index, rsi_overbought, 100, color='red', alpha=0.1)
    ax2_twin.fill_between(test_data.index, 0, rsi_oversold, color='green', alpha=0.1)
    ax2_twin.set_ylabel('RSI', fontsize=10, color='purple')
    ax2_twin.tick_params(axis='y', labelcolor='purple')
    ax2_twin.set_ylim(0, 100)
    ax2_twin.legend(loc='upper right', fontsize=8)

    # Plot 3: RSI Distribution Comparison
    ax3 = axes[2]
    ax3.hist(train_data['rsi'].dropna(), bins=50, alpha=0.6,
             color='blue', label=f'Train (n={len(train_data)})', edgecolor='black')
    ax3.hist(test_data['rsi'].dropna(), bins=50, alpha=0.6,
             color='orange', label=f'Test (n={len(test_data)})', edgecolor='black')
    ax3.axvline(x=rsi_oversold, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Oversold ({rsi_oversold})')
    ax3.axvline(x=50, color='gray', linestyle='-', linewidth=1.5, alpha=0.7, label='Neutral (50)')
    ax3.axvline(x=rsi_overbought, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Overbought ({rsi_overbought})')
    ax3.set_xlabel('RSI Value', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('RSI Distribution: Train vs Test', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: RSI by Regime (if available)
    ax4 = axes[3]
    if 'regime' in train_data.columns and 'regime' in test_data.columns:
        train_regimes = train_data.groupby('regime')['rsi'].apply(list)
        test_regimes = test_data.groupby('regime')['rsi'].apply(list)

        positions_train = [i - 0.2 for i in range(len(train_regimes))]
        positions_test = [i + 0.2 for i in range(len(test_regimes))]

        bp1 = ax4.boxplot([train_regimes[i] for i in sorted(train_regimes.index)],
                           positions=positions_train, widths=0.35,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='darkblue', linewidth=2))
        bp2 = ax4.boxplot([test_regimes[i] for i in sorted(test_regimes.index)],
                           positions=positions_test, widths=0.35,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor='lightcoral', alpha=0.7),
                           medianprops=dict(color='darkred', linewidth=2))

        ax4.axhline(y=rsi_overbought, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axhline(y=50, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax4.axhline(y=rsi_oversold, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xticks(range(len(train_regimes)))
        ax4.set_xticklabels([f'Regime {int(i)}' for i in sorted(train_regimes.index)])
        ax4.set_ylabel('RSI Value', fontsize=10)
        ax4.set_title('RSI Distribution by Volatility Regime', fontsize=12, fontweight='bold')
        ax4.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Train', 'Test'], loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 100)
    else:
        ax4.text(0.5, 0.5, 'Regime data not available',
                 ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('RSI by Regime (Not Available)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Print RSI statistics
    if show_stats:
        print("\n" + "="*70)
        print("RSI STATISTICS")
        print("="*70)
        print("\nTRAIN DATA:")
        print(f"  Oversold (<{rsi_oversold}): {(train_data['rsi'] < rsi_oversold).sum()} days ({(train_data['rsi'] < rsi_oversold).sum()/len(train_data)*100:.1f}%)")
        print(f"  Neutral ({rsi_oversold}-{rsi_overbought}): {((train_data['rsi'] >= rsi_oversold) & (train_data['rsi'] <= rsi_overbought)).sum()} days ({((train_data['rsi'] >= rsi_oversold) & (train_data['rsi'] <= rsi_overbought)).sum()/len(train_data)*100:.1f}%)")
        print(f"  Overbought (>{rsi_overbought}): {(train_data['rsi'] > rsi_overbought).sum()} days ({(train_data['rsi'] > rsi_overbought).sum()/len(train_data)*100:.1f}%)")

        print("\nTEST DATA:")
        print(f"  Oversold (<{rsi_oversold}): {(test_data['rsi'] < rsi_oversold).sum()} days ({(test_data['rsi'] < rsi_oversold).sum()/len(test_data)*100:.1f}%)")
        print(f"  Neutral ({rsi_oversold}-{rsi_overbought}): {((test_data['rsi'] >= rsi_oversold) & (test_data['rsi'] <= rsi_overbought)).sum()} days ({((test_data['rsi'] >= rsi_oversold) & (test_data['rsi'] <= rsi_overbought)).sum()/len(test_data)*100:.1f}%)")
        print(f"  Overbought (>{rsi_overbought}): {(test_data['rsi'] > rsi_overbought).sum()} days ({(test_data['rsi'] > rsi_overbought).sum()/len(test_data)*100:.1f}%)")

    return train_data, test_data, fig




# %%
def mr_rsi_hyperparameter_test(df,
                       rsi_period_range=[10, 14, 20, 30],
                       rsi_oversold_range=[20, 25, 30, 35],
                       rsi_overbought_range=[65, 70, 75, 80],
                       rsi_exit_long_range=[45, 50, 55],
                       rsi_exit_short_range=[45, 50, 55],
                       position_sizing_methods=['fixed', 'capital_based'],
                       position_sizes=[150],
                       capital_allocation_pcts=[0.98],
                       initial_cash=100000,
                       sort_by='Sharpe Ratio',
                       top_n=20,
                       verbose=True,
                       n_jobs=-1,
                       target_regime=None):
    """
    Test multiple hyperparameter combinations for the RSI-based mean reversion strategy with PARALLEL PROCESSING.

    Parameters:
    -----------
    df : DataFrame with 'Adj_Close' column
    rsi_period_range : list of int, RSI calculation periods to test (default=[10, 14, 20, 30])
    rsi_oversold_range : list of float, RSI oversold thresholds for long entry (default=[20, 25, 30, 35])
    rsi_overbought_range : list of float, RSI overbought thresholds for short entry (default=[65, 70, 75, 80])
    rsi_exit_long_range : list of float, RSI exit thresholds for long positions (default=[45, 50, 55])
    rsi_exit_short_range : list of float, RSI exit thresholds for short positions (default=[45, 50, 55])
    position_sizing_methods : list of str, methods to test ['fixed', 'capital_based']
    position_sizes : list of int, position sizes for 'fixed' method
    capital_allocation_pcts : list of float, allocation % for 'capital_based' method
    initial_cash : float, starting capital
    sort_by : str, metric to sort results by (default='Sharpe Ratio')
    top_n : int, number of top results to return (default=20)
    verbose : bool, print progress updates
    n_jobs : int, number of parallel jobs (-1 uses all cores, 1 for sequential, default=-1)
    target_regime : int or None, if specified, only trade when current regime matches (default=None)

    Returns:
    --------
    DataFrame with results sorted by specified metric
    """
    import itertools
    from datetime import datetime
    from joblib import Parallel, delayed
    import io
    import sys
    
    results = []

    # Generate all parameter combinations for RSI strategy
    param_combinations = []

    for rsi_period in rsi_period_range:
        for rsi_oversold in rsi_oversold_range:
            for rsi_overbought in rsi_overbought_range:
                # Only test valid threshold combinations (overbought > oversold)
                if rsi_overbought > rsi_oversold:
                    for rsi_exit_long in rsi_exit_long_range:
                        for rsi_exit_short in rsi_exit_short_range:
                            # Validate exit thresholds are between entry thresholds
                            if rsi_oversold < rsi_exit_long < rsi_overbought and rsi_oversold < rsi_exit_short < rsi_overbought:
                                for method in position_sizing_methods:
                                    if method == 'fixed':
                                        for pos_size in position_sizes:
                                            param_combinations.append({
                                                'rsi_period': rsi_period,
                                                'rsi_oversold': rsi_oversold,
                                                'rsi_overbought': rsi_overbought,
                                                'rsi_exit_long': rsi_exit_long,
                                                'rsi_exit_short': rsi_exit_short,
                                                'position_sizing_method': method,
                                                'position_size': pos_size,
                                                'capital_allocation_pct': None
                                            })
                                    else:  # capital_based
                                        for cap_alloc in capital_allocation_pcts:
                                            param_combinations.append({
                                                'rsi_period': rsi_period,
                                                'rsi_oversold': rsi_oversold,
                                                'rsi_overbought': rsi_overbought,
                                                'rsi_exit_long': rsi_exit_long,
                                                'rsi_exit_short': rsi_exit_short,
                                                'position_sizing_method': method,
                                                'position_size': None,
                                                'capital_allocation_pct': cap_alloc
                                            })
    
    # Helper function to test a single parameter combination
    def test_single_combination(params):
        try:
            # Run RSI-based strategy
            if params['position_sizing_method'] == 'fixed':
                df_signals = mr_compute_mean_reversion_rsi_strategy(
                    df,
                    rsi_period=params['rsi_period'],
                    rsi_oversold=params['rsi_oversold'],
                    rsi_overbought=params['rsi_overbought'],
                    rsi_exit_long=params['rsi_exit_long'],
                    rsi_exit_short=params['rsi_exit_short'],
                    position_size=params['position_size'],
                    initial_cash=initial_cash,
                    position_sizing_method=params['position_sizing_method'],
                    target_regime=target_regime
                )
            else:  # capital_based
                df_signals = mr_compute_mean_reversion_rsi_strategy(
                    df,
                    rsi_period=params['rsi_period'],
                    rsi_oversold=params['rsi_oversold'],
                    rsi_overbought=params['rsi_overbought'],
                    rsi_exit_long=params['rsi_exit_long'],
                    rsi_exit_short=params['rsi_exit_short'],
                    initial_cash=initial_cash,
                    position_sizing_method=params['position_sizing_method'],
                    capital_allocation_pct=params['capital_allocation_pct'],
                    target_regime=target_regime
                )

            # Calculate metrics (suppress print output)
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            metrics = calculate_performance_metrics(df_signals, initial_cash=initial_cash)

            sys.stdout = old_stdout

            # Store results with RSI parameter names
            result = {
                'RSI_Period': params['rsi_period'],
                'RSI_Oversold': params['rsi_oversold'],
                'RSI_Overbought': params['rsi_overbought'],
                'RSI_Exit_Long': params['rsi_exit_long'],
                'RSI_Exit_Short': params['rsi_exit_short'],
                'Method': params['position_sizing_method'],
                'Position_Size': params['position_size'],
                'Capital_Alloc_Pct': params['capital_allocation_pct'],
                **metrics  # Add all metrics
            }
            return result

        except Exception as e:
            return None
    
    total_tests = len(param_combinations)
    if verbose:
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TEST - PARALLEL PROCESSING")
        print(f"{'='*70}")
        print(f"Testing {total_tests} parameter combinations...")
        print(f"Using {n_jobs if n_jobs > 0 else 'ALL'} CPU cores")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")
    
    # Run tests in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(test_single_combination)(params) for params in param_combinations
    )
    
    # Filter out None results (failed tests)
    results = [r for r in results if r is not None]
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\nCompleted at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Successful tests: {len(results_df)}/{total_tests}")
    
    # Sort by specified metric (descending for most metrics)
    if sort_by in results_df.columns:
        ascending = True if 'Drawdown' in sort_by else False  # Drawdown should be ascending (less negative is better)
        results_df = results_df.sort_values(by=sort_by, ascending=ascending)
    
    # Return top N results
    return results_df.head(top_n) if top_n else results_df


# %%
def mr_rsi_calculate_performance_metrics(df_signals, initial_cash=100000):
    """
    Calculate comprehensive performance metrics for the mean reversion strategy.

    Parameters:
    -----------
    df_signals : DataFrame returned from mr_compute_mean_reversion_rsi_strategy()
    initial_cash : float, starting capital (default=100000)

    Returns:
    --------
    Dictionary with performance metrics
    """
    # Calculate signal changes to identify individual trades
    df_signals['signal_change'] = df_signals['signal'].diff()

    # Identify trade periods (when position is held)
    df_long = df_signals[df_signals['signal'] == 1].copy()
    df_short = df_signals[df_signals['signal'] == -1].copy()

    # Calculate trade-level P&L by grouping consecutive positions
    trades = []
    current_signal = 0
    entry_idx = None

    for idx, row in df_signals.iterrows():
        if row['signal'] != current_signal:
            # Position changed - close previous trade if exists
            if current_signal != 0 and entry_idx is not None:
                exit_idx = idx
                trade_pnl = df_signals.loc[entry_idx:idx, 'pnl'].sum()
                trades.append({
                    'entry_date': entry_idx,
                    'exit_date': exit_idx,
                    'direction': 'Long' if current_signal == 1 else 'Short',
                    'pnl': trade_pnl
                })

            # Start new trade if entering position
            if row['signal'] != 0:
                entry_idx = idx
            current_signal = row['signal']

    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)

    # Separate by direction and win/loss
    if len(trades_df) > 0:
        winning_longs = trades_df[(trades_df['direction'] == 'Long') & (trades_df['pnl'] > 0)]
        losing_longs = trades_df[(trades_df['direction'] == 'Long') & (trades_df['pnl'] <= 0)]
        winning_shorts = trades_df[(trades_df['direction'] == 'Short') & (trades_df['pnl'] > 0)]
        losing_shorts = trades_df[(trades_df['direction'] == 'Short') & (trades_df['pnl'] <= 0)]
    else:
        winning_longs = losing_longs = winning_shorts = losing_shorts = pd.DataFrame()

    # Calculate total returns
    # Use portfolio_value if available (more accurate), otherwise calculate from pnl
    if 'portfolio_value' in df_signals.columns:
        final_value = df_signals['portfolio_value'].iloc[-1]
        total_pnl = final_value - initial_cash
    else:
        total_pnl = df_signals['pnl'].sum()
        final_value = initial_cash + total_pnl
    
    total_return_pct = (total_pnl / initial_cash) * 100

    # Calculate CAGR
    years = (df_signals.index[-1] - df_signals.index[0]).days / 365.25
    cagr = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Calculate Sharpe Ratio (annualized)
    daily_returns = df_signals['strategy_returns'].dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Calculate Max Drawdown
    cumulative_returns = (1 + df_signals['strategy_returns']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Calculate volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0

    # Calculate final P&L by direction
    final_long_pnl = (winning_longs['pnl'].sum() if len(winning_longs) > 0 else 0) + \
                     (losing_longs['pnl'].sum() if len(losing_longs) > 0 else 0)
    final_short_pnl = (winning_shorts['pnl'].sum() if len(winning_shorts) > 0 else 0) + \
                      (losing_shorts['pnl'].sum() if len(losing_shorts) > 0 else 0)

    # Compile metrics
    metrics = {
        'Total P&L ($)': total_pnl,
        'Total Return (%)': total_return_pct,
        'CAGR (%)': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Volatility (% annualized)': volatility,
        'Total Trades': len(trades_df),
        'Winning Longs': len(winning_longs),
        'Winning Longs P&L ($)': winning_longs['pnl'].sum() if len(winning_longs) > 0 else 0,
        'Losing Longs': len(losing_longs),
        'Losing Longs P&L ($)': losing_longs['pnl'].sum() if len(losing_longs) > 0 else 0,
        'Final Long P&L ($)': final_long_pnl,
        'Long Win Rate (%)': (len(winning_longs) / (len(winning_longs) + len(losing_longs)) * 100) if (len(winning_longs) + len(losing_longs)) > 0 else 0,
        'Winning Shorts': len(winning_shorts),
        'Winning Shorts P&L ($)': winning_shorts['pnl'].sum() if len(winning_shorts) > 0 else 0,
        'Losing Shorts': len(losing_shorts),
        'Losing Shorts P&L ($)': losing_shorts['pnl'].sum() if len(losing_shorts) > 0 else 0,
        'Final Short P&L ($)': final_short_pnl,
        'Short Win Rate (%)': (len(winning_shorts) / (len(winning_shorts) + len(losing_shorts)) * 100) if (len(winning_shorts) + len(losing_shorts)) > 0 else 0,
    }

    # Print formatted summary
    print(f"\n{'='*70}")
    print(f"{'PERFORMANCE METRICS SUMMARY':^70}")
    print(f"{'='*70}\n")

    print(f"{'OVERALL PERFORMANCE':-^70}")
    print(f"Total P&L:                ${metrics['Total P&L ($)']:>15,.2f}")
    print(f"Total Return:             {metrics['Total Return (%)']:>15,.2f}%")
    print(f"CAGR:                     {metrics['CAGR (%)']:>15,.2f}%")
    print(f"Sharpe Ratio:             {metrics['Sharpe Ratio']:>15,.3f}")
    print(f"Max Drawdown:             {metrics['Max Drawdown (%)']:>15,.2f}%")
    print(f"Volatility (annualized):  {metrics['Volatility (% annualized)']:>15,.2f}%")
    print(f"\n{'TRADE STATISTICS':-^70}")
    print(f"Total Trades:             {metrics['Total Trades']:>15,}")
    print(f"\n{'LONG POSITIONS':-^70}")
    print(f"Winning Longs:            {metrics['Winning Longs']:>15,} trades")
    print(f"Winning Longs P&L:        ${metrics['Winning Longs P&L ($)']:>15,.2f}")
    print(f"Losing Longs:             {metrics['Losing Longs']:>15,} trades")
    print(f"Losing Longs P&L:         ${metrics['Losing Longs P&L ($)']:>15,.2f}")
    print(f"Final Long P&L:           ${metrics['Final Long P&L ($)']:>15,.2f}")
    print(f"Long Win Rate:            {metrics['Long Win Rate (%)']:>15,.2f}%")
    print(f"\n{'SHORT POSITIONS':-^70}")
    print(f"Winning Shorts:           {metrics['Winning Shorts']:>15,} trades")
    print(f"Winning Shorts P&L:       ${metrics['Winning Shorts P&L ($)']:>15,.2f}")
    print(f"Losing Shorts:            {metrics['Losing Shorts']:>15,} trades")
    print(f"Losing Shorts P&L:        ${metrics['Losing Shorts P&L ($)']:>15,.2f}")
    print(f"Final Short P&L:          ${metrics['Final Short P&L ($)']:>15,.2f}")
    print(f"Short Win Rate:           {metrics['Short Win Rate (%)']:>15,.2f}%")
    print(f"\n{'='*70}\n")

    return metrics



# %%
def mr_viz_backtest_rsi_mean_reversion(df_signals, start_date=None, end_date=None):
    """
    Visualize RSI-based mean reversion strategy with entry/exit signals.

    Parameters:
    -----------
    df_signals : DataFrame returned from mr_compute_mean_reversion_rsi_strategy()
    start_date : str or datetime, optional start date for zoom (e.g., '2020-01-01')
    end_date : str or datetime, optional end date for zoom (e.g., '2020-12-31')
    """
    # Filter by date range if specified
    df_plot = df_signals.copy()
    if start_date is not None:
        df_plot = df_plot[df_plot.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot[df_plot.index <= pd.to_datetime(end_date)]

    # Create subplots: price on top, RSI on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 10), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1]})

    # ============= TOP PLOT: PRICE WITH SIGNALS =============
    ax1.plot(df_plot.index, df_plot["Adj_Close"], label="Price = Adjusted Close", linewidth=1.5)

    # Calculate signal changes (actual trade executions)
    df_plot['signal_change'] = df_plot['signal'].diff()
    df_plot['prev_signal'] = df_plot['signal'].shift(1)

    # LONG Entry: signal changes to 1 (0→1 or -1→1)
    long_entries = df_plot[(df_plot['signal'] == 1) & (df_plot['prev_signal'] != 1)]
    if len(long_entries) > 0:
        ax1.scatter(long_entries.index, long_entries["Adj_Close"],
                   label="Long Entry", marker="^", color="green", s=100, zorder=5)

    # LONG Exit: signal changes from 1 to something else
    long_exits = df_plot[(df_plot['prev_signal'] == 1) & (df_plot['signal'] != 1)]
    if len(long_exits) > 0:
        ax1.scatter(long_exits.index, long_exits["Adj_Close"],
                   label="Long Exit", marker="v", color="lightgreen", s=100, zorder=5)

    # SHORT Entry: signal changes to -1 (0→-1 or 1→-1)
    short_entries = df_plot[(df_plot['signal'] == -1) & (df_plot['prev_signal'] != -1)]
    if len(short_entries) > 0:
        ax1.scatter(short_entries.index, short_entries["Adj_Close"],
                   label="Short Entry", marker="v", color="red", s=100, zorder=5)

    # SHORT Exit: signal changes from -1 to something else
    short_exits = df_plot[(df_plot['prev_signal'] == -1) & (df_plot['signal'] != -1)]
    if len(short_exits) > 0:
        ax1.scatter(short_exits.index, short_exits["Adj_Close"],
                   label="Short Exit", marker="^", color="lightcoral", s=100, zorder=5)

    ax1.set_ylabel("Price", fontsize=12)
    ax1.set_title("RSI Mean Reversion Strategy - Backtesting with Entry & Exit Signals", fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # ============= BOTTOM PLOT: RSI WITH THRESHOLDS =============
    ax2.plot(df_plot.index, df_plot["rsi"], label="RSI", color="purple", linewidth=1.5)

    # Plot RSI threshold lines if they exist in the dataframe
    if 'rsi_oversold' in df_plot.columns:
        rsi_oversold = df_plot['rsi_oversold'].iloc[0]
        ax2.axhline(y=rsi_oversold, color='green', linestyle='--', linewidth=1.5,
                   label=f'Oversold (Entry Long): {rsi_oversold}')

    if 'rsi_overbought' in df_plot.columns:
        rsi_overbought = df_plot['rsi_overbought'].iloc[0]
        ax2.axhline(y=rsi_overbought, color='red', linestyle='--', linewidth=1.5,
                   label=f'Overbought (Entry Short): {rsi_overbought}')

    if 'rsi_exit_long' in df_plot.columns:
        rsi_exit_long = df_plot['rsi_exit_long'].iloc[0]
        ax2.axhline(y=rsi_exit_long, color='lightgreen', linestyle=':', linewidth=1.5,
                   label=f'Exit Long: {rsi_exit_long}')

    if 'rsi_exit_short' in df_plot.columns:
        rsi_exit_short = df_plot['rsi_exit_short'].iloc[0]
        ax2.axhline(y=rsi_exit_short, color='lightcoral', linestyle=':', linewidth=1.5,
                   label=f'Exit Short: {rsi_exit_short}')

    # Add reference lines at 30 and 70 (traditional RSI levels)
    ax2.axhline(y=30, color='gray', linestyle='-.', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=70, color='gray', linestyle='-.', linewidth=0.8, alpha=0.5)

    # Fill oversold/overbought regions
    ax2.fill_between(df_plot.index, 0, 30, alpha=0.1, color='green', label='Traditional Oversold')
    ax2.fill_between(df_plot.index, 70, 100, alpha=0.1, color='red', label='Traditional Overbought')

    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("RSI", fontsize=12)
    ax2.set_ylim([0, 100])
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



# %%
def mr_ris_viz_regime_aware_backtest_signals(df_results, start_date=None, end_date=None, regime_names_dict=None):
    """
    Visualize regime-aware RSI mean reversion strategy with regime backgrounds and entry/exit signals.

    Parameters:
    -----------
    df_results : DataFrame from compute_regime_aware_strategy()
    start_date : str or datetime, optional start date for zoom (e.g., '2020-01-01')
    end_date : str or datetime, optional end date for zoom (e.g., '2020-12-31')
    regime_names_dict : dict mapping regime IDs to names
    """
    # Filter by date range if specified
    df_plot = df_results.copy()
    if start_date is not None:
        df_plot = df_plot[df_plot.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot[df_plot.index <= pd.to_datetime(end_date)]

    # Define regime colors (gradient from calm to crisis)
    regime_colors = {
        0: 'lightgreen',    # Very Low Vol: Green (calm)
        1: 'lightblue',     # Low Vol: Blue (stable)
        2: 'lightyellow',   # Med Vol: Yellow (normal)
        3: 'orange',        # High Vol: Orange (elevated)
        4: 'lightcoral'     # Very High Vol: Red (crisis)
    }

    # Create 2-panel figure: Price chart on top, RSI indicator on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [2, 1]})

    # ============================================================
    # TOP PANEL: Price Chart with Entry/Exit Signals
    # ============================================================
    # Add regime background first (behind everything)
    if 'regime' in df_plot.columns:
        for regime in df_plot['regime'].dropna().unique():
            regime_mask = df_plot['regime'] == regime
            regime_label = regime_names_dict.get(int(regime), f'Regime {int(regime)}') if regime_names_dict else f'Regime {int(regime)}'
            ax1.fill_between(df_plot.index,
                           df_plot["Adj_Close"].min() * 0.9,
                           df_plot["Adj_Close"].max() * 1.1,
                           where=regime_mask,
                           alpha=0.2,
                           color=regime_colors.get(int(regime), 'lightgray'),
                           label=regime_label)

    # Plot price
    ax1.plot(df_plot.index, df_plot["Adj_Close"], label="SPY Price", linewidth=2, color='black', zorder=3)
    
    # Calculate signal changes (actual trade executions)
    df_plot['signal_change'] = df_plot['signal'].diff()
    df_plot['prev_signal'] = df_plot['signal'].shift(1)

    # LONG Entry: signal changes to 1
    long_entries = df_plot[(df_plot['signal'] == 1) & (df_plot['prev_signal'] != 1)]
    if len(long_entries) > 0:
        ax1.scatter(long_entries.index, long_entries["Adj_Close"],
                   label="LONG Entry", marker="^", s=150, color="darkgreen",
                   edgecolors='black', linewidths=1.5, zorder=5)

    # LONG Exit: signal changes from 1 to something else
    long_exits = df_plot[(df_plot['prev_signal'] == 1) & (df_plot['signal'] != 1)]
    if len(long_exits) > 0:
        ax1.scatter(long_exits.index, long_exits["Adj_Close"],
                   label="LONG Exit", marker="v", s=150, color="lightgreen",
                   edgecolors='black', linewidths=1.5, zorder=5)

    # SHORT Entry: signal changes to -1
    short_entries = df_plot[(df_plot['signal'] == -1) & (df_plot['prev_signal'] != -1)]
    if len(short_entries) > 0:
        ax1.scatter(short_entries.index, short_entries["Adj_Close"],
                   label="SHORT Entry", marker="v", s=150, color="darkred",
                   edgecolors='black', linewidths=1.5, zorder=5)

    # SHORT Exit: signal changes from -1 to something else
    short_exits = df_plot[(df_plot['prev_signal'] == -1) & (df_plot['signal'] != -1)]
    if len(short_exits) > 0:
        ax1.scatter(short_exits.index, short_exits["Adj_Close"],
                   label="SHORT Exit", marker="^", s=150, color="lightcoral",
                   edgecolors='black', linewidths=1.5, zorder=5)

    # Top panel formatting
    date_range = f"{df_plot.index[0].strftime('%Y-%m-%d')} to {df_plot.index[-1].strftime('%Y-%m-%d')}"
    ax1.set_title(f"Regime-Aware RSI Mean Reversion Strategy - Entry/Exit Signals\n{date_range}",
                 fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='best', fontsize=10, ncol=2)
    ax1.grid(alpha=0.3)

    # ============================================================
    # BOTTOM PANEL: RSI Indicator with Threshold Lines
    # ============================================================
    # Add regime background to RSI panel
    if 'regime' in df_plot.columns:
        for regime in df_plot['regime'].dropna().unique():
            regime_mask = df_plot['regime'] == regime
            ax2.fill_between(df_plot.index, 0, 100,
                           where=regime_mask,
                           alpha=0.2,
                           color=regime_colors.get(int(regime), 'lightgray'))

    # Plot RSI
    if 'rsi' in df_plot.columns:
        ax2.plot(df_plot.index, df_plot['rsi'], label='RSI', linewidth=2, color='purple', zorder=3)

    # Plot standard RSI reference levels
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Neutral (50)')

    # Mark signal points on RSI panel
    if len(long_entries) > 0:
        ax2.scatter(long_entries.index, long_entries['rsi'],
                   marker="^", s=100, color="darkgreen", edgecolors='black', linewidths=1, zorder=5)
    if len(long_exits) > 0:
        ax2.scatter(long_exits.index, long_exits['rsi'],
                   marker="v", s=100, color="lightgreen", edgecolors='black', linewidths=1, zorder=5)
    if len(short_entries) > 0:
        ax2.scatter(short_entries.index, short_entries['rsi'],
                   marker="v", s=100, color="darkred", edgecolors='black', linewidths=1, zorder=5)
    if len(short_exits) > 0:
        ax2.scatter(short_exits.index, short_exits['rsi'],
                   marker="^", s=100, color="lightcoral", edgecolors='black', linewidths=1, zorder=5)

    # Bottom panel formatting
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    # Print signal summary
    print("\n" + "="*70)
    print("SIGNAL SUMMARY")
    print("="*70)
    print(f"Long Entries:  {len(long_entries)}")
    print(f"Long Exits:    {len(long_exits)}")
    print(f"Short Entries: {len(short_entries)}")
    print(f"Short Exits:   {len(short_exits)}")
    print(f"Total Signals: {len(long_entries) + len(long_exits) + len(short_entries) + len(short_exits)}")
    print("="*70)



# %%
def mr_rsi_viz_backtest_returns(df_signals, initial_cash=100000):
    """
    Visualize cumulative returns: strategy vs buy-and-hold.

    Parameters:
    -----------
    df_signals : DataFrame returned from mr_compute_mean_reversion_rsi_strategy()
    initial_cash : float, starting capital (default=100000)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[17, 10])

    # Calculate cumulative returns
    df_plot = df_signals.copy()

    # Strategy cumulative returns
    df_plot['cum_strategy_returns'] = (1 + df_plot['strategy_returns']).cumprod() - 1

    # Buy-and-hold cumulative returns
    df_plot['cum_asset_returns'] = (1 + df_plot['returns']).cumprod() - 1

    # Calculate running maximum for drawdown
    strategy_cummax = (1 + df_plot['strategy_returns']).cumprod().expanding().max()
    df_plot['drawdown'] = ((1 + df_plot['strategy_returns']).cumprod() - strategy_cummax) / strategy_cummax

    # Plot 1: Cumulative Returns
    df_plot[['cum_asset_returns', 'cum_strategy_returns']].dropna().plot(ax=ax1, linewidth=1.5)

    # Format y-axis as percentage
    from matplotlib.ticker import PercentFormatter
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax1.legend(['Buy & Hold', 'Mean Reversion Strategy'], loc='best')
    ax1.set_title('Cumulative Returns: Strategy vs Buy-and-Hold', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(alpha=0.3)

    # Plot 2: Drawdown
    df_plot['drawdown'].plot(ax=ax2, color='red', linewidth=1.5)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax2.set_title('Strategy Drawdown', fontsize=14)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.grid(alpha=0.3)
    ax2.fill_between(df_plot.index, df_plot['drawdown'], 0, color='red', alpha=0.3)

    plt.tight_layout()
    plt.show()



# %%
def mr_rsi_viz_regime_aware_backtest_comparison(train_results, test_results, regime_names_dict=None):
    """
    Visualize train vs test backtest returns with regime overlay.
    
    Parameters:
    -----------
    train_results : DataFrame from compute_regime_aware_strategy() for training data
    test_results : DataFrame from compute_regime_aware_strategy() for test data
    regime_names_dict : dict mapping regime IDs to names
    """
    fig, axes = plt.subplots(3, 2, figsize=(24, 16))
    
    # Define regime colors (gradient from calm to crisis)
    regime_colors = {
        0: 'lightgreen',    # Very Low Vol: Green (calm)
        1: 'lightblue',     # Low Vol: Blue (stable)
        2: 'lightyellow',   # Med Vol: Yellow (normal)
        3: 'orange',        # High Vol: Orange (elevated)
        4: 'lightcoral'     # Very High Vol: Red (crisis)
    }
    
    # ============================================================
    # TRAINING DATA (Left Column)
    # ============================================================
    
    # 1. Training: Cumulative Returns with Regime Background
    ax_train_returns = axes[0, 0]
    train_cum_strategy = (1 + train_results['strategy_returns']).cumprod() - 1
    train_cum_bh = (1 + train_results['returns_simple']).cumprod() - 1 if 'returns_simple' in train_results.columns else None
    
    # Add regime background
    if 'regime' in train_results.columns:
        for regime in train_results['regime'].dropna().unique():
            regime_mask = train_results['regime'] == regime
            ax_train_returns.fill_between(train_results.index,
                                         train_cum_strategy.min() * 1.1,
                                         train_cum_strategy.max() * 1.1,
                                         where=regime_mask,
                                         alpha=0.15,
                                         color=regime_colors.get(int(regime), 'lightgray'))
    
    ax_train_returns.plot(train_results.index, train_cum_strategy * 100,
                         linewidth=2, color='blue', label='Strategy')
    if train_cum_bh is not None:
        ax_train_returns.plot(train_results.index, train_cum_bh * 100,
                             linewidth=2, color='gray', alpha=0.7, label='Buy & Hold')
    
    ax_train_returns.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_train_returns.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax_train_returns.set_title('TRAINING: Cumulative Returns by Regime', fontsize=14, fontweight='bold')
    ax_train_returns.legend(loc='upper left', fontsize=10)
    ax_train_returns.grid(alpha=0.3)
    
    # 2. Training: Drawdown
    ax_train_dd = axes[1, 0]
    train_cummax = (1 + train_results['strategy_returns']).cumprod().expanding().max()
    train_dd = ((1 + train_results['strategy_returns']).cumprod() - train_cummax) / train_cummax
    
    ax_train_dd.plot(train_results.index, train_dd * 100, linewidth=2, color='red')
    ax_train_dd.fill_between(train_results.index, train_dd * 100, 0, color='red', alpha=0.3)
    ax_train_dd.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_train_dd.set_ylabel('Drawdown (%)', fontsize=12)
    ax_train_dd.set_title('TRAINING: Strategy Drawdown', fontsize=14, fontweight='bold')
    ax_train_dd.grid(alpha=0.3)
    
    # 3. Training: Portfolio Value
    ax_train_pv = axes[2, 0]
    ax_train_pv.plot(train_results.index, train_results['portfolio_value'],
                    linewidth=2, color='green')
    ax_train_pv.axhline(y=100000, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax_train_pv.fill_between(train_results.index, 100000, train_results['portfolio_value'],
                            where=(train_results['portfolio_value'] >= 100000),
                            alpha=0.3, color='green')
    ax_train_pv.fill_between(train_results.index, 100000, train_results['portfolio_value'],
                            where=(train_results['portfolio_value'] < 100000),
                            alpha=0.3, color='red')
    ax_train_pv.set_xlabel('Date', fontsize=12)
    ax_train_pv.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax_train_pv.set_title('TRAINING: Portfolio Growth', fontsize=14, fontweight='bold')
    ax_train_pv.grid(alpha=0.3)
    
    # ============================================================
    # TEST DATA (Right Column)
    # ============================================================
    
    # 1. Test: Cumulative Returns with Regime Background
    ax_test_returns = axes[0, 1]
    test_cum_strategy = (1 + test_results['strategy_returns']).cumprod() - 1
    test_cum_bh = (1 + test_results['returns_simple']).cumprod() - 1 if 'returns_simple' in test_results.columns else None
    
    # Add regime background
    if 'regime' in test_results.columns:
        for regime in test_results['regime'].dropna().unique():
            regime_mask = test_results['regime'] == regime
            regime_label = regime_names_dict.get(int(regime), f'Regime {int(regime)}') if regime_names_dict else f'Regime {int(regime)}'
            ax_test_returns.fill_between(test_results.index,
                                        test_cum_strategy.min() * 1.1,
                                        test_cum_strategy.max() * 1.1,
                                        where=regime_mask,
                                        alpha=0.15,
                                        color=regime_colors.get(int(regime), 'lightgray'),
                                        label=regime_label)
    
    ax_test_returns.plot(test_results.index, test_cum_strategy * 100,
                        linewidth=2, color='blue', label='Strategy')
    if test_cum_bh is not None:
        ax_test_returns.plot(test_results.index, test_cum_bh * 100,
                            linewidth=2, color='gray', alpha=0.7, label='Buy & Hold')
    
    ax_test_returns.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_test_returns.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax_test_returns.set_title('TEST: Cumulative Returns by Regime (Out-of-Sample)', fontsize=14, fontweight='bold')
    ax_test_returns.legend(loc='upper left', fontsize=10)
    ax_test_returns.grid(alpha=0.3)
    
    # 2. Test: Drawdown
    ax_test_dd = axes[1, 1]
    test_cummax = (1 + test_results['strategy_returns']).cumprod().expanding().max()
    test_dd = ((1 + test_results['strategy_returns']).cumprod() - test_cummax) / test_cummax
    
    ax_test_dd.plot(test_results.index, test_dd * 100, linewidth=2, color='red')
    ax_test_dd.fill_between(test_results.index, test_dd * 100, 0, color='red', alpha=0.3)
    ax_test_dd.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_test_dd.set_ylabel('Drawdown (%)', fontsize=12)
    ax_test_dd.set_title('TEST: Strategy Drawdown', fontsize=14, fontweight='bold')
    ax_test_dd.grid(alpha=0.3)
    
    # 3. Test: Portfolio Value
    ax_test_pv = axes[2, 1]
    ax_test_pv.plot(test_results.index, test_results['portfolio_value'],
                   linewidth=2, color='green')
    ax_test_pv.axhline(y=100000, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax_test_pv.fill_between(test_results.index, 100000, test_results['portfolio_value'],
                           where=(test_results['portfolio_value'] >= 100000),
                           alpha=0.3, color='green')
    ax_test_pv.fill_between(test_results.index, 100000, test_results['portfolio_value'],
                           where=(test_results['portfolio_value'] < 100000),
                           alpha=0.3, color='red')
    ax_test_pv.set_xlabel('Date', fontsize=12)
    ax_test_pv.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax_test_pv.set_title('TEST: Portfolio Growth', fontsize=14, fontweight='bold')
    ax_test_pv.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Calculate SPY (buy-and-hold) drawdowns
    if train_cum_bh is not None:
        train_spy_cummax = (1 + train_results['returns_simple']).cumprod().expanding().max()
        train_spy_dd = ((1 + train_results['returns_simple']).cumprod() - train_spy_cummax) / train_spy_cummax
        train_spy_final_value = 100000 * (1 + train_cum_bh.iloc[-1])
    else:
        train_spy_dd = pd.Series([0])
        train_spy_final_value = 100000

    if test_cum_bh is not None:
        test_spy_cummax = (1 + test_results['returns_simple']).cumprod().expanding().max()
        test_spy_dd = ((1 + test_results['returns_simple']).cumprod() - test_spy_cummax) / test_spy_cummax
        test_spy_final_value = 100000 * (1 + test_cum_bh.iloc[-1])
    else:
        test_spy_dd = pd.Series([0])
        test_spy_final_value = 100000

    # Print summary statistics
    print("\n" + "="*100)
    print("SIDE-BY-SIDE COMPARISON SUMMARY")
    print("="*100)
    print(f"\n{'Metric':<30} {'Strategy (Train)':>18} {'Strategy (Test)':>18} {'SPY B&H (Train)':>18} {'SPY B&H (Test)':>18}")
    print("-"*100)
    print(f"{'Final Return (%)':<30} {train_cum_strategy.iloc[-1]*100:>18.2f} {test_cum_strategy.iloc[-1]*100:>18.2f} {train_cum_bh.iloc[-1]*100 if train_cum_bh is not None else 0:>18.2f} {test_cum_bh.iloc[-1]*100 if test_cum_bh is not None else 0:>18.2f}")
    print(f"{'Max Drawdown (%)':<30} {train_dd.min()*100:>18.2f} {test_dd.min()*100:>18.2f} {train_spy_dd.min()*100:>18.2f} {test_spy_dd.min()*100:>18.2f}")
    print(f"{'Final Portfolio ($)':<30} ${train_results['portfolio_value'].iloc[-1]:>17,.2f} ${test_results['portfolio_value'].iloc[-1]:>17,.2f} ${train_spy_final_value:>17,.2f} ${test_spy_final_value:>17,.2f}")
    print("="*100)


# %%
# ============================================================
# REGIME-AWARE STRATEGY WITH DYNAMIC PARAMETERS
# ============================================================

def mr_rsi_compute_regime_aware_strategy(df_with_regimes, regime_params_dict,
                                  initial_cash=100000, position_sizing_method='capital_based',
                                  capital_allocation_pct=0.98, signal_shift=0):
    """
    Compute RSI mean reversion strategy with DYNAMIC parameters based on current regime.

    This handles regime transitions properly - parameters change as regime changes!

    Parameters:
    -----------
    df_with_regimes : DataFrame with 'Adj_Close' and 'regime' columns
    regime_params_dict : dict, format: {regime_id: {'RSI_Period': X, 'RSI_Oversold': Y, ...}}
        Example: {0: {'RSI_Period': 14, 'RSI_Oversold': 30, 'RSI_Overbought': 70,
                      'RSI_Exit_Long': 50, 'RSI_Exit_Short': 50},
                  1: {'RSI_Period': 20, 'RSI_Oversold': 25, 'RSI_Overbought': 75,
                      'RSI_Exit_Long': 55, 'RSI_Exit_Short': 45}}
    initial_cash : float, starting capital (default=100000)
    position_sizing_method : str, 'fixed' or 'capital_based' (default='capital_based')
    capital_allocation_pct : float, % of capital to use per trade (default=0.98)
    signal_shift : int, number of days to delay signal execution (default=0)
        - 0: execute signals immediately
        - >0: delay signal execution by this many days

    Returns:
    --------
    DataFrame with signals, positions, P&L calculations, and portfolio tracking
    """
    df_signals = df_with_regimes.copy()

    # Step 1: Calculate RSI for each unique period used across regimes
    # (Following the pattern from mr_compute_mean_reversion_rsi_strategy - calculate indicators first)
    unique_periods = set[int](int(params['RSI_Period']) for params in regime_params_dict.values())

    # Calculate RSI for each unique period
    for period in unique_periods:
        col_name = f'rsi_{period}'
        df_signals[col_name] = mr_calculate_rsi(df_signals['Adj_Close'], period=period)

    # Initialize signal columns
    df_signals['entry_long'] = False
    df_signals['exit_long'] = False
    df_signals['entry_short'] = False
    df_signals['exit_short'] = False
    df_signals['rsi'] = np.nan  # Store the regime-specific RSI being used

    # Step 2: Generate signals bar-by-bar using regime-specific thresholds
    # (Following the original pattern but with regime-aware threshold selection)
    for i in range(len(df_signals)):
        # Get current regime (use regime 0 params if NaN)
        current_regime = df_signals['regime'].iloc[i]
        if pd.isna(current_regime):
            current_regime = 0  # Default to first regime if missing

        # Get regime-specific parameters
        params = regime_params_dict.get(int(current_regime), regime_params_dict[0])
        rsi_period = int(params['RSI_Period'])
        rsi_oversold = float(params['RSI_Oversold'])
        rsi_overbought = float(params['RSI_Overbought'])
        rsi_exit_long = float(params['RSI_Exit_Long'])
        rsi_exit_short = float(params['RSI_Exit_Short'])

        # Get the RSI value for this regime's period
        rsi_col = f'rsi_{rsi_period}'
        current_rsi = df_signals[rsi_col].iloc[i]
        df_signals.loc[df_signals.index[i], 'rsi'] = current_rsi

        # Generate signals (need previous RSI for crossover detection)
        if i > 0:
            # Get previous regime's parameters to get previous RSI
            prev_regime = df_signals['regime'].iloc[i-1]
            if pd.isna(prev_regime):
                prev_regime = 0
            prev_params = regime_params_dict.get(int(prev_regime), regime_params_dict[0])
            prev_rsi_period = int(prev_params['RSI_Period'])
            prev_rsi_col = f'rsi_{prev_rsi_period}'
            prev_rsi = df_signals[prev_rsi_col].iloc[i-1]

            # Entry Long: RSI crosses UP through oversold threshold
            df_signals.loc[df_signals.index[i], 'entry_long'] = (
                (prev_rsi <= rsi_oversold) and (current_rsi > rsi_oversold)
            )

            # Exit Long: RSI crosses UP through exit threshold
            df_signals.loc[df_signals.index[i], 'exit_long'] = (
                (prev_rsi < rsi_exit_long) and (current_rsi >= rsi_exit_long)
            )

            # Entry Short: RSI crosses DOWN through overbought threshold
            df_signals.loc[df_signals.index[i], 'entry_short'] = (
                (prev_rsi >= rsi_overbought) and (current_rsi < rsi_overbought)
            )

            # Exit Short: RSI crosses DOWN through exit threshold
            df_signals.loc[df_signals.index[i], 'exit_short'] = (
                (prev_rsi > rsi_exit_short) and (current_rsi <= rsi_exit_short)
            )

    # Apply signal shift if specified (shift signals forward to delay execution)
    if signal_shift > 0:
        df_signals['entry_long'] = df_signals['entry_long'].shift(signal_shift).fillna(False)
        df_signals['exit_long'] = df_signals['exit_long'].shift(signal_shift).fillna(False)
        df_signals['entry_short'] = df_signals['entry_short'].shift(signal_shift).fillna(False)
        df_signals['exit_short'] = df_signals['exit_short'].shift(signal_shift).fillna(False)

    # ============================================================
    # Calculate Position State and Portfolio Tracking
    # ============================================================
    positions = []
    shares_held = []
    cash_balance = []
    portfolio_values = []
    pnl_daily = []
    
    position = 0  # 0 = flat, 1 = long, -1 = short
    shares = 0
    cash = initial_cash
    entry_price = 0
    
    for i in range(len(df_signals)):
        price = df_signals['Adj_Close'].iloc[i]
        
        # Default to previous state
        new_position = position
        new_shares = shares
        new_cash = cash
        daily_pnl = 0
        
        # Process signals
        if df_signals['entry_long'].iloc[i] and position <= 0:
            # Close short if exists
            cash_after_closing = cash
            if position == -1:
                cover_cost = shares * price
                daily_pnl = shares * (entry_price - price)  # Short P&L (for tracking)
                cash_after_closing = cash - cover_cost  # P&L already implicit in cash flow
                
            # Enter long (use cash after closing short if any)
            if position_sizing_method == 'capital_based':
                shares_to_buy = int((cash_after_closing * capital_allocation_pct) / price)
            else:
                shares_to_buy = 150  # Default position size
            
            cost = shares_to_buy * price
            if cost <= cash_after_closing:
                new_shares = shares_to_buy
                new_cash = cash_after_closing - cost
                new_position = 1
                entry_price = price
                if position != -1:  # Don't double count if closed short
                    daily_pnl = 0
                    
        elif df_signals['exit_long'].iloc[i] and position == 1:
            # Exit long
            proceeds = shares * price
            daily_pnl = shares * (price - entry_price)
            new_cash = cash + proceeds
            new_position = 0
            new_shares = 0
            entry_price = 0
            
        elif df_signals['entry_short'].iloc[i] and position >= 0:
            # Close long if exists
            cash_from_closing = 0
            if position == 1:
                proceeds = shares * price
                daily_pnl = shares * (price - entry_price)  # Long P&L
                cash_from_closing = proceeds
                
            # Enter short (use cash + proceeds from closing long if any)
            available_cash = cash + cash_from_closing
            if position_sizing_method == 'capital_based':
                shares_to_short = int((available_cash * capital_allocation_pct) / price)
            else:
                shares_to_short = 150
            
            if shares_to_short > 0:
                proceeds_from_short = shares_to_short * price
                new_shares = shares_to_short
                new_cash = available_cash + proceeds_from_short
                new_position = -1
                entry_price = price
                if position != 1:  # Don't double count if closed long
                    daily_pnl = 0
                    
        elif df_signals['exit_short'].iloc[i] and position == -1:
            # Exit short: buy back shares to cover
            cover_cost = shares * price
            daily_pnl = shares * (entry_price - price)  # For tracking only
            new_cash = cash - cover_cost  # P&L already implicit in cash flow
            new_position = 0
            new_shares = 0
            entry_price = 0
        
        # Update state
        position = new_position
        shares = new_shares
        cash = new_cash
        
        # Calculate portfolio value
        if position == 1:
            portfolio_value = cash + shares * price
        elif position == -1:
            portfolio_value = cash - shares * price
        else:
            portfolio_value = cash
        
        # Store results
        positions.append(position)
        shares_held.append(shares)
        cash_balance.append(cash)
        portfolio_values.append(portfolio_value)
        pnl_daily.append(daily_pnl)
    
    # Add to dataframe
    df_signals['signal'] = positions
    df_signals['shares'] = shares_held
    df_signals['cash'] = cash_balance
    df_signals['portfolio_value'] = portfolio_values
    df_signals['pnl'] = pnl_daily
    
    # Calculate strategy returns
    df_signals['strategy_returns'] = df_signals['portfolio_value'].pct_change().fillna(0)
    
    return df_signals

# %%
# ============================================================
# REGIME-AWARE STRATEGY V2 - EXACT MIRROR OF STANDARD STRATEGY
# ============================================================

def mr_rsi_compute_regime_aware_strategy_v2(df_with_regimes, regime_params_dict,
                                    position_size=150, initial_cash=100000,
                                    position_sizing_method='capital_based',
                                    capital_allocation_pct=0.98, signal_shift=0):
    """
    Compute regime-aware RSI mean reversion strategy that EXACTLY mirrors mr_compute_mean_reversion_rsi_strategy.

    The ONLY difference: RSI parameters change dynamically based on current regime.
    Position tracking, cash management, and signal logic are IDENTICAL to standard version.

    Parameters:
    -----------
    df_with_regimes : DataFrame with 'Adj_Close' and 'regime' columns
    regime_params_dict : dict, format: {regime_id: {'RSI_Period': X, 'RSI_Oversold': Y, ...}}
        Example: {0: {'RSI_Period': 14, 'RSI_Oversold': 30, 'RSI_Overbought': 70,
                      'RSI_Exit_Long': 50, 'RSI_Exit_Short': 50},
                  1: {'RSI_Period': 20, 'RSI_Oversold': 25, 'RSI_Overbought': 75,
                      'RSI_Exit_Long': 55, 'RSI_Exit_Short': 45}}
    position_size : int, number of shares per trade when using 'fixed' method (default=150)
    initial_cash : float, starting capital (default=100000)
    position_sizing_method : str, 'fixed' or 'capital_based' (default='capital_based')
    capital_allocation_pct : float, % of capital to use per trade (default=0.98)
    signal_shift : int, number of days to delay signal execution (default=0)
        - 0: execute signals immediately
        - >0: delay signal execution by this many days (e.g., signal_shift=1 means trade 1 day after signal)

    Returns:
    --------
    DataFrame with signals, positions, P&L calculations (identical structure to standard strategy)
    """
    df_signals = df_with_regimes.copy()

    # Get max RSI period for warmup period
    max_window = max(int(params['RSI_Period']) for params in regime_params_dict.values())
    
    # ============================================================
    # 1. Calculate RSI Indicators for each unique period
    # (Following the pattern from mr_compute_mean_reversion_rsi_strategy)
    # ============================================================
    # Get unique RSI periods used across regimes
    unique_periods = set(int(params['RSI_Period']) for params in regime_params_dict.values())

    # Calculate RSI for each unique period
    for period in unique_periods:
        col_name = f'rsi_{period}'
        df_signals[col_name] = mr_calculate_rsi(df_signals['Adj_Close'], period=period)

    # Initialize signal columns
    df_signals['entry_long'] = False
    df_signals['exit_long'] = False
    df_signals['entry_short'] = False
    df_signals['exit_short'] = False
    df_signals['rsi'] = np.nan  # Store the regime-specific RSI being used
    
    # ============================================================
    # 2. Generate Entry/Exit Signals (using regime-specific thresholds)
    # ============================================================
    # Generate signals bar-by-bar using regime-specific thresholds
    for i in range(len(df_signals)):
        # Get current regime (default to 0 if NaN)
        current_regime = df_signals['regime'].iloc[i]
        if pd.isna(current_regime):
            current_regime = 0

        # Get regime-specific parameters
        params = regime_params_dict.get(int(current_regime), regime_params_dict[0])
        rsi_period = int(params['RSI_Period'])
        rsi_oversold = float(params['RSI_Oversold'])
        rsi_overbought = float(params['RSI_Overbought'])
        rsi_exit_long = float(params['RSI_Exit_Long'])
        rsi_exit_short = float(params['RSI_Exit_Short'])

        # Get the RSI value for this regime's period
        rsi_col = f'rsi_{rsi_period}'
        current_rsi = df_signals[rsi_col].iloc[i]
        df_signals.loc[df_signals.index[i], 'rsi'] = current_rsi

        # Generate signals (need previous RSI for crossover detection)
        if i > 0:
            # Get previous regime's parameters to get previous RSI
            prev_regime = df_signals['regime'].iloc[i-1]
            if pd.isna(prev_regime):
                prev_regime = 0
            prev_params = regime_params_dict.get(int(prev_regime), regime_params_dict[0])
            prev_rsi_period = int(prev_params['RSI_Period'])
            prev_rsi_col = f'rsi_{prev_rsi_period}'
            prev_rsi = df_signals[prev_rsi_col].iloc[i-1]

            # Entry Long: RSI crosses UP through oversold threshold
            df_signals.loc[df_signals.index[i], 'entry_long'] = (
                (prev_rsi <= rsi_oversold) and (current_rsi > rsi_oversold)
            )

            # Exit Long: RSI crosses UP through exit threshold
            df_signals.loc[df_signals.index[i], 'exit_long'] = (
                (prev_rsi < rsi_exit_long) and (current_rsi >= rsi_exit_long)
            )

            # Entry Short: RSI crosses DOWN through overbought threshold
            df_signals.loc[df_signals.index[i], 'entry_short'] = (
                (prev_rsi >= rsi_overbought) and (current_rsi < rsi_overbought)
            )

            # Exit Short: RSI crosses DOWN through exit threshold
            df_signals.loc[df_signals.index[i], 'exit_short'] = (
                (prev_rsi > rsi_exit_short) and (current_rsi <= rsi_exit_short)
            )

    # Apply signal shift if specified (shift signals forward to delay execution)
    if signal_shift > 0:
        df_signals['entry_long'] = df_signals['entry_long'].shift(signal_shift).fillna(False)
        df_signals['exit_long'] = df_signals['exit_long'].shift(signal_shift).fillna(False)
        df_signals['entry_short'] = df_signals['entry_short'].shift(signal_shift).fillna(False)
        df_signals['exit_short'] = df_signals['exit_short'].shift(signal_shift).fillna(False)

    # ============================================================
    # 3. Calculate Position State and Portfolio Tracking
    # (EXACT MIRROR of mr_compute_mean_reversion_rsi_strategy lines 712-799)
    # ============================================================
    positions = []
    shares_held = []
    cash_balance = []
    equity_value = []
    total_portfolio = []
    
    current_position = 0
    current_shares = 0
    current_cash = initial_cash
    
    for i in range(len(df_signals)):
        price = df_signals.iloc[i]['Adj_Close']
        
        # First (max_window-1) days: no positions (indicator lookback period)
        if i < max_window - 1:
            positions.append(0)
            shares_held.append(0)
            cash_balance.append(current_cash)
            equity_value.append(0)
            total_portfolio.append(current_cash)
            continue
        
        # Process exit signals first (priority) - EXACT COPY
        if current_position == 1 and df_signals.iloc[i]['exit_long']:
            # Exit long: sell shares
            current_cash += current_shares * price
            current_shares = 0
            current_position = 0
            
        elif current_position == -1 and df_signals.iloc[i]['exit_short']:
            # Exit short: buy back shares to cover
            current_cash += current_shares * price  # current_shares is negative for short
            current_shares = 0
            current_position = 0
            
        # Process entry signals (only if flat) - EXACT COPY
        elif current_position == 0:
            if df_signals.iloc[i]['entry_long']:
                # Enter long position
                if position_sizing_method == 'capital_based':
                    # Calculate max shares based on available capital
                    available_capital = current_cash * capital_allocation_pct
                    current_shares = int(available_capital / price)
                else:
                    # Use fixed position size
                    current_shares = position_size
                
                # Execute trade if we can afford at least 1 share
                if current_shares > 0:
                    current_cash -= current_shares * price
                    current_position = 1
                else:
                    current_shares = 0
                    
            elif df_signals.iloc[i]['entry_short']:
                # Enter short position
                if position_sizing_method == 'capital_based':
                    # Calculate max shares based on available capital
                    available_capital = current_cash * capital_allocation_pct
                    current_shares = -int(available_capital / price)  # Negative for short
                else:
                    # Use fixed position size (negative for short)
                    current_shares = -position_size
                
                # Execute trade if we can afford at least 1 share
                if current_shares < 0:
                    current_cash -= current_shares * price  # Subtracting negative adds to cash
                    current_position = -1
                else:
                    current_shares = 0
        
        # Record state - EXACT COPY
        positions.append(current_position)
        shares_held.append(current_shares)
        cash_balance.append(current_cash)
        
        # Calculate equity value (mark-to-market) - EXACT COPY
        equity = current_shares * price
        equity_value.append(equity)
        total_portfolio.append(current_cash + equity)
    
    df_signals['signal'] = positions
    df_signals['shares'] = shares_held
    df_signals['cash'] = cash_balance
    df_signals['equity'] = equity_value
    df_signals['portfolio_value'] = total_portfolio
    
    # ============================================================
    # 4. Calculate Returns and PnL - EXACT COPY
    # ============================================================
    df_signals['returns'] = df_signals['Adj_Close'].pct_change()
    df_signals['position'] = df_signals['signal'].shift(1).fillna(0)
    df_signals['strategy_returns'] = df_signals['returns'] * df_signals['position']
    
    # Calculate daily PnL from portfolio value changes
    df_signals['pnl'] = df_signals['portfolio_value'].diff()
    
    # Calculate cumulative PnL
    df_signals['cumulative_pnl'] = df_signals['portfolio_value'] - initial_cash
    
    return df_signals




def backtest_mean_reversion_rsi_strategy(spy_train_features, spy_test_features):

    nclust = 5
    if nclust == 2:
        regime_names = {0: 'Low Vol', 1: 'High Vol'}
    elif nclust == 3:
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
    elif nclust == 4:
        regime_names = {0: 'Very Low Vol', 1: 'Low Vol', 2: 'Med Vol', 3: 'High Vol'}
    elif nclust == 5:
        regime_names = {
            0: 'Very Low Vol (Calm)',
            1: 'Low Vol (Stable)', 
            2: 'Medium Vol (Normal)',
            3: 'High Vol (Elevated)',
            4: 'Very High Vol (Crisis)'
        }
    else:
        regime_names = {i: f'Regime {i}' for i in range(nclust)}

    # %%
    print("\n" + "="*70)
    print("LOADING RSI REGIME PARAMETERS FROM CSV FILES (FROM TRAINING DATA)")
    print("="*70)

    # Initialize regime_best_params dictionary
    regime_best_params = {}

    for regime_id in range(5):  # Assuming 5 regimes (0-4)
        csv_path = os.path.join(f'mr_rsi_hyperparameter_results_regime_{regime_id}.csv')

        if os.path.exists(csv_path):
            # Read CSV and get best parameters (first row)
            regime_csv = pd.read_csv(csv_path)

            if not regime_csv.empty:
                best_row = regime_csv.iloc[0]
                regime_best_params[regime_id] = best_row.to_dict()

                regime_name = regime_names.get(regime_id, f'Regime {regime_id}')
                print(f"✅ Loaded {regime_name}:")
                print(f"   RSI_Period={best_row['RSI_Period']}, "
                    f"RSI_Oversold={best_row['RSI_Oversold']}, "
                    f"RSI_Overbought={best_row['RSI_Overbought']}")
            else:
                print(f"⚠️  Warning: Empty CSV for regime {regime_id}")
        else:
            print(f"❌ CSV not found: {csv_path}")

    print(f"\n✅ Loaded parameters for {len(regime_best_params)} regimes from CSV")
    print("="*70)


    # %%
    # ============================================================
    # RUN UNIFIED REGIME-AWARE BACKTEST ON FULL TRAIN DATA
    # ============================================================

    print("\n" + "="*70)
    print("UNIFIED REGIME-AWARE BACKTEST - TRAINING DATA")
    print("="*70)
    print(f"\nTraining Data Period:")
    print(f"  Start Date: {spy_train_features.index[0].strftime('%Y-%m-%d')}")
    print(f"  End Date:   {spy_train_features.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Days: {len(spy_train_features)}")
    print("\nTesting strategy on FULL training data with regime transitions...")

    # Prepare regime parameters dictionary from optimization results
    regime_params_for_strategy = {}
    for regime_id, best_params in regime_best_params.items():
        regime_params_for_strategy[int(regime_id)] = {
            'RSI_Period': best_params['RSI_Period'],
            'RSI_Oversold': best_params['RSI_Oversold'],
            'RSI_Overbought': best_params['RSI_Overbought'],
            'RSI_Exit_Long': best_params['RSI_Exit_Long'],
            'RSI_Exit_Short': best_params['RSI_Exit_Short']
        }

    print("\nRegime Parameters Being Used:")
    for regime_id, params in regime_params_for_strategy.items():
        regime_name = regime_names.get(regime_id, f'Regime {regime_id}')
        print(f"  {regime_name}: RSI_Period={params['RSI_Period']}, RSI_Oversold={params['RSI_Oversold']}, "
            f"RSI_Overbought={params['RSI_Overbought']}, RSI_Exit_Long={params['RSI_Exit_Long']}, "
            f"RSI_Exit_Short={params['RSI_Exit_Short']}")

    # Run unified backtest on full training data
    train_unified_results = mr_rsi_compute_regime_aware_strategy(
        spy_train_features,
        regime_params_for_strategy,
        initial_cash=100000,
        position_sizing_method='capital_based',
        capital_allocation_pct=0.98
    )

    # Calculate metrics
    train_unified_metrics = mr_rsi_calculate_performance_metrics(train_unified_results, initial_cash=100000)

    print("\n" + "="*70)
    print("UNIFIED BACKTEST RESULTS - TRAINING DATA")
    print("="*70)
    print(f"Total Return:     {train_unified_metrics['Total Return (%)']:.2f}%")
    print(f"CAGR:             {train_unified_metrics['CAGR (%)']:.2f}%")
    print(f"Sharpe Ratio:     {train_unified_metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:     {train_unified_metrics['Max Drawdown (%)']:.2f}%")
    print(f"Total Trades:     {train_unified_metrics['Total Trades']}")
    print("="*70)

    # %%
    # ============================================================
    # RUN UNIFIED REGIME-AWARE BACKTEST ON TEST DATA
    # ============================================================

    print("\n" + "="*70)
    print("UNIFIED REGIME-AWARE BACKTEST - TEST DATA")
    print("="*70)
    print(f"\nTest Data Period:")
    print(f"  Start Date: {spy_test_features.index[0].strftime('%Y-%m-%d')}")
    print(f"  End Date:   {spy_test_features.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Days: {len(spy_test_features)}")
    print("\nTesting strategy on FULL test data with regime transitions...")

    # Run unified backtest on full test data (using same params from training)
    test_unified_results = mr_rsi_compute_regime_aware_strategy(
        spy_test_features,
        regime_params_for_strategy,
        initial_cash=100000,
        position_sizing_method='capital_based',
        capital_allocation_pct=0.98
    )

    # Calculate metrics
    test_unified_metrics = mr_rsi_calculate_performance_metrics(test_unified_results, initial_cash=100000)

    print("\n" + "="*70)
    print("UNIFIED BACKTEST RESULTS - TEST DATA")
    print("="*70)
    print(f"Total Return:     {test_unified_metrics['Total Return (%)']:.2f}%")
    print(f"CAGR:             {test_unified_metrics['CAGR (%)']:.2f}%")
    print(f"Sharpe Ratio:     {test_unified_metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:     {test_unified_metrics['Max Drawdown (%)']:.2f}%")
    print(f"Total Trades:     {test_unified_metrics['Total Trades']}")
    print("="*70)

    # %%
    # ============================================================
    # COMPARE: REGIME-AWARE VS REGIME-AGNOSTIC
    # ============================================================

    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: TRAIN vs TEST")
    print("="*70)

    comparison_table = pd.DataFrame({
        'Metric': ['Total Return (%)', 'CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Total Trades'],
        'Train': [
            f"{train_unified_metrics['Total Return (%)']:.2f}",
            f"{train_unified_metrics['CAGR (%)']:.2f}",
            f"{train_unified_metrics['Sharpe Ratio']:.2f}",
            f"{train_unified_metrics['Max Drawdown (%)']:.2f}",
            f"{train_unified_metrics['Total Trades']}"
        ],
        'Test': [
            f"{test_unified_metrics['Total Return (%)']:.2f}",
            f"{test_unified_metrics['CAGR (%)']:.2f}",
            f"{test_unified_metrics['Sharpe Ratio']:.2f}",
            f"{test_unified_metrics['Max Drawdown (%)']:.2f}",
            f"{test_unified_metrics['Total Trades']}"
        ]
    })

    display(comparison_table)

    print("\n💡 INTERPRETATION:")
    print("   - Similar Sharpe ratios → Good generalization, no overfitting")
    print("   - Test Sharpe much lower → May be overfitted to train data")
    print("   - Test Sharpe higher → Got lucky with test period or train was more volatile")
    print("="*70)

    #%%
    # ============================================================
    # VISUALIZE TRAIN VS TEST COMPARISON WITH REGIME OVERLAY
    # ============================================================

    print("\n" + "="*70)
    print("GENERATING TRAIN vs TEST VISUALIZATION WITH REGIME OVERLAY...")
    print("="*70)

    # Create regime names dict based on actual number of clusters
    # Current setting: nclust = 5 (see line 316)
    if nclust == 2:
        regime_names = {0: 'Low Vol', 1: 'High Vol'}
    elif nclust == 3:
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
    elif nclust == 4:
        regime_names = {0: 'Very Low Vol', 1: 'Low Vol', 2: 'Med Vol', 3: 'High Vol'}
    elif nclust == 5:
        regime_names = {
            0: 'Very Low Vol (Calm)',
            1: 'Low Vol (Stable)', 
            2: 'Medium Vol (Normal)',
            3: 'High Vol (Elevated)',
            4: 'Very High Vol (Crisis)'
        }
    else:
        regime_names = {i: f'Regime {i}' for i in range(nclust)}

    # Call the comparison visualization
    mr_rsi_viz_regime_aware_backtest_comparison(
        train_unified_results,
        test_unified_results,
        regime_names_dict=regime_names
    )
    
    return test_unified_results



# %%
# %%
def mr_compute_mean_reversion_sma_pct_candle_strategy(df, window=21, position_size=150, initial_cash=100000,
                                   pct_threshold_long=2.0, pct_threshold_short=2.0, min_candles=3,
                                   position_sizing_method='capital_based', capital_allocation_pct=0.98,
                                   target_regime=None, signal_shift=0):
    """
    Compute mean reversion strategy based on percentage distance from SMA and consecutive candle count.

    Parameters:
    -----------
    df : DataFrame with 'Adj_Close' column
    window : int, rolling window for SMA calculation (default=20)
    position_size : int, number of shares per trade when using 'fixed' method (default=150)
    initial_cash : float, starting capital (default=100000)
    pct_threshold_long : float, percentage distance below SMA for long entry (default=2.0)
        - e.g., 2.0 means enter long when price is 2%+ below SMA
    pct_threshold_short : float, percentage distance above SMA for short entry (default=2.0)
        - e.g., 2.0 means enter short when price is 2%+ above SMA
    min_candles : int, minimum consecutive candles above/below SMA required for entry (default=3)
    position_sizing_method : str, either 'fixed' or 'capital_based' (default='capital_based')
        - 'fixed': uses fixed position_size parameter
        - 'capital_based': calculates max shares based on available capital
    capital_allocation_pct : float, percentage of available capital to use per trade (default=0.98)
        Only applies when position_sizing_method='capital_based'
    target_regime : int or None, if specified, only trade when current regime matches this value (default=None)
        - None: trade in all regimes (standard behavior)
        - int: only generate entry signals when df['regime'] == target_regime
    signal_shift : int, number of days to delay signal execution (default=0)
        - 0: execute signals immediately
        - >0: delay signal execution by this many days (e.g., signal_shift=1 means trade 1 day after signal)

    Returns:
    --------
    DataFrame with signals, positions, PnL calculations, and portfolio tracking
    """
    df_signals = df.copy()
    # ============================================================
    # 1. Calculate Technical Indicators
    # ============================================================
    # Calculate dynamic SMA
    df_signals['dynamic_sma'] = df_signals['Adj_Close'].rolling(
        window=window, min_periods=1, center=False
    ).mean()

    # Calculate percentage distance from SMA
    # Positive % = price above SMA, Negative % = price below SMA
    df_signals['pct_distance'] = (
        (df_signals['Adj_Close'] - df_signals['dynamic_sma']) / df_signals['dynamic_sma'] * 100
    )
    df_signals['pct_distance'] = df_signals['pct_distance'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Calculate percentage-based bands for visualization
    df_signals['range_upper'] = df_signals['dynamic_sma'] * (1 + pct_threshold_short / 100)
    df_signals['range_lower'] = df_signals['dynamic_sma'] * (1 - pct_threshold_long / 100)

    # Calculate consecutive candle counts
    # Identify when price is above or below SMA
    above_sma = (df_signals['Adj_Close'] > df_signals['dynamic_sma']).astype(int)
    below_sma = (df_signals['Adj_Close'] < df_signals['dynamic_sma']).astype(int)

    # Calculate consecutive runs
    # When value changes, reset counter; when same, increment counter
    df_signals['candles_above_count'] = above_sma.groupby((above_sma != above_sma.shift()).cumsum()).cumsum()
    df_signals['candles_below_count'] = below_sma.groupby((below_sma != below_sma.shift()).cumsum()).cumsum() 

    # ============================================================
    # 2. Generate Entry/Exit Signals (using % distance + candle count)
    # ============================================================
    # Entry Long: Price is X% below SMA AND has been below for min_candles+ consecutive candles
    # Both conditions must be TRUE (AND logic)
    df_signals['entry_long'] = (
        (df_signals['pct_distance'] < -pct_threshold_long) &  # Price is X% below SMA
        (df_signals['candles_below_count'] >= min_candles)     # Been below for N+ candles
    )

    # Exit Long: Price crosses back above SMA (mean reversion complete)
    df_signals['exit_long'] = (
        (df_signals['pct_distance'] >= 0)  # Price at or above SMA
    )

    # Entry Short: Price is X% above SMA AND has been above for min_candles+ consecutive candles
    # Both conditions must be TRUE (AND logic)
    df_signals['entry_short'] = (
        (df_signals['pct_distance'] > pct_threshold_short) &  # Price is X% above SMA
        (df_signals['candles_above_count'] >= min_candles)     # Been above for N+ candles
    )

    # Exit Short: Price crosses back below SMA (mean reversion complete)
    df_signals['exit_short'] = (
        (df_signals['pct_distance'] <= 0)  # Price at or below SMA
    )

    # Apply signal shift if specified (shift signals forward to delay execution)
    if signal_shift > 0:
        df_signals['entry_long'] = df_signals['entry_long'].shift(signal_shift).fillna(False)
        df_signals['exit_long'] = df_signals['exit_long'].shift(signal_shift).fillna(False)
        df_signals['entry_short'] = df_signals['entry_short'].shift(signal_shift).fillna(False)
        df_signals['exit_short'] = df_signals['exit_short'].shift(signal_shift).fillna(False)

    # ============================================================
    # 3. Calculate Position State and Portfolio Tracking
    # ============================================================
    # Position states: 0 = flat, 1 = long, -1 = short
    positions = []
    shares_held = []
    cash_balance = []
    equity_value = []
    total_portfolio = []
    
    current_position = 0
    current_shares = 0
    current_cash = initial_cash

    for i in range(len(df_signals)):
        price = df_signals.iloc[i]['Adj_Close']
        
        # First (window-1) days: no positions (indicator lookback period)
        if i < window - 1:
            positions.append(0)
            shares_held.append(0)
            cash_balance.append(current_cash)
            equity_value.append(0)
            total_portfolio.append(current_cash)
            continue

        # Process exit signals first (priority)
        if current_position == 1 and df_signals.iloc[i]['exit_long']:
            # Exit long: sell shares
            current_cash += current_shares * price
            current_shares = 0
            current_position = 0
            
        elif current_position == -1 and df_signals.iloc[i]['exit_short']:
            # Exit short: buy back shares to cover
            current_cash += current_shares * price  # current_shares is negative for short
            current_shares = 0
            current_position = 0
            
        # Process entry signals (only if flat)
        elif current_position == 0:
            # Check if we should trade in this regime (if target_regime is specified)
            regime_check = True
            if target_regime is not None and 'regime' in df_signals.columns:
                current_regime = df_signals.iloc[i]['regime']
                # Only trade if regime matches (skip NaN regimes too)
                regime_check = (not pd.isna(current_regime)) and (current_regime == target_regime)
            
            if regime_check and df_signals.iloc[i]['entry_long']:
                # Enter long position
                if position_sizing_method == 'capital_based':
                    # Calculate max shares based on available capital
                    available_capital = current_cash * capital_allocation_pct
                    current_shares = int(available_capital / price)
                else:
                    # Use fixed position size
                    current_shares = position_size
                
                # Execute trade if we can afford at least 1 share
                if current_shares > 0:
                    current_cash -= current_shares * price
                    current_position = 1
                else:
                    current_shares = 0
                    
            elif regime_check and df_signals.iloc[i]['entry_short']:
                # Enter short position
                if position_sizing_method == 'capital_based':
                    # Calculate max shares based on available capital
                    available_capital = current_cash * capital_allocation_pct
                    current_shares = -int(available_capital / price)  # Negative for short
                else:
                    # Use fixed position size (negative for short)
                    current_shares = -position_size
                
                # Execute trade if we can afford at least 1 share
                if current_shares < 0:
                    current_cash -= current_shares * price  # Subtracting negative adds to cash
                    current_position = -1
                else:
                    current_shares = 0

        # Record state
        positions.append(current_position)
        shares_held.append(current_shares)
        cash_balance.append(current_cash)
        
        # Calculate equity value (mark-to-market)
        equity = current_shares * price
        equity_value.append(equity)
        total_portfolio.append(current_cash + equity)

    df_signals['signal'] = positions
    df_signals['shares'] = shares_held
    df_signals['cash'] = cash_balance
    df_signals['equity'] = equity_value
    df_signals['portfolio_value'] = total_portfolio

    # ============================================================
    # 4. Calculate Returns and PnL
    # ============================================================
    df_signals['returns'] = df_signals['Adj_Close'].pct_change()
    df_signals['position'] = df_signals['signal'].shift(1).fillna(0)
    df_signals['strategy_returns'] = df_signals['returns'] * df_signals['position']       

    # Calculate daily PnL from portfolio value changes
    df_signals['pnl'] = df_signals['portfolio_value'].diff()
    
    # Calculate cumulative PnL
    df_signals['cumulative_pnl'] = df_signals['portfolio_value'] - initial_cash

    return df_signals


# %%
def mr_sma_pct_candle_viz_pct_distance_analysis(df_train, df_test, window=20,
                                         pct_threshold_long=2.0, pct_threshold_short=2.0,
                                         regime_names_dict=None):
    """
    Visualize percentage distance from SMA across train and test data.

    This function provides comprehensive visualization of the percentage distance metric
    used in the mean reversion strategy, comparing training and test periods.

    Parameters:
    -----------
    df_train : DataFrame with 'Adj_Close' column (training data)
    df_test : DataFrame with 'Adj_Close' column (test data)
    window : int, SMA window for calculation (default=20)
    pct_threshold_long : float, percentage below SMA for long entry threshold (default=2.0)
    pct_threshold_short : float, percentage above SMA for short entry threshold (default=2.0)
    regime_names_dict : dict, optional mapping of regime_id to regime names

    Visualizes:
    -----------
    Row 1: Price vs SMA comparison (train | test)
    Row 2: Percentage distance from SMA with threshold lines (train | test)
    Row 3: Distribution of percentage distances (train | test)
    Row 4: Consecutive candle counts above/below SMA (train | test)
    """

    # Calculate SMA and percentage distance for both datasets
    def calc_pct_distance(df, window):
        df_calc = df.copy()
        df_calc['sma'] = df_calc['Adj_Close'].rolling(window=window, min_periods=1).mean()
        df_calc['pct_distance'] = ((df_calc['Adj_Close'] - df_calc['sma']) / df_calc['sma']) * 100

        # Track position relative to SMA
        df_calc['above_sma'] = df_calc['Adj_Close'] > df_calc['sma']
        df_calc['below_sma'] = df_calc['Adj_Close'] < df_calc['sma']

        # Count consecutive candles using groupby pattern
        df_calc['above_group'] = (df_calc['above_sma'] != df_calc['above_sma'].shift()).cumsum()
        df_calc['below_group'] = (df_calc['below_sma'] != df_calc['below_sma'].shift()).cumsum()

        df_calc['candles_above'] = df_calc.groupby('above_group').cumcount() + 1
        df_calc['candles_below'] = df_calc.groupby('below_group').cumcount() + 1

        # Only count when actually above/below
        df_calc.loc[~df_calc['above_sma'], 'candles_above'] = 0
        df_calc.loc[~df_calc['below_sma'], 'candles_below'] = 0

        return df_calc

    train_calc = calc_pct_distance(df_train, window)
    test_calc = calc_pct_distance(df_test, window)

    # Create 4x2 subplot figure
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    fig.suptitle(f'Percentage Distance from SMA Analysis (Window={window})\n'
                 f'Long Threshold: -{pct_threshold_long}% | Short Threshold: +{pct_threshold_short}%',
                 fontsize=14, fontweight='bold', y=0.98)

    # ===== ROW 1: Price vs SMA =====
    # Train
    ax1 = axes[0, 0]
    ax1.plot(train_calc.index, train_calc['Adj_Close'], label='Price', color='black', linewidth=1)
    ax1.plot(train_calc.index, train_calc['sma'], label=f'SMA({window})', color='blue', linewidth=1.5, alpha=0.8)
    ax1.fill_between(train_calc.index,
                     train_calc['sma'] * (1 - pct_threshold_long/100),
                     train_calc['sma'] * (1 + pct_threshold_short/100),
                     alpha=0.2, color='gray', label='Threshold Band')
    ax1.set_title('TRAIN: Price vs SMA', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Test
    ax2 = axes[0, 1]
    ax2.plot(test_calc.index, test_calc['Adj_Close'], label='Price', color='black', linewidth=1)
    ax2.plot(test_calc.index, test_calc['sma'], label=f'SMA({window})', color='blue', linewidth=1.5, alpha=0.8)
    ax2.fill_between(test_calc.index,
                     test_calc['sma'] * (1 - pct_threshold_long/100),
                     test_calc['sma'] * (1 + pct_threshold_short/100),
                     alpha=0.2, color='gray', label='Threshold Band')
    ax2.set_title('TEST: Price vs SMA', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ===== ROW 2: Percentage Distance with Thresholds =====
    # Train
    ax3 = axes[1, 0]
    ax3.plot(train_calc.index, train_calc['pct_distance'], label='% Distance', color='purple', linewidth=0.8)
    ax3.axhline(y=pct_threshold_short, color='red', linestyle='--', linewidth=1.5, label=f'Short Entry (+{pct_threshold_short}%)')
    ax3.axhline(y=-pct_threshold_long, color='green', linestyle='--', linewidth=1.5, label=f'Long Entry (-{pct_threshold_long}%)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax3.fill_between(train_calc.index, train_calc['pct_distance'], 0,
                     where=(train_calc['pct_distance'] > pct_threshold_short),
                     color='red', alpha=0.3, label='Above Short Threshold')
    ax3.fill_between(train_calc.index, train_calc['pct_distance'], 0,
                     where=(train_calc['pct_distance'] < -pct_threshold_long),
                     color='green', alpha=0.3, label='Below Long Threshold')
    ax3.set_title('TRAIN: Percentage Distance from SMA', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('% Distance from SMA')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Test
    ax4 = axes[1, 1]
    ax4.plot(test_calc.index, test_calc['pct_distance'], label='% Distance', color='purple', linewidth=0.8)
    ax4.axhline(y=pct_threshold_short, color='red', linestyle='--', linewidth=1.5, label=f'Short Entry (+{pct_threshold_short}%)')
    ax4.axhline(y=-pct_threshold_long, color='green', linestyle='--', linewidth=1.5, label=f'Long Entry (-{pct_threshold_long}%)')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax4.fill_between(test_calc.index, test_calc['pct_distance'], 0,
                     where=(test_calc['pct_distance'] > pct_threshold_short),
                     color='red', alpha=0.3, label='Above Short Threshold')
    ax4.fill_between(test_calc.index, test_calc['pct_distance'], 0,
                     where=(test_calc['pct_distance'] < -pct_threshold_long),
                     color='green', alpha=0.3, label='Below Long Threshold')
    ax4.set_title('TEST: Percentage Distance from SMA', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('% Distance from SMA')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ===== ROW 3: Distribution Histograms =====
    # Train
    ax5 = axes[2, 0]
    train_pct = train_calc['pct_distance'].dropna()
    ax5.hist(train_pct, bins=50, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.axvline(x=pct_threshold_short, color='red', linestyle='--', linewidth=2, label=f'Short (+{pct_threshold_short}%)')
    ax5.axvline(x=-pct_threshold_long, color='green', linestyle='--', linewidth=2, label=f'Long (-{pct_threshold_long}%)')
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax5.axvline(x=train_pct.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean ({train_pct.mean():.2f}%)')

    # Add statistics text box
    stats_text = f'Mean: {train_pct.mean():.2f}%\nStd: {train_pct.std():.2f}%\nMin: {train_pct.min():.2f}%\nMax: {train_pct.max():.2f}%'
    ax5.text(0.98, 0.98, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax5.set_title('TRAIN: Distribution of % Distance', fontsize=12, fontweight='bold')
    ax5.set_xlabel('% Distance from SMA')
    ax5.set_ylabel('Frequency')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Test
    ax6 = axes[2, 1]
    test_pct = test_calc['pct_distance'].dropna()
    ax6.hist(test_pct, bins=50, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax6.axvline(x=pct_threshold_short, color='red', linestyle='--', linewidth=2, label=f'Short (+{pct_threshold_short}%)')
    ax6.axvline(x=-pct_threshold_long, color='green', linestyle='--', linewidth=2, label=f'Long (-{pct_threshold_long}%)')
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax6.axvline(x=test_pct.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean ({test_pct.mean():.2f}%)')

    # Add statistics text box
    stats_text = f'Mean: {test_pct.mean():.2f}%\nStd: {test_pct.std():.2f}%\nMin: {test_pct.min():.2f}%\nMax: {test_pct.max():.2f}%'
    ax6.text(0.98, 0.98, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax6.set_title('TEST: Distribution of % Distance', fontsize=12, fontweight='bold')
    ax6.set_xlabel('% Distance from SMA')
    ax6.set_ylabel('Frequency')
    ax6.legend(loc='upper left', fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ===== ROW 4: Consecutive Candle Counts =====
    # Train
    ax7 = axes[3, 0]
    ax7.bar(train_calc.index, train_calc['candles_above'], color='red', alpha=0.6, width=1, label='Candles Above SMA')
    ax7.bar(train_calc.index, -train_calc['candles_below'], color='green', alpha=0.6, width=1, label='Candles Below SMA')
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax7.set_title('TRAIN: Consecutive Candles Above/Below SMA', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Consecutive Candles (Above +, Below -)')
    ax7.legend(loc='best', fontsize=9)
    ax7.grid(True, alpha=0.3)

    # Test
    ax8 = axes[3, 1]
    ax8.bar(test_calc.index, test_calc['candles_above'], color='red', alpha=0.6, width=1, label='Candles Above SMA')
    ax8.bar(test_calc.index, -test_calc['candles_below'], color='green', alpha=0.6, width=1, label='Candles Below SMA')
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax8.set_title('TEST: Consecutive Candles Above/Below SMA', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Consecutive Candles (Above +, Below -)')
    ax8.legend(loc='best', fontsize=9)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()

    # Print summary statistics table
    print("\n" + "="*80)
    print("PERCENTAGE DISTANCE SUMMARY STATISTICS")
    print("="*80)
    print(f"\n{'Metric':<30} {'TRAIN':>20} {'TEST':>20}")
    print("-"*70)
    print(f"{'Mean % Distance':<30} {train_pct.mean():>19.2f}% {test_pct.mean():>19.2f}%")
    print(f"{'Std Dev % Distance':<30} {train_pct.std():>19.2f}% {test_pct.std():>19.2f}%")
    print(f"{'Min % Distance':<30} {train_pct.min():>19.2f}% {test_pct.min():>19.2f}%")
    print(f"{'Max % Distance':<30} {train_pct.max():>19.2f}% {test_pct.max():>19.2f}%")
    print(f"{'Days Above Short Threshold':<30} {(train_pct > pct_threshold_short).sum():>20} {(test_pct > pct_threshold_short).sum():>20}")
    print(f"{'Days Below Long Threshold':<30} {(train_pct < -pct_threshold_long).sum():>20} {(test_pct < -pct_threshold_long).sum():>20}")
    print(f"{'Max Consecutive Above SMA':<30} {train_calc['candles_above'].max():>20} {test_calc['candles_above'].max():>20}")
    print(f"{'Max Consecutive Below SMA':<30} {train_calc['candles_below'].max():>20} {test_calc['candles_below'].max():>20}")
    print("="*80)

    return train_calc, test_calc



# %%
def mr_sma_pct_candle_hyperparameter_test(df,
                       window_range=[10, 20, 30, 50],
                       pct_threshold_long_range=[1.0, 1.5, 2.0, 2.5],
                       pct_threshold_short_range=[1.0, 1.5, 2.0, 2.5],
                       min_candles_range=[2, 3, 4, 5],
                       position_sizing_methods=['fixed', 'capital_based'],
                       position_sizes=[150],
                       capital_allocation_pcts=[0.98],
                       initial_cash=100000,
                       sort_by='Sharpe Ratio',
                       top_n=20,
                       verbose=True,
                       n_jobs=-1,
                       target_regime=None):
    """
    Test multiple hyperparameter combinations for the mean reversion strategy with PARALLEL PROCESSING.

    Parameters:
    -----------
    df : DataFrame with 'Adj_Close' column
    window_range : list of int, rolling window values to test
    pct_threshold_long_range : list of float, percentage distance below SMA for long entry (positive values, e.g., [1.0, 1.5, 2.0])
    pct_threshold_short_range : list of float, percentage distance above SMA for short entry (positive values, e.g., [1.0, 1.5, 2.0])
    min_candles_range : list of int, minimum consecutive candles above/below SMA required for entry (e.g., [2, 3, 4, 5])
    position_sizing_methods : list of str, methods to test ['fixed', 'capital_based']
    position_sizes : list of int, position sizes for 'fixed' method
    capital_allocation_pcts : list of float, allocation % for 'capital_based' method
    initial_cash : float, starting capital
    sort_by : str, metric to sort results by (default='Sharpe Ratio')
    top_n : int, number of top results to return (default=20)
    verbose : bool, print progress updates
    n_jobs : int, number of parallel jobs (-1 uses all cores, 1 for sequential, default=-1)
    target_regime : int or None, if specified, only trade when current regime matches (default=None)

    Returns:
    --------
    DataFrame with results sorted by specified metric
    """
    import itertools
    from datetime import datetime
    from joblib import Parallel, delayed
    import io
    import sys
    
    results = []
    
    # Generate all parameter combinations
    param_combinations = []

    for window in window_range:
        for pct_long in pct_threshold_long_range:
            for pct_short in pct_threshold_short_range:
                for min_cand in min_candles_range:
                    for method in position_sizing_methods:
                        if method == 'fixed':
                            for pos_size in position_sizes:
                                param_combinations.append({
                                    'window': window,
                                    'pct_threshold_long': pct_long,
                                    'pct_threshold_short': pct_short,
                                    'min_candles': min_cand,
                                    'position_sizing_method': method,
                                    'position_size': pos_size,
                                    'capital_allocation_pct': None
                                })
                        else:  # capital_based
                            for cap_alloc in capital_allocation_pcts:
                                param_combinations.append({
                                    'window': window,
                                    'pct_threshold_long': pct_long,
                                    'pct_threshold_short': pct_short,
                                    'min_candles': min_cand,
                                    'position_sizing_method': method,
                                    'position_size': None,
                                    'capital_allocation_pct': cap_alloc
                                })
    
    # Helper function to test a single parameter combination
    def test_single_combination(params):
        try:
            # Run strategy
            if params['position_sizing_method'] == 'fixed':
                df_signals = mr_compute_mean_reversion_sma_pct_candle_strategy(
                    df,
                    window=params['window'],
                    position_size=params['position_size'],
                    initial_cash=initial_cash,
                    pct_threshold_long=params['pct_threshold_long'],
                    pct_threshold_short=params['pct_threshold_short'],
                    min_candles=params['min_candles'],
                    position_sizing_method=params['position_sizing_method'],
                    target_regime=target_regime
                )
            else:  # capital_based
                df_signals = mr_compute_mean_reversion_sma_pct_candle_strategy(
                    df,
                    window=params['window'],
                    initial_cash=initial_cash,
                    pct_threshold_long=params['pct_threshold_long'],
                    pct_threshold_short=params['pct_threshold_short'],
                    min_candles=params['min_candles'],
                    position_sizing_method=params['position_sizing_method'],
                    capital_allocation_pct=params['capital_allocation_pct'],
                    target_regime=target_regime
                )

            # Calculate metrics (suppress print output)
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            metrics = mr_sma_bb_calculate_performance_metrics(df_signals, initial_cash=initial_cash)

            sys.stdout = old_stdout

            # Store results
            result = {
                'Window': params['window'],
                'Pct_Long': params['pct_threshold_long'],
                'Pct_Short': params['pct_threshold_short'],
                'Min_Candles': params['min_candles'],
                'Method': params['position_sizing_method'],
                'Position_Size': params['position_size'],
                'Capital_Alloc_Pct': params['capital_allocation_pct'],
                **metrics  # Add all metrics
            }
            return result

        except Exception as e:
            return None
    
    total_tests = len(param_combinations)
    if verbose:
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TEST - PARALLEL PROCESSING")
        print(f"{'='*70}")
        print(f"Testing {total_tests} parameter combinations...")
        print(f"Using {n_jobs if n_jobs > 0 else 'ALL'} CPU cores")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")
    
    # Run tests in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(test_single_combination)(params) for params in param_combinations
    )
    
    # Filter out None results (failed tests)
    results = [r for r in results if r is not None]
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\nCompleted at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Successful tests: {len(results_df)}/{total_tests}")
    
    # Sort by specified metric (descending for most metrics)
    if sort_by in results_df.columns:
        ascending = True if 'Drawdown' in sort_by else False  # Drawdown should be ascending (less negative is better)
        results_df = results_df.sort_values(by=sort_by, ascending=ascending)
    
    # Return top N results
    return results_df.head(top_n) if top_n else results_df




# %%
def mr_sma_pct_candle_calculate_performance_metrics(df_signals, initial_cash=100000):
    """
    Calculate comprehensive performance metrics for the mean reversion strategy.

    Parameters:
    -----------
    df_signals : DataFrame returned from mr_compute_mean_reversion_sma_pct_candle_strategy()
    initial_cash : float, starting capital (default=100000)

    Returns:
    --------
    Dictionary with performance metrics
    """
    # Calculate signal changes to identify individual trades
    df_signals['signal_change'] = df_signals['signal'].diff()

    # Identify trade periods (when position is held)
    df_long = df_signals[df_signals['signal'] == 1].copy()
    df_short = df_signals[df_signals['signal'] == -1].copy()

    # Calculate trade-level P&L by grouping consecutive positions
    trades = []
    current_signal = 0
    entry_idx = None

    for idx, row in df_signals.iterrows():
        if row['signal'] != current_signal:
            # Position changed - close previous trade if exists
            if current_signal != 0 and entry_idx is not None:
                exit_idx = idx
                trade_pnl = df_signals.loc[entry_idx:idx, 'pnl'].sum()
                trades.append({
                    'entry_date': entry_idx,
                    'exit_date': exit_idx,
                    'direction': 'Long' if current_signal == 1 else 'Short',
                    'pnl': trade_pnl
                })

            # Start new trade if entering position
            if row['signal'] != 0:
                entry_idx = idx
            current_signal = row['signal']

    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)

    # Separate by direction and win/loss
    if len(trades_df) > 0:
        winning_longs = trades_df[(trades_df['direction'] == 'Long') & (trades_df['pnl'] > 0)]
        losing_longs = trades_df[(trades_df['direction'] == 'Long') & (trades_df['pnl'] <= 0)]
        winning_shorts = trades_df[(trades_df['direction'] == 'Short') & (trades_df['pnl'] > 0)]
        losing_shorts = trades_df[(trades_df['direction'] == 'Short') & (trades_df['pnl'] <= 0)]
    else:
        winning_longs = losing_longs = winning_shorts = losing_shorts = pd.DataFrame()

    # Calculate total returns
    # Use portfolio_value if available (more accurate), otherwise calculate from pnl
    if 'portfolio_value' in df_signals.columns:
        final_value = df_signals['portfolio_value'].iloc[-1]
        total_pnl = final_value - initial_cash
    else:
        total_pnl = df_signals['pnl'].sum()
        final_value = initial_cash + total_pnl
    
    total_return_pct = (total_pnl / initial_cash) * 100

    # Calculate CAGR
    years = (df_signals.index[-1] - df_signals.index[0]).days / 365.25
    cagr = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Calculate Sharpe Ratio (annualized)
    daily_returns = df_signals['strategy_returns'].dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Calculate Max Drawdown
    cumulative_returns = (1 + df_signals['strategy_returns']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Calculate volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0

    # Calculate final P&L by direction
    final_long_pnl = (winning_longs['pnl'].sum() if len(winning_longs) > 0 else 0) + \
                     (losing_longs['pnl'].sum() if len(losing_longs) > 0 else 0)
    final_short_pnl = (winning_shorts['pnl'].sum() if len(winning_shorts) > 0 else 0) + \
                      (losing_shorts['pnl'].sum() if len(losing_shorts) > 0 else 0)

    # Compile metrics
    metrics = {
        'Total P&L ($)': total_pnl,
        'Total Return (%)': total_return_pct,
        'CAGR (%)': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Volatility (% annualized)': volatility,
        'Total Trades': len(trades_df),
        'Winning Longs': len(winning_longs),
        'Winning Longs P&L ($)': winning_longs['pnl'].sum() if len(winning_longs) > 0 else 0,
        'Losing Longs': len(losing_longs),
        'Losing Longs P&L ($)': losing_longs['pnl'].sum() if len(losing_longs) > 0 else 0,
        'Final Long P&L ($)': final_long_pnl,
        'Long Win Rate (%)': (len(winning_longs) / (len(winning_longs) + len(losing_longs)) * 100) if (len(winning_longs) + len(losing_longs)) > 0 else 0,
        'Winning Shorts': len(winning_shorts),
        'Winning Shorts P&L ($)': winning_shorts['pnl'].sum() if len(winning_shorts) > 0 else 0,
        'Losing Shorts': len(losing_shorts),
        'Losing Shorts P&L ($)': losing_shorts['pnl'].sum() if len(losing_shorts) > 0 else 0,
        'Final Short P&L ($)': final_short_pnl,
        'Short Win Rate (%)': (len(winning_shorts) / (len(winning_shorts) + len(losing_shorts)) * 100) if (len(winning_shorts) + len(losing_shorts)) > 0 else 0,
    }

    # Print formatted summary
    print(f"\n{'='*70}")
    print(f"{'PERFORMANCE METRICS SUMMARY':^70}")
    print(f"{'='*70}\n")

    print(f"{'OVERALL PERFORMANCE':-^70}")
    print(f"Total P&L:                ${metrics['Total P&L ($)']:>15,.2f}")
    print(f"Total Return:             {metrics['Total Return (%)']:>15,.2f}%")
    print(f"CAGR:                     {metrics['CAGR (%)']:>15,.2f}%")
    print(f"Sharpe Ratio:             {metrics['Sharpe Ratio']:>15,.3f}")
    print(f"Max Drawdown:             {metrics['Max Drawdown (%)']:>15,.2f}%")
    print(f"Volatility (annualized):  {metrics['Volatility (% annualized)']:>15,.2f}%")
    print(f"\n{'TRADE STATISTICS':-^70}")
    print(f"Total Trades:             {metrics['Total Trades']:>15,}")
    print(f"\n{'LONG POSITIONS':-^70}")
    print(f"Winning Longs:            {metrics['Winning Longs']:>15,} trades")
    print(f"Winning Longs P&L:        ${metrics['Winning Longs P&L ($)']:>15,.2f}")
    print(f"Losing Longs:             {metrics['Losing Longs']:>15,} trades")
    print(f"Losing Longs P&L:         ${metrics['Losing Longs P&L ($)']:>15,.2f}")
    print(f"Final Long P&L:           ${metrics['Final Long P&L ($)']:>15,.2f}")
    print(f"Long Win Rate:            {metrics['Long Win Rate (%)']:>15,.2f}%")
    print(f"\n{'SHORT POSITIONS':-^70}")
    print(f"Winning Shorts:           {metrics['Winning Shorts']:>15,} trades")
    print(f"Winning Shorts P&L:       ${metrics['Winning Shorts P&L ($)']:>15,.2f}")
    print(f"Losing Shorts:            {metrics['Losing Shorts']:>15,} trades")
    print(f"Losing Shorts P&L:        ${metrics['Losing Shorts P&L ($)']:>15,.2f}")
    print(f"Final Short P&L:          ${metrics['Final Short P&L ($)']:>15,.2f}")
    print(f"Short Win Rate:           {metrics['Short Win Rate (%)']:>15,.2f}%")
    print(f"\n{'='*70}\n")

    return metrics


# %%
def mr_sma_pct_candle_viz_backtest_mean_reversion(df_signals, start_date=None, end_date=None):
    """
    Visualize mean reversion strategy (percentage distance + consecutive candles) with entry/exit signals.

    This function visualizes the backtesting results from the percentage distance + consecutive candle count
    mean reversion strategy. It shows:
    - Price action with SMA
    - Percentage-based threshold bands (upper/lower)
    - Entry/exit signals for long and short positions

    Entry signals require BOTH:
    1. Price X% away from SMA (percentage distance threshold)
    2. Price has been away for N+ consecutive candles

    Parameters:
    -----------
    df_signals : DataFrame returned from mr_compute_mean_reversion_sma_pct_candle_strategy()
    start_date : str or datetime, optional start date for zoom (e.g., '2020-01-01')
    end_date : str or datetime, optional end date for zoom (e.g., '2020-12-31')
    """
    # Filter by date range if specified
    df_plot = df_signals.copy()
    if start_date is not None:
        df_plot = df_plot[df_plot.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot[df_plot.index <= pd.to_datetime(end_date)]

    plt.figure(figsize=[17, 6])

    # Plot price
    plt.plot(df_plot["Adj_Close"], label="Price = Adjusted Close", linewidth=1.5, color='black')

    # Plot threshold bands (percentage-based)
    plt.plot(df_plot["range_upper"], label="Upper % Threshold (Short Entry)", ls="-.", linewidth=1.2, color='red', alpha=0.7)
    plt.plot(df_plot["range_lower"], label="Lower % Threshold (Long Entry)", ls="-.", linewidth=1.2, color='green', alpha=0.7)

    # Plot moving average
    plt.plot(df_plot["dynamic_sma"], label="Dynamic SMA", linewidth=2, color='blue', alpha=0.8)

    # Calculate signal changes (actual trade executions)
    df_plot['signal_change'] = df_plot['signal'].diff()
    df_plot['prev_signal'] = df_plot['signal'].shift(1)

    # LONG Entry: signal changes to 1 (0→1 or -1→1)
    long_entries = df_plot[(df_plot['signal'] == 1) & (df_plot['prev_signal'] != 1)]
    if len(long_entries) > 0:
        long_entries["Adj_Close"].plot(label="Long Entry", style="g^", markersize=12)

    # LONG Exit: signal changes from 1 to something else
    long_exits = df_plot[(df_plot['prev_signal'] == 1) & (df_plot['signal'] != 1)]
    if len(long_exits) > 0:
        long_exits["Adj_Close"].plot(label="Long Exit", style="v", color="lightgreen", markersize=12)

    # SHORT Entry: signal changes to -1 (0→-1 or 1→-1)
    short_entries = df_plot[(df_plot['signal'] == -1) & (df_plot['prev_signal'] != -1)]
    if len(short_entries) > 0:
        short_entries["Adj_Close"].plot(label="Short Entry", style="rv", markersize=12)

    # SHORT Exit: signal changes from -1 to something else
    short_exits = df_plot[(df_plot['prev_signal'] == -1) & (df_plot['signal'] != -1)]
    if len(short_exits) > 0:
        short_exits["Adj_Close"].plot(label="Short Exit", style="^", color="lightcoral", markersize=12)

    plt.title("Mean Reversion Strategy - Percentage Distance + Consecutive Candles\nEntry/Exit Signals",
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    


# %%
def mr_sma_pct_candle_regime_aware_backtest_signals(df_results, start_date=None, end_date=None, regime_names_dict=None):
    """
    Visualize regime-aware mean reversion strategy with regime backgrounds and entry/exit signals.
    
    Parameters:
    -----------
    df_results : DataFrame from mr_sma_bb_compute_regime_aware_strategy() 
    start_date : str or datetime, optional start date for zoom (e.g., '2020-01-01')
    end_date : str or datetime, optional end date for zoom (e.g., '2020-12-31')
    regime_names_dict : dict mapping regime IDs to names
    """
    # Filter by date range if specified
    df_plot = df_results.copy()
    if start_date is not None:
        df_plot = df_plot[df_plot.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot[df_plot.index <= pd.to_datetime(end_date)]
    
    # Define regime colors (gradient from calm to crisis)
    regime_colors = {
        0: 'lightgreen',    # Very Low Vol: Green (calm)
        1: 'lightblue',     # Low Vol: Blue (stable)
        2: 'lightyellow',   # Med Vol: Yellow (normal)
        3: 'orange',        # High Vol: Orange (elevated)
        4: 'lightcoral'     # Very High Vol: Red (crisis)
    }
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Add regime background first (behind everything)
    if 'regime' in df_plot.columns:
        for regime in df_plot['regime'].dropna().unique():
            regime_mask = df_plot['regime'] == regime
            regime_label = regime_names_dict.get(int(regime), f'Regime {int(regime)}') if regime_names_dict else f'Regime {int(regime)}'
            ax.fill_between(df_plot.index,
                           df_plot["Adj_Close"].min() * 0.9,
                           df_plot["Adj_Close"].max() * 1.1,
                           where=regime_mask,
                           alpha=0.2,
                           color=regime_colors.get(int(regime), 'lightgray'),
                           label=regime_label)
    
    # Plot price
    ax.plot(df_plot.index, df_plot["Adj_Close"], label="SPY Price", linewidth=2, color='black', zorder=3)
    
    # Plot threshold bands
    ax.plot(df_plot.index, df_plot["range_upper"], label="Upper Threshold", 
            ls="--", linewidth=1.5, color='red', alpha=0.7, zorder=2)
    ax.plot(df_plot.index, df_plot["range_lower"], label="Lower Threshold", 
            ls="--", linewidth=1.5, color='green', alpha=0.7, zorder=2)
    
    # Plot dynamic SMA
    ax.plot(df_plot.index, df_plot["dynamic_sma"], label="Dynamic SMA (Regime-Aware)", 
            linewidth=2, color='blue', alpha=0.8, zorder=2)
    
    # Calculate signal changes (actual trade executions)
    df_plot['signal_change'] = df_plot['signal'].diff()
    df_plot['prev_signal'] = df_plot['signal'].shift(1)
    
    # LONG Entry: signal changes to 1
    long_entries = df_plot[(df_plot['signal'] == 1) & (df_plot['prev_signal'] != 1)]
    if len(long_entries) > 0:
        ax.scatter(long_entries.index, long_entries["Adj_Close"], 
                  label="LONG Entry", marker="^", s=150, color="darkgreen", 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # LONG Exit: signal changes from 1 to something else
    long_exits = df_plot[(df_plot['prev_signal'] == 1) & (df_plot['signal'] != 1)]
    if len(long_exits) > 0:
        ax.scatter(long_exits.index, long_exits["Adj_Close"], 
                  label="LONG Exit", marker="v", s=150, color="lightgreen", 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # SHORT Entry: signal changes to -1
    short_entries = df_plot[(df_plot['signal'] == -1) & (df_plot['prev_signal'] != -1)]
    if len(short_entries) > 0:
        ax.scatter(short_entries.index, short_entries["Adj_Close"], 
                  label="SHORT Entry", marker="v", s=150, color="darkred", 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # SHORT Exit: signal changes from -1 to something else
    short_exits = df_plot[(df_plot['prev_signal'] == -1) & (df_plot['signal'] != -1)]
    if len(short_exits) > 0:
        ax.scatter(short_exits.index, short_exits["Adj_Close"], 
                  label="SHORT Exit", marker="^", s=150, color="lightcoral", 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # Formatting
    date_range = f"{df_plot.index[0].strftime('%Y-%m-%d')} to {df_plot.index[-1].strftime('%Y-%m-%d')}"
    ax.set_title(f"Regime-Aware Mean Reversion Strategy - Entry/Exit Signals\n{date_range}", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print signal summary
    print("\n" + "="*70)
    print("SIGNAL SUMMARY")
    print("="*70)
    print(f"Long Entries:  {len(long_entries)}")
    print(f"Long Exits:    {len(long_exits)}")
    print(f"Short Entries: {len(short_entries)}")
    print(f"Short Exits:   {len(short_exits)}")
    print(f"Total Signals: {len(long_entries) + len(long_exits) + len(short_entries) + len(short_exits)}")
    print("="*70)


# %%
def mr_sma_pct_candle_viz_backtest_returns(df_signals, initial_cash=100000):
    """
    Visualize cumulative returns: strategy vs buy-and-hold.

    Parameters:
    -----------
    df_signals : DataFrame returned from mr_compute_mean_reversion_sma_pct_candle_strategy()
    initial_cash : float, starting capital (default=100000)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[17, 10])

    # Calculate cumulative returns
    df_plot = df_signals.copy()

    # Strategy cumulative returns
    df_plot['cum_strategy_returns'] = (1 + df_plot['strategy_returns']).cumprod() - 1

    # Buy-and-hold cumulative returns
    df_plot['cum_asset_returns'] = (1 + df_plot['returns']).cumprod() - 1

    # Calculate running maximum for drawdown
    strategy_cummax = (1 + df_plot['strategy_returns']).cumprod().expanding().max()
    df_plot['drawdown'] = ((1 + df_plot['strategy_returns']).cumprod() - strategy_cummax) / strategy_cummax

    # Plot 1: Cumulative Returns
    df_plot[['cum_asset_returns', 'cum_strategy_returns']].dropna().plot(ax=ax1, linewidth=1.5)

    # Format y-axis as percentage
    from matplotlib.ticker import PercentFormatter
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax1.legend(['Buy & Hold', 'Mean Reversion Strategy'], loc='best')
    ax1.set_title('Cumulative Returns: Strategy vs Buy-and-Hold', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(alpha=0.3)

    # Plot 2: Drawdown
    df_plot['drawdown'].plot(ax=ax2, color='red', linewidth=1.5)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax2.set_title('Strategy Drawdown', fontsize=14)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.grid(alpha=0.3)
    ax2.fill_between(df_plot.index, df_plot['drawdown'], 0, color='red', alpha=0.3)

    plt.tight_layout()
    plt.show()


# %%
def mr_sma_pct_candle_viz_regime_aware_backtest_comparison(train_results, test_results, regime_names_dict=None):
    """
    Visualize train vs test backtest returns with regime overlay.
    
    Parameters:
    -----------
    train_results : DataFrame from mr_sma_bb_compute_regime_aware_strategy() for training data
    test_results : DataFrame from mr_sma_bb_compute_regime_aware_strategy() for test data
    regime_names_dict : dict mapping regime IDs to names
    """
    fig, axes = plt.subplots(3, 2, figsize=(24, 16))
    
    # Define regime colors (gradient from calm to crisis)
    regime_colors = {
        0: 'lightgreen',    # Very Low Vol: Green (calm)
        1: 'lightblue',     # Low Vol: Blue (stable)
        2: 'lightyellow',   # Med Vol: Yellow (normal)
        3: 'orange',        # High Vol: Orange (elevated)
        4: 'lightcoral'     # Very High Vol: Red (crisis)
    }
    
    # ============================================================
    # TRAINING DATA (Left Column)
    # ============================================================
    
    # 1. Training: Cumulative Returns with Regime Background
    ax_train_returns = axes[0, 0]
    train_cum_strategy = (1 + train_results['strategy_returns']).cumprod() - 1
    train_cum_bh = (1 + train_results['returns_simple']).cumprod() - 1 if 'returns_simple' in train_results.columns else None
    
    # Add regime background
    if 'regime' in train_results.columns:
        for regime in train_results['regime'].dropna().unique():
            regime_mask = train_results['regime'] == regime
            ax_train_returns.fill_between(train_results.index,
                                         train_cum_strategy.min() * 1.1,
                                         train_cum_strategy.max() * 1.1,
                                         where=regime_mask,
                                         alpha=0.15,
                                         color=regime_colors.get(int(regime), 'lightgray'))
    
    ax_train_returns.plot(train_results.index, train_cum_strategy * 100,
                         linewidth=2, color='blue', label='Strategy')
    if train_cum_bh is not None:
        ax_train_returns.plot(train_results.index, train_cum_bh * 100,
                             linewidth=2, color='gray', alpha=0.7, label='Buy & Hold')
    
    ax_train_returns.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_train_returns.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax_train_returns.set_title('TRAINING: Cumulative Returns by Regime', fontsize=14, fontweight='bold')
    ax_train_returns.legend(loc='upper left', fontsize=10)
    ax_train_returns.grid(alpha=0.3)
    
    # 2. Training: Drawdown
    ax_train_dd = axes[1, 0]
    train_cummax = (1 + train_results['strategy_returns']).cumprod().expanding().max()
    train_dd = ((1 + train_results['strategy_returns']).cumprod() - train_cummax) / train_cummax
    
    ax_train_dd.plot(train_results.index, train_dd * 100, linewidth=2, color='red')
    ax_train_dd.fill_between(train_results.index, train_dd * 100, 0, color='red', alpha=0.3)
    ax_train_dd.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_train_dd.set_ylabel('Drawdown (%)', fontsize=12)
    ax_train_dd.set_title('TRAINING: Strategy Drawdown', fontsize=14, fontweight='bold')
    ax_train_dd.grid(alpha=0.3)
    
    # 3. Training: Portfolio Value
    ax_train_pv = axes[2, 0]
    ax_train_pv.plot(train_results.index, train_results['portfolio_value'],
                    linewidth=2, color='green')
    ax_train_pv.axhline(y=100000, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax_train_pv.fill_between(train_results.index, 100000, train_results['portfolio_value'],
                            where=(train_results['portfolio_value'] >= 100000),
                            alpha=0.3, color='green')
    ax_train_pv.fill_between(train_results.index, 100000, train_results['portfolio_value'],
                            where=(train_results['portfolio_value'] < 100000),
                            alpha=0.3, color='red')
    ax_train_pv.set_xlabel('Date', fontsize=12)
    ax_train_pv.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax_train_pv.set_title('TRAINING: Portfolio Growth', fontsize=14, fontweight='bold')
    ax_train_pv.grid(alpha=0.3)
    
    # ============================================================
    # TEST DATA (Right Column)
    # ============================================================
    
    # 1. Test: Cumulative Returns with Regime Background
    ax_test_returns = axes[0, 1]
    test_cum_strategy = (1 + test_results['strategy_returns']).cumprod() - 1
    test_cum_bh = (1 + test_results['returns_simple']).cumprod() - 1 if 'returns_simple' in test_results.columns else None
    
    # Add regime background
    if 'regime' in test_results.columns:
        for regime in test_results['regime'].dropna().unique():
            regime_mask = test_results['regime'] == regime
            regime_label = regime_names_dict.get(int(regime), f'Regime {int(regime)}') if regime_names_dict else f'Regime {int(regime)}'
            ax_test_returns.fill_between(test_results.index,
                                        test_cum_strategy.min() * 1.1,
                                        test_cum_strategy.max() * 1.1,
                                        where=regime_mask,
                                        alpha=0.15,
                                        color=regime_colors.get(int(regime), 'lightgray'),
                                        label=regime_label)
    
    ax_test_returns.plot(test_results.index, test_cum_strategy * 100,
                        linewidth=2, color='blue', label='Strategy')
    if test_cum_bh is not None:
        ax_test_returns.plot(test_results.index, test_cum_bh * 100,
                            linewidth=2, color='gray', alpha=0.7, label='Buy & Hold')
    
    ax_test_returns.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_test_returns.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax_test_returns.set_title('TEST: Cumulative Returns by Regime (Out-of-Sample)', fontsize=14, fontweight='bold')
    ax_test_returns.legend(loc='upper left', fontsize=10)
    ax_test_returns.grid(alpha=0.3)
    
    # 2. Test: Drawdown
    ax_test_dd = axes[1, 1]
    test_cummax = (1 + test_results['strategy_returns']).cumprod().expanding().max()
    test_dd = ((1 + test_results['strategy_returns']).cumprod() - test_cummax) / test_cummax
    
    ax_test_dd.plot(test_results.index, test_dd * 100, linewidth=2, color='red')
    ax_test_dd.fill_between(test_results.index, test_dd * 100, 0, color='red', alpha=0.3)
    ax_test_dd.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_test_dd.set_ylabel('Drawdown (%)', fontsize=12)
    ax_test_dd.set_title('TEST: Strategy Drawdown', fontsize=14, fontweight='bold')
    ax_test_dd.grid(alpha=0.3)
    
    # 3. Test: Portfolio Value
    ax_test_pv = axes[2, 1]
    ax_test_pv.plot(test_results.index, test_results['portfolio_value'],
                   linewidth=2, color='green')
    ax_test_pv.axhline(y=100000, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax_test_pv.fill_between(test_results.index, 100000, test_results['portfolio_value'],
                           where=(test_results['portfolio_value'] >= 100000),
                           alpha=0.3, color='green')
    ax_test_pv.fill_between(test_results.index, 100000, test_results['portfolio_value'],
                           where=(test_results['portfolio_value'] < 100000),
                           alpha=0.3, color='red')
    ax_test_pv.set_xlabel('Date', fontsize=12)
    ax_test_pv.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax_test_pv.set_title('TEST: Portfolio Growth', fontsize=14, fontweight='bold')
    ax_test_pv.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Calculate SPY (buy-and-hold) drawdowns
    if train_cum_bh is not None:
        train_spy_cummax = (1 + train_results['returns_simple']).cumprod().expanding().max()
        train_spy_dd = ((1 + train_results['returns_simple']).cumprod() - train_spy_cummax) / train_spy_cummax
        train_spy_final_value = 100000 * (1 + train_cum_bh.iloc[-1])
    else:
        train_spy_dd = pd.Series([0])
        train_spy_final_value = 100000

    if test_cum_bh is not None:
        test_spy_cummax = (1 + test_results['returns_simple']).cumprod().expanding().max()
        test_spy_dd = ((1 + test_results['returns_simple']).cumprod() - test_spy_cummax) / test_spy_cummax
        test_spy_final_value = 100000 * (1 + test_cum_bh.iloc[-1])
    else:
        test_spy_dd = pd.Series([0])
        test_spy_final_value = 100000

    # Print summary statistics
    print("\n" + "="*100)
    print("SIDE-BY-SIDE COMPARISON SUMMARY")
    print("="*100)
    print(f"\n{'Metric':<30} {'Strategy (Train)':>18} {'Strategy (Test)':>18} {'SPY B&H (Train)':>18} {'SPY B&H (Test)':>18}")
    print("-"*100)
    print(f"{'Final Return (%)':<30} {train_cum_strategy.iloc[-1]*100:>18.2f} {test_cum_strategy.iloc[-1]*100:>18.2f} {train_cum_bh.iloc[-1]*100 if train_cum_bh is not None else 0:>18.2f} {test_cum_bh.iloc[-1]*100 if test_cum_bh is not None else 0:>18.2f}")
    print(f"{'Max Drawdown (%)':<30} {train_dd.min()*100:>18.2f} {test_dd.min()*100:>18.2f} {train_spy_dd.min()*100:>18.2f} {test_spy_dd.min()*100:>18.2f}")
    print(f"{'Final Portfolio ($)':<30} ${train_results['portfolio_value'].iloc[-1]:>17,.2f} ${test_results['portfolio_value'].iloc[-1]:>17,.2f} ${train_spy_final_value:>17,.2f} ${test_spy_final_value:>17,.2f}")
    print("="*100)
    

def mr_sma_pct_candles_compute_regime_aware_strategy(df_with_regimes, regime_params_dict,
                                  initial_cash=100000, position_sizing_method='capital_based',
                                  capital_allocation_pct=0.98, signal_shift=0):
    """
    Compute mean reversion strategy with DYNAMIC parameters based on current regime.

    This handles regime transitions properly - parameters change as regime changes!

    Parameters:
    -----------
    df_with_regimes : DataFrame with 'Adj_Close' and 'regime' columns
    regime_params_dict : dict, format: {regime_id: {'window': X, 'pct_long': Y, 'pct_short': Z, 'min_candles': N}}
        Example: {0: {'window': 20, 'pct_long': -1.5, 'pct_short': 2.0, 'min_candles': 3},
                  1: {'window': 10, 'pct_long': -2.5, 'pct_short': 3.0, 'min_candles': 2}}
    initial_cash : float, starting capital (default=100000)
    position_sizing_method : str, 'fixed' or 'capital_based' (default='capital_based')
    capital_allocation_pct : float, % of capital to use per trade (default=0.98)
    signal_shift : int, number of days to delay signal execution (default=0)
        - 0: execute signals immediately
        - >0: delay signal execution by this many days

    Returns:
    --------
    DataFrame with signals, positions, P&L calculations, and portfolio tracking
    """
    df_signals = df_with_regimes.copy()

    # Initialize columns
    df_signals['dynamic_sma'] = np.nan
    df_signals['pct_distance'] = 0.0
    df_signals['range_upper'] = np.nan
    df_signals['range_lower'] = np.nan
    df_signals['above_sma'] = False
    df_signals['below_sma'] = False
    df_signals['candles_above'] = 0
    df_signals['candles_below'] = 0
    df_signals['entry_long'] = False
    df_signals['exit_long'] = False
    df_signals['entry_short'] = False
    df_signals['exit_short'] = False

    # Calculate indicators dynamically based on regime
    for i in range(len(df_signals)):
        # Get current regime (use regime 0 params if NaN)
        current_regime = df_signals['regime'].iloc[i]
        if pd.isna(current_regime):
            current_regime = 0  # Default to first regime if missing

        # Get regime-specific parameters
        params = regime_params_dict.get(int(current_regime), regime_params_dict[0])
        window = int(params['window'])
        pct_long = float(params['pct_long'])
        pct_short = float(params['pct_short'])
        min_candles = int(params['min_candles'])

        # Calculate percentage distance with regime-specific window
        # Use min_periods=1 to match pandas rolling behavior (calculate from first bar)
        if i >= 0:  # Calculate for all bars (min_periods=1)
            # Get up to 'window' prices including current, but allow fewer for early bars
            start_idx = max(0, i - window + 1)
            prices = df_signals['Adj_Close'].iloc[start_idx:i+1]
            sma = prices.mean()

            df_signals.loc[df_signals.index[i], 'dynamic_sma'] = sma

            # Calculate percentage distance from SMA
            price = df_signals['Adj_Close'].iloc[i]
            if sma > 0:
                pct_distance = ((price - sma) / sma) * 100
                df_signals.loc[df_signals.index[i], 'pct_distance'] = pct_distance
            else:
                pct_distance = 0
                df_signals.loc[df_signals.index[i], 'pct_distance'] = 0

            # Calculate percentage-based threshold bands for visualization
            df_signals.loc[df_signals.index[i], 'range_upper'] = sma * (1 + pct_short / 100)
            df_signals.loc[df_signals.index[i], 'range_lower'] = sma * (1 - pct_long / 100)

            # Track position relative to SMA
            df_signals.loc[df_signals.index[i], 'above_sma'] = (price > sma)
            df_signals.loc[df_signals.index[i], 'below_sma'] = (price < sma)

    # Count consecutive candles above/below SMA using groupby pattern
    df_signals['candles_above'] = (
        df_signals['above_sma']
        .groupby((df_signals['above_sma'] != df_signals['above_sma'].shift()).cumsum())
        .cumsum()
        .astype(int)
    )

    df_signals['candles_below'] = (
        df_signals['below_sma']
        .groupby((df_signals['below_sma'] != df_signals['below_sma'].shift()).cumsum())
        .cumsum()
        .astype(int)
    )

    # Generate signals using regime-specific thresholds
    for i in range(len(df_signals)):
        # Get current regime and parameters again for signal generation
        current_regime = df_signals['regime'].iloc[i]
        if pd.isna(current_regime):
            current_regime = 0

        params = regime_params_dict.get(int(current_regime), regime_params_dict[0])
        pct_long = float(params['pct_long'])
        pct_short = float(params['pct_short'])
        min_candles = int(params['min_candles'])

        pct_dist = df_signals['pct_distance'].iloc[i]
        candles_below = df_signals['candles_below'].iloc[i]
        candles_above = df_signals['candles_above'].iloc[i]

        # Entry Long: price below SMA with both conditions met
        # pct_long is NEGATIVE (e.g., -1.5 means 1.5% below SMA)
        df_signals.loc[df_signals.index[i], 'entry_long'] = (
            (pct_dist <= pct_long) and (candles_below >= min_candles)
        )

        # Exit Long: price crosses back above SMA
        df_signals.loc[df_signals.index[i], 'exit_long'] = (pct_dist >= 0)

        # Entry Short: price above SMA with both conditions met
        # pct_short is POSITIVE (e.g., 2.0 means 2.0% above SMA)
        df_signals.loc[df_signals.index[i], 'entry_short'] = (
            (pct_dist >= pct_short) and (candles_above >= min_candles)
        )

        # Exit Short: price crosses back below SMA
        df_signals.loc[df_signals.index[i], 'exit_short'] = (pct_dist <= 0)

    # Apply signal shift if specified (shift signals forward to delay execution)
    if signal_shift > 0:
        df_signals['entry_long'] = df_signals['entry_long'].shift(signal_shift).fillna(False)
        df_signals['exit_long'] = df_signals['exit_long'].shift(signal_shift).fillna(False)
        df_signals['entry_short'] = df_signals['entry_short'].shift(signal_shift).fillna(False)
        df_signals['exit_short'] = df_signals['exit_short'].shift(signal_shift).fillna(False)

    # ============================================================
    # Calculate Position State and Portfolio Tracking
    # ============================================================
    positions = []
    shares_held = []
    cash_balance = []
    portfolio_values = []
    pnl_daily = []
    
    position = 0  # 0 = flat, 1 = long, -1 = short
    shares = 0
    cash = initial_cash
    entry_price = 0
    
    for i in range(len(df_signals)):
        price = df_signals['Adj_Close'].iloc[i]
        
        # Default to previous state
        new_position = position
        new_shares = shares
        new_cash = cash
        daily_pnl = 0
        
        # Process signals
        if df_signals['entry_long'].iloc[i] and position <= 0:
            # Close short if exists
            cash_after_closing = cash
            if position == -1:
                cover_cost = shares * price
                daily_pnl = shares * (entry_price - price)  # Short P&L (for tracking)
                cash_after_closing = cash - cover_cost  # P&L already implicit in cash flow
                
            # Enter long (use cash after closing short if any)
            if position_sizing_method == 'capital_based':
                shares_to_buy = int((cash_after_closing * capital_allocation_pct) / price)
            else:
                shares_to_buy = 150  # Default position size
            
            cost = shares_to_buy * price
            if cost <= cash_after_closing:
                new_shares = shares_to_buy
                new_cash = cash_after_closing - cost
                new_position = 1
                entry_price = price
                if position != -1:  # Don't double count if closed short
                    daily_pnl = 0
                    
        elif df_signals['exit_long'].iloc[i] and position == 1:
            # Exit long
            proceeds = shares * price
            daily_pnl = shares * (price - entry_price)
            new_cash = cash + proceeds
            new_position = 0
            new_shares = 0
            entry_price = 0
            
        elif df_signals['entry_short'].iloc[i] and position >= 0:
            # Close long if exists
            cash_from_closing = 0
            if position == 1:
                proceeds = shares * price
                daily_pnl = shares * (price - entry_price)  # Long P&L
                cash_from_closing = proceeds
                
            # Enter short (use cash + proceeds from closing long if any)
            available_cash = cash + cash_from_closing
            if position_sizing_method == 'capital_based':
                shares_to_short = int((available_cash * capital_allocation_pct) / price)
            else:
                shares_to_short = 150
            
            if shares_to_short > 0:
                proceeds_from_short = shares_to_short * price
                new_shares = shares_to_short
                new_cash = available_cash + proceeds_from_short
                new_position = -1
                entry_price = price
                if position != 1:  # Don't double count if closed long
                    daily_pnl = 0
                    
        elif df_signals['exit_short'].iloc[i] and position == -1:
            # Exit short: buy back shares to cover
            cover_cost = shares * price
            daily_pnl = shares * (entry_price - price)  # For tracking only
            new_cash = cash - cover_cost  # P&L already implicit in cash flow
            new_position = 0
            new_shares = 0
            entry_price = 0
        
        # Update state
        position = new_position
        shares = new_shares
        cash = new_cash
        
        # Calculate portfolio value
        if position == 1:
            portfolio_value = cash + shares * price
        elif position == -1:
            portfolio_value = cash - shares * price
        else:
            portfolio_value = cash
        
        # Store results
        positions.append(position)
        shares_held.append(shares)
        cash_balance.append(cash)
        portfolio_values.append(portfolio_value)
        pnl_daily.append(daily_pnl)
    
    # Add to dataframe
    df_signals['signal'] = positions
    df_signals['shares'] = shares_held
    df_signals['cash'] = cash_balance
    df_signals['portfolio_value'] = portfolio_values
    df_signals['pnl'] = pnl_daily
    
    # Calculate strategy returns
    df_signals['strategy_returns'] = df_signals['portfolio_value'].pct_change().fillna(0)
    
    return df_signals


def mr_sma_pct_candles_compute_regime_aware_strategy_v2(df_with_regimes, regime_params_dict,
                                    position_size=150, initial_cash=100000,
                                    position_sizing_method='capital_based',
                                    capital_allocation_pct=0.98, signal_shift=0):
    """
    Compute regime-aware mean reversion strategy that EXACTLY mirrors mr_compute_mean_reversion_sma_pct_candle_strategy.
    
    The ONLY difference: window, z_long, z_short change dynamically based on current regime.
    Position tracking, cash management, and signal logic are IDENTICAL to standard version.
    
    Parameters:
    -----------
    df_with_regimes : DataFrame with 'Adj_Close' and 'regime' columns
    regime_params_dict : dict, format: {regime_id: {'window': X, 'z_long': Y, 'z_short': Z}}
        Example: {0: {'window': 20, 'z_long': -1.5, 'z_short': 2.0},
                  1: {'window': 10, 'z_long': -2.5, 'z_short': 3.0}}
    position_size : int, number of shares per trade when using 'fixed' method (default=150)
    initial_cash : float, starting capital (default=100000)
    position_sizing_method : str, 'fixed' or 'capital_based' (default='capital_based')
    capital_allocation_pct : float, % of capital to use per trade (default=0.98)
    signal_shift : int, number of days to delay signal execution (default=0)
        - 0: execute signals immediately
        - >0: delay signal execution by this many days (e.g., signal_shift=1 means trade 1 day after signal)

    Returns:
    --------
    DataFrame with signals, positions, P&L calculations (identical structure to standard strategy)
    """
    df_signals = df_with_regimes.copy()
    
    # Get max window for warmup period
    max_window = max(int(params['window']) for params in regime_params_dict.values())
    
    # ============================================================
    # 1. Calculate Technical Indicators DYNAMICALLY per regime
    # ============================================================
    df_signals['dynamic_sma'] = np.nan
    df_signals['dynamic_std'] = np.nan
    df_signals['z_score'] = 0.0
    df_signals['range_upper'] = np.nan
    df_signals['range_lower'] = np.nan
    df_signals['entry_long'] = False
    df_signals['exit_long'] = False
    df_signals['entry_short'] = False
    df_signals['exit_short'] = False
    
    # Calculate indicators day-by-day with regime-specific parameters
    for i in range(len(df_signals)):
        # Get current regime (default to 0 if NaN)
        current_regime = df_signals['regime'].iloc[i]
        if pd.isna(current_regime):
            current_regime = 0
        
        # Get regime-specific parameters
        params = regime_params_dict.get(int(current_regime), regime_params_dict[0])
        window = int(params['window'])
        z_long = float(params['z_long'])
        z_short = float(params['z_short'])
        
        # Calculate SMA and STD with regime-specific window (using min_periods=1 like pandas rolling)
        if i >= 0:  # Calculate from first bar
            start_idx = max(0, i - window + 1)
            prices = df_signals['Adj_Close'].iloc[start_idx:i+1]
            sma = prices.mean()
            std = prices.std() if len(prices) > 1 else 0.0  # Need 2+ points for std
            
            df_signals.loc[df_signals.index[i], 'dynamic_sma'] = sma
            df_signals.loc[df_signals.index[i], 'dynamic_std'] = std if std > 0 else np.nan
            
            # Calculate z-score
            if std > 0:
                z_score = (df_signals['Adj_Close'].iloc[i] - sma) / std
                df_signals.loc[df_signals.index[i], 'z_score'] = z_score
            else:
                z_score = 0.0
                df_signals.loc[df_signals.index[i], 'z_score'] = 0.0
            
            # Calculate bands
            std_for_bands = std if std > 0 else 0.0
            df_signals.loc[df_signals.index[i], 'range_upper'] = sma + z_short * std_for_bands
            df_signals.loc[df_signals.index[i], 'range_lower'] = sma + z_long * std_for_bands
    
    # ============================================================
    # 2. Generate Entry/Exit Signals (using regime-specific thresholds)
    # ============================================================
    # We need to recalculate signals day-by-day because thresholds change with regime
    for i in range(1, len(df_signals)):  # Start from 1 (need previous z-score)
        # Get current regime
        current_regime = df_signals['regime'].iloc[i]
        if pd.isna(current_regime):
            current_regime = 0
        
        # Get regime-specific thresholds
        params = regime_params_dict.get(int(current_regime), regime_params_dict[0])
        z_long = float(params['z_long'])
        z_short = float(params['z_short'])
        
        z_prev = df_signals['z_score'].iloc[i-1]
        z_curr = df_signals['z_score'].iloc[i]
        
        # Entry Long: z[t-1] > z_threshold_long AND z[t] <= z_threshold_long
        df_signals.loc[df_signals.index[i], 'entry_long'] = (
            (z_prev > z_long) and (z_curr <= z_long)
        )
        
        # Exit Long: z[t-1] < 0 AND z[t] >= 0
        df_signals.loc[df_signals.index[i], 'exit_long'] = (
            (z_prev < 0) and (z_curr >= 0)
        )
        
        # Entry Short: z[t-1] < z_threshold_short AND z[t] >= z_threshold_short
        df_signals.loc[df_signals.index[i], 'entry_short'] = (
            (z_prev < z_short) and (z_curr >= z_short)
        )
        
        # Exit Short: z[t-1] > 0 AND z[t] <= 0
        df_signals.loc[df_signals.index[i], 'exit_short'] = (
            (z_prev > 0) and (z_curr <= 0)
        )

    # Apply signal shift if specified (shift signals forward to delay execution)
    if signal_shift > 0:
        df_signals['entry_long'] = df_signals['entry_long'].shift(signal_shift).fillna(False)
        df_signals['exit_long'] = df_signals['exit_long'].shift(signal_shift).fillna(False)
        df_signals['entry_short'] = df_signals['entry_short'].shift(signal_shift).fillna(False)
        df_signals['exit_short'] = df_signals['exit_short'].shift(signal_shift).fillna(False)

    # ============================================================
    # 3. Calculate Position State and Portfolio Tracking
    # (EXACT MIRROR of mr_compute_mean_reversion_sma_pct_candle_strategy lines 712-799)
    # ============================================================
    positions = []
    shares_held = []
    cash_balance = []
    equity_value = []
    total_portfolio = []
    
    current_position = 0
    current_shares = 0
    current_cash = initial_cash
    
    for i in range(len(df_signals)):
        price = df_signals.iloc[i]['Adj_Close']
        
        # First (max_window-1) days: no positions (indicator lookback period)
        if i < max_window - 1:
            positions.append(0)
            shares_held.append(0)
            cash_balance.append(current_cash)
            equity_value.append(0)
            total_portfolio.append(current_cash)
            continue
        
        # Process exit signals first (priority) - EXACT COPY
        if current_position == 1 and df_signals.iloc[i]['exit_long']:
            # Exit long: sell shares
            current_cash += current_shares * price
            current_shares = 0
            current_position = 0
            
        elif current_position == -1 and df_signals.iloc[i]['exit_short']:
            # Exit short: buy back shares to cover
            current_cash += current_shares * price  # current_shares is negative for short
            current_shares = 0
            current_position = 0
            
        # Process entry signals (only if flat) - EXACT COPY
        elif current_position == 0:
            if df_signals.iloc[i]['entry_long']:
                # Enter long position
                if position_sizing_method == 'capital_based':
                    # Calculate max shares based on available capital
                    available_capital = current_cash * capital_allocation_pct
                    current_shares = int(available_capital / price)
                else:
                    # Use fixed position size
                    current_shares = position_size
                
                # Execute trade if we can afford at least 1 share
                if current_shares > 0:
                    current_cash -= current_shares * price
                    current_position = 1
                else:
                    current_shares = 0
                    
            elif df_signals.iloc[i]['entry_short']:
                # Enter short position
                if position_sizing_method == 'capital_based':
                    # Calculate max shares based on available capital
                    available_capital = current_cash * capital_allocation_pct
                    current_shares = -int(available_capital / price)  # Negative for short
                else:
                    # Use fixed position size (negative for short)
                    current_shares = -position_size
                
                # Execute trade if we can afford at least 1 share
                if current_shares < 0:
                    current_cash -= current_shares * price  # Subtracting negative adds to cash
                    current_position = -1
                else:
                    current_shares = 0
        
        # Record state - EXACT COPY
        positions.append(current_position)
        shares_held.append(current_shares)
        cash_balance.append(current_cash)
        
        # Calculate equity value (mark-to-market) - EXACT COPY
        equity = current_shares * price
        equity_value.append(equity)
        total_portfolio.append(current_cash + equity)
    
    df_signals['signal'] = positions
    df_signals['shares'] = shares_held
    df_signals['cash'] = cash_balance
    df_signals['equity'] = equity_value
    df_signals['portfolio_value'] = total_portfolio
    
    # ============================================================
    # 4. Calculate Returns and PnL - EXACT COPY
    # ============================================================
    df_signals['returns'] = df_signals['Adj_Close'].pct_change()
    df_signals['position'] = df_signals['signal'].shift(1).fillna(0)
    df_signals['strategy_returns'] = df_signals['returns'] * df_signals['position']
    
    # Calculate daily PnL from portfolio value changes
    df_signals['pnl'] = df_signals['portfolio_value'].diff()
    
    # Calculate cumulative PnL
    df_signals['cumulative_pnl'] = df_signals['portfolio_value'] - initial_cash
    
    return df_signals


def backtest_mean_reversion_sma_pct_candle_strategy(spy_train_features, spy_test_features):
    nclust = 5
    if nclust == 2:
        regime_names = {0: 'Low Vol', 1: 'High Vol'}
    elif nclust == 3:
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
    elif nclust == 4:
        regime_names = {0: 'Very Low Vol', 1: 'Low Vol', 2: 'Med Vol', 3: 'High Vol'}
    elif nclust == 5:
        regime_names = {
        0: 'Very Low Vol (Calm)',
        1: 'Low Vol (Stable)', 
        2: 'Medium Vol (Normal)',
        3: 'High Vol (Elevated)',
        4: 'Very High Vol (Crisis)'
        }
    else:
        regime_names = {i: f'Regime {i}' for i in range(nclust)}
        
    print("LOADING SMA PCT DISTANCE CANDLE + REGIME PARAMETERS FROM CSV FILES (FROM TRAINING DATA)")
    # Initialize regime_best_params dictionary (in case optimization section was skipped)
    regime_best_params = {}

    for regime_id in range(5):  # Assuming 5 regimes (0-4)
        csv_path = os.path.join(f'mr_sma_pct_distance_candle_hyperparameter_results_regime_{regime_id}.csv')

        if os.path.exists(csv_path):
            # Read CSV and get best parameters (first row)
            regime_csv = pd.read_csv(csv_path)

            if not regime_csv.empty:
                best_row = regime_csv.iloc[0]
                regime_best_params[regime_id] = best_row.to_dict()

                regime_name = regime_names.get(regime_id, f'Regime {regime_id}')
                print(f"✅ Loaded {regime_name}: Window={best_row['Window']}, Pct_Long={best_row['Pct_Long']}, Pct_Short={best_row['Pct_Short']}, Min_Candles={best_row['Min_Candles']}")
            else:
                print(f"⚠️  Warning: Empty CSV for regime {regime_id}")
        else:
            print(f"❌ CSV not found: {csv_path}")

    print(f"\n✅ Loaded parameters for {len(regime_best_params)} regimes from CSV")
    print("="*70)


    print("\n" + "="*70)
    print("UNIFIED REGIME-AWARE BACKTEST - TRAINING DATA")
    print("="*70)
    print(f"\nTraining Data Period:")
    print(f"  Start Date: {spy_train_features.index[0].strftime('%Y-%m-%d')}")
    print(f"  End Date:   {spy_train_features.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Days: {len(spy_train_features)}")
    print("\nTesting strategy on FULL training data with regime transitions...")

    # Prepare regime parameters dictionary from optimization results
    regime_params_for_strategy = {}
    for regime_id, best_params in regime_best_params.items():
        regime_params_for_strategy[int(regime_id)] = {
            'window': best_params['Window'],
            'pct_long': best_params['Pct_Long'],
            'pct_short': best_params['Pct_Short'],
            'min_candles': best_params['Min_Candles']
        }

    print("\nRegime Parameters Being Used:")
    for regime_id, params in regime_params_for_strategy.items():
        regime_name = regime_names.get(regime_id, f'Regime {regime_id}')
        print(f"  {regime_name}: window={params['window']}, pct_long={params['pct_long']}, pct_short={params['pct_short']}, min_candles={params['min_candles']}")

    # Run unified backtest on full training data
    train_unified_results = mr_sma_pct_candles_compute_regime_aware_strategy(
        spy_train_features,
        regime_params_for_strategy,
        initial_cash=100000,
        position_sizing_method='capital_based',
        capital_allocation_pct=0.98
    )

    # Calculate metrics
    train_unified_metrics = mr_sma_pct_candle_calculate_performance_metrics(train_unified_results, initial_cash=100000)

    print("\n" + "="*70)
    print("UNIFIED BACKTEST RESULTS - TRAINING DATA")
    print("="*70)
    print(f"Total Return:     {train_unified_metrics['Total Return (%)']:.2f}%")
    print(f"CAGR:             {train_unified_metrics['CAGR (%)']:.2f}%")
    print(f"Sharpe Ratio:     {train_unified_metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:     {train_unified_metrics['Max Drawdown (%)']:.2f}%")
    print(f"Total Trades:     {train_unified_metrics['Total Trades']}")
    print("="*70)

    # %%
    # ============================================================
    # RUN UNIFIED REGIME-AWARE BACKTEST ON TEST DATA
    # ============================================================

    print("\n" + "="*70)
    print("UNIFIED REGIME-AWARE BACKTEST - TEST DATA")
    print("="*70)
    print(f"\nTest Data Period:")
    print(f"  Start Date: {spy_test_features.index[0].strftime('%Y-%m-%d')}")
    print(f"  End Date:   {spy_test_features.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Days: {len(spy_test_features)}")
    print("\nTesting strategy on FULL test data with regime transitions...")

    # Run unified backtest on full test data (using same params from training)
    test_unified_results = mr_sma_pct_candles_compute_regime_aware_strategy(
        spy_test_features,
        regime_params_for_strategy,
        initial_cash=100000,
        position_sizing_method='capital_based',
        capital_allocation_pct=0.98
    )

    # Calculate metrics
    test_unified_metrics = mr_sma_pct_candle_calculate_performance_metrics(test_unified_results, initial_cash=100000)

    print("\n" + "="*70)
    print("UNIFIED BACKTEST RESULTS - TEST DATA")
    print("="*70)
    print(f"Total Return:     {test_unified_metrics['Total Return (%)']:.2f}%")
    print(f"CAGR:             {test_unified_metrics['CAGR (%)']:.2f}%")
    print(f"Sharpe Ratio:     {test_unified_metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:     {test_unified_metrics['Max Drawdown (%)']:.2f}%")
    print(f"Total Trades:     {test_unified_metrics['Total Trades']}")
    print("="*70)

    # %%
    # ============================================================
    # COMPARE: REGIME-AWARE VS REGIME-AGNOSTIC
    # ============================================================

    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: TRAIN vs TEST")
    print("="*70)

    comparison_table = pd.DataFrame({
        'Metric': ['Total Return (%)', 'CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Total Trades'],
        'Train': [
            f"{train_unified_metrics['Total Return (%)']:.2f}",
            f"{train_unified_metrics['CAGR (%)']:.2f}",
            f"{train_unified_metrics['Sharpe Ratio']:.2f}",
            f"{train_unified_metrics['Max Drawdown (%)']:.2f}",
            f"{train_unified_metrics['Total Trades']}"
        ],
        'Test': [
            f"{test_unified_metrics['Total Return (%)']:.2f}",
            f"{test_unified_metrics['CAGR (%)']:.2f}",
            f"{test_unified_metrics['Sharpe Ratio']:.2f}",
            f"{test_unified_metrics['Max Drawdown (%)']:.2f}",
            f"{test_unified_metrics['Total Trades']}"
        ]
    })

    display(comparison_table)

    print("\n💡 INTERPRETATION:")
    print("   - Similar Sharpe ratios → Good generalization, no overfitting")
    print("   - Test Sharpe much lower → May be overfitted to train data")
    print("   - Test Sharpe higher → Got lucky with test period or train was more volatile")
    print("="*70)

    #%%
    # ============================================================
    # VISUALIZE TRAIN VS TEST COMPARISON WITH REGIME OVERLAY
    # ============================================================

    print("\n" + "="*70)
    print("GENERATING TRAIN vs TEST VISUALIZATION WITH REGIME OVERLAY...")
    print("="*70)

    # Create regime names dict based on actual number of clusters
    # Current setting: nclust = 5 (see line 316)
    if nclust == 2:
        regime_names = {0: 'Low Vol', 1: 'High Vol'}
    elif nclust == 3:
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
    elif nclust == 4:
        regime_names = {0: 'Very Low Vol', 1: 'Low Vol', 2: 'Med Vol', 3: 'High Vol'}
    elif nclust == 5:
        regime_names = {
            0: 'Very Low Vol (Calm)',
            1: 'Low Vol (Stable)', 
            2: 'Medium Vol (Normal)',
            3: 'High Vol (Elevated)',
            4: 'Very High Vol (Crisis)'
        }
    else:
        regime_names = {i: f'Regime {i}' for i in range(nclust)}

    # Call the comparison visualization
    mr_sma_pct_candle_viz_regime_aware_backtest_comparison(
        train_unified_results,
        test_unified_results,
        regime_names_dict=regime_names
    )
    return test_unified_results


def viz_mr_sma_pct_candle_regime_aware_backtest_signals(test_unified_results):
    nclust = 5
    if nclust == 2:
        regime_names = {0: 'Low Vol', 1: 'High Vol'}
    elif nclust == 3:
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
    elif nclust == 4:
        regime_names = {0: 'Very Low Vol', 1: 'Low Vol', 2: 'Med Vol', 3: 'High Vol'}
    elif nclust == 5:
        regime_names = {
        0: 'Very Low Vol (Calm)',
        1: 'Low Vol (Stable)', 
        2: 'Medium Vol (Normal)',
        3: 'High Vol (Elevated)',
        4: 'Very High Vol (Crisis)'
        }
    else:
        regime_names = {i: f'Regime {i}' for i in range(nclust)}
        

    print("\n" + "="*70)
    print("TEST DATA: Entry/Exit Signals with Regime Background (Year by Year)")
    print("="*70)

    # Get unique years in test data
    test_years = test_unified_results.index.year.unique()
    print(f"\nTest data spans {len(test_years)} years: {test_years.min()} - {test_years.max()}")
    print(f"Generating {len(test_years)} separate visualizations...\n")

    # Visualize each year separately
    for year in sorted(test_years):
        print(f"\n{'='*70}")
        print(f"YEAR {year}")
        print(f"{'='*70}")
        
        # Define start and end date for the year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        # Filter data for this year
        year_data = test_unified_results[test_unified_results.index.year == year]
        
        if len(year_data) > 0:
            print(f"Trading days: {len(year_data)}")
            print(f"Date range: {year_data.index[0].strftime('%Y-%m-%d')} to {year_data.index[-1].strftime('%Y-%m-%d')}")
            
            # Visualize this year
            mr_sma_pct_candle_regime_aware_backtest_signals(
                test_unified_results,
                start_date=start_date,
                end_date=end_date,
                regime_names_dict=regime_names
            )
        else:
            print(f"No data available for {year}")

    print("\n" + "="*70)
    print("COMPLETED: All yearly visualizations generated")
    print("="*70)
    
    

def viz_mr_sma_bb_viz_regime_aware_backtest_signals(test_unified_results):

    nclust = 5
    if nclust == 2:
        regime_names = {0: 'Low Vol', 1: 'High Vol'}
    elif nclust == 3:
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
    elif nclust == 4:
        regime_names = {0: 'Very Low Vol', 1: 'Low Vol', 2: 'Med Vol', 3: 'High Vol'}
    elif nclust == 5:
        regime_names = {
        0: 'Very Low Vol (Calm)',
        1: 'Low Vol (Stable)', 
        2: 'Medium Vol (Normal)',
        3: 'High Vol (Elevated)',
        4: 'Very High Vol (Crisis)'
        }
    else:
        regime_names = {i: f'Regime {i}' for i in range(nclust)}
        
    print("\n" + "="*70)
    print("TEST DATA: Entry/Exit Signals with Regime Background (Year by Year)")
    print("="*70)

    # Get unique years in test data
    test_years = test_unified_results.index.year.unique()
    print(f"\nTest data spans {len(test_years)} years: {test_years.min()} - {test_years.max()}")
    print(f"Generating {len(test_years)} separate visualizations...\n")

    # Visualize each year separately
    for year in sorted(test_years):
        print(f"\n{'='*70}")
        print(f"YEAR {year}")
        print(f"{'='*70}")
        
        # Define start and end date for the year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        # Filter data for this year
        year_data = test_unified_results[test_unified_results.index.year == year]
        
        if len(year_data) > 0:
            print(f"Trading days: {len(year_data)}")
            print(f"Date range: {year_data.index[0].strftime('%Y-%m-%d')} to {year_data.index[-1].strftime('%Y-%m-%d')}")
            
            # Visualize this year
            mr_sma_bb_viz_regime_aware_backtest_signals(
                test_unified_results,
                start_date=start_date,
                end_date=end_date,
                regime_names_dict=regime_names
            )
        else:
            print(f"No data available for {year}")

    print("\n" + "="*70)
    print("COMPLETED: All yearly visualizations generated")
    print("="*70)