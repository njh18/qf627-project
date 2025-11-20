import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed


def SMA(
    df: pd.DataFrame,
    windows: list[int],
    price_column: str = "Close",
    allow_short: bool = False
) -> pd.DataFrame:

    if len(windows) != 2:
        raise ValueError("windows must be a list of two integers: [short_window, long_window]")

    short_w, long_w = windows

    if long_w <= short_w:
        raise ValueError("Long SMA window must be greater than short SMA window.")

    df = df.copy()

    # --- Compute moving averages ---
    df[f"SMA{short_w}"] = df[price_column].rolling(short_w).mean()
    df[f"SMA{long_w}"]  = df[price_column].rolling(long_w).mean()


    signal_vals = [1, -1] if allow_short else [1, 0]

    # --- Signal generation ---
    # Bullish = short SMA > long SMA
    # Bearish = short SMA < long SMA
    df["signal"] = np.select(
        [
            df[f"SMA{short_w}"] > df[f"SMA{long_w}"],   # bullish
            df[f"SMA{short_w}"] < df[f"SMA{long_w}"],   # bearish
        ],
        signal_vals,
        default=np.nan
    )

    df["positions"] =\
    ( 
        df['signal']
        .ffill()
        .shift(1)
        .fillna(0)
    )

    return df


def compute_portfolio(
    df: pd.DataFrame,
    price_column: str,
    position_column: str,
    initial_capital: float = 100_000.0,
    cost_per_trade: float = 0.0,
) -> pd.DataFrame:

    df = df.copy()

    # Initialize
    df["our_cash"] = initial_capital
    df["shares"] = 0.0
    df["trade_flag"] = df[position_column].diff().fillna(0)
    df["transaction_cost"] = 0.0

    for i in range(1, len(df)):
        prev_cash = df["our_cash"].iloc[i-1]
        prev_shares = df["shares"].iloc[i-1]
        price = df[price_column].iloc[i]

        prev_pos = df[position_column].iloc[i-1]
        new_pos = df[position_column].iloc[i]

        cash = prev_cash
        shares = prev_shares

        # --- CASE 0: No position change ---
        if new_pos == prev_pos:
            df.loc[df.index[i], "our_cash"] = cash
            df.loc[df.index[i], "shares"] = shares
            continue

        # --- CASE 1: Move to FLAT (0) ---
        if new_pos == 0:
            # If previously long → sell all shares
            if prev_pos == 1:
                cash += prev_shares * price - cost_per_trade

            # If previously short → buy back shares
            elif prev_pos == -1:
                cash -= abs(prev_shares) * price + cost_per_trade

            shares = 0.0

        # --- CASE 2: Move to LONG (1) ---
        elif new_pos == 1:
            # Cover short first
            if prev_pos == -1:
                cash -= abs(prev_shares) * price + cost_per_trade
                prev_shares = 0.0

            # Now buy with all available cash
            shares_to_buy = int(cash // price)
            cash -= shares_to_buy * price + cost_per_trade
            shares = shares_to_buy

        # --- CASE 3: Move to SHORT (-1) ---
        elif new_pos == -1:
            # Sell long first
            if prev_pos == 1:
                cash += prev_shares * price - cost_per_trade
                prev_shares = 0.0

            # Now short: borrow shares & sell them
            shares_to_short = int(cash // price)
            cash += shares_to_short * price - cost_per_trade
            shares = -shares_to_short

        df.loc[df.index[i], "our_cash"] = cash
        df.loc[df.index[i], "shares"] = shares
        df.loc[df.index[i], "transaction_cost"] = cost_per_trade

    # --- Compute final portfolio value ---
    df["our_holdings"] = df["shares"] * df[price_column]
    df["total"] = df["our_cash"] + df["our_holdings"]

    # Log returns
    df["strategy_returns"] = np.log(df["total"] / df["total"].shift(1)).fillna(0)
    df["cumulative_strategy_returns"] = np.exp(df["strategy_returns"].cumsum())

    return df


def evaluate_momentum_params(
    short_w: int,
    long_w: int,
    df_train: pd.DataFrame,
    price_col: str,
    starting_capital: float,
    allow_short: bool,
    target_regime: int | None = None,
):
    """
    Computes the momentum strategy using specific SMA windows.
    """

    windows = [short_w, long_w]
    df = df_train.copy()

    # 1. Run SMA Momentum
    df = SMA(df, windows, price_column=price_col, allow_short=allow_short)

    # 2. If regime-aware, gate trading
    if target_regime is not None:
        mask = df["regime"] != target_regime
        df.loc[mask, "positions"] = 0

    # 3. Portfolio
    df = compute_portfolio(
        df,
        price_column=price_col,
        position_column="positions",
        initial_capital=starting_capital,
        cost_per_trade=0.0
    )

    final_value = df["cumulative_strategy_returns"].iloc[-1]
    ret = df["strategy_returns"]
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else np.nan

    return {
        "short_window": short_w,
        "long_window": long_w,
        "regime": target_regime,
        "cumulative_return": final_value,
        "sharpe": sharpe,
    }


def run_momentum_hyperparam_search(
    df: pd.DataFrame,
    price_col: str,
    momentum_task_list: list,
    starting_capital: float,
    allow_short: bool,
    cache_file: str,
):

    if os.path.exists(cache_file):
        print(f"Loaded cached Momentum results from: {cache_file}")
        return pd.read_csv(cache_file, index_col=0)

    print(f"Total Momentum permutations: {len(momentum_task_list)}")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_momentum_params)(
            short_w, long_w,
            df_train=df,
            price_col=price_col,
            starting_capital=starting_capital,
            allow_short=allow_short,
            target_regime=None
        )
        for (short_w, long_w) in momentum_task_list
    )

    results_df = (
        pd.DataFrame(results)
        .sort_values("cumulative_return", ascending=False)
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    results_df.to_csv(cache_file)
    print(f"Saved Momentum results to: {cache_file}")

    return results_df
