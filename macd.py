import pandas as pd
import numpy as np

def compute_MACD(
    df: pd.DataFrame,
    span: list[int, int, int],
    price_column: str = "Close"

    ) -> pd.DataFrame:

    df[f"{span[0]}_ewma"] =\
    (
        df
        [price_column]
        .ewm(span = span[0]
            )
        .mean()
    )

    df[f"{span[1]}_ewma"] =\
    (
        df
        [price_column]
        .ewm(span = span[1]
            )
        .mean()
    )

    df["macd"] =\
    (
        df[f"{span[0]}_ewma"]
        -
        df[f"{span[1]}_ewma"]

    )

    df["macd_signal_line"] =\
    (
        df
        ["macd"]
        .ewm(span = span[2]
            )
        .mean()
    )
    return df


def generate_signal(
    df: pd.DataFrame,
    allow_short: bool = True
):
    df = df.copy()

    df["difference"] = df["macd"] - df["macd_signal_line"]


    df["cross"] =\
    (
        np.sign(
            df["difference"] 
            * df["difference"].shift(1)
        ) == -1 # this will trigger as long as 1 is neg 1 is postive
    )

    df["macd_slope"] =\
    (
        np
        .sign(
            df["macd"].diff()
        ) # determines if its bulissh or bearish
    )

    df["signal"] =\
    (
        np.select(
        [
            (df["cross"]) & (df["macd_slope"] > 0),  # bullish
            (df["cross"]) & (df["macd_slope"] < 0),  # bearish
        ],
        [1, -1],
        default=np.nan)
    )

    df['positions'] = 0


    if allow_short:
        df["positions"] =\
        (
            df["signal"]
            .ffill()
            .shift(1)
            .fillna(0)
        )
    else:
        df["positions"] =\
        (
            np.select(
                [
                    (df["signal"] == 1),  
                    (df["signal"] == -1), 
                ],
                [1, 0],
                default=np.nan
            )  
        )
        df["positions"] = df["positions"].ffill().shift(1).fillna(0)

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
    df["shares"] = 0
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

            shares = 0

        # --- CASE 2: Move to LONG (1) ---
        elif new_pos == 1:
            # Cover short first
            if prev_pos == -1:
                cash -= abs(prev_shares) * price + cost_per_trade
                prev_shares = 0

            # Now buy with all available cash
            shares_to_buy = int(cash // price)
            cash -= shares_to_buy * price + cost_per_trade
            shares = shares_to_buy

        # --- CASE 3: Move to SHORT (-1) ---
        elif new_pos == -1:
            # Sell long first
            if prev_pos == 1:
                cash += prev_shares * price - cost_per_trade
                prev_shares = 0

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
