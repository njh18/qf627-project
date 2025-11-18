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

def compute_portfolio_fixed_shares(
    df: pd.DataFrame,
    num_of_shares: int,
    price_column: str,
    position_column: str,
    initial_capital: float,
    cost_per_trade: float
) -> pd.DataFrame:
    df = df.copy()

    df["total_shares"] =\
    (
        df[position_column] * num_of_shares
    )

    df["diff_in_shares_owned"]=\
    (
        df["total_shares"]
        .diff()
        .fillna(0)
    )

    df['num_of_trades'] =\
    (
        np.abs(
            df[position_column]
            .diff()
        )
    )

    df['transaction_cost'] =\
    (
        df['num_of_trades']
        *
        cost_per_trade
    )


    df['cumulative_transaction_cost'] =\
    (
        df['transaction_cost']
        .cumsum()
    )

    df['our_cash'] =\
    (
        initial_capital 
        - (
            df['diff_in_shares_owned']
            *
            df[price_column]
        ).cumsum() 
        - df['cumulative_transaction_cost']
    )

    df["our_holdings"] =\
    (
        df["total_shares"] * df[price_column]
    )

    df["total"] =\
    (
        df["our_holdings"] + df["our_cash"]
    )

    df["strategy_returns"] =\
    (
        np.log(df["total"] 
               / df["total"].shift(1)
        )
    )

    df["cumulative_strategy_returns"] =\
    (
        np.exp(
            df["strategy_returns"]
            .cumsum()
        )
    )

    return df


def compute_portfolio(
    df: pd.DataFrame,
    price_column: str,
    position_column: str,
    initial_capital: float = 100_000.0,
    cost_per_trade: float = 0.0
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
        pos_change = df["trade_flag"].iloc[i]

        cash = prev_cash
        shares = prev_shares
        cost = 0

        # ---- BUY signal (0 → 1) ----
        if pos_change == 1:
            # Buy maximum shares possible
            shares_to_buy = int(prev_cash // price)
            cost = shares_to_buy * price + cost_per_trade
            cash = prev_cash - cost
            shares = prev_shares + shares_to_buy

        # ---- SELL signal (1 → 0) ----
        elif pos_change == -1:
            # Sell everything
            cash = prev_cash + prev_shares * price - cost_per_trade
            shares = 0
            cost = cost_per_trade

        # ---- No change (stay long or stay flat) ----
        else:
            cash = prev_cash
            shares = prev_shares

        df.loc[df.index[i], "our_cash"] = cash
        df.loc[df.index[i], "shares"] = shares
        df.loc[df.index[i], "transaction_cost"] = cost_per_trade if pos_change != 0 else 0

    # ---- Compute portfolio value ----
    df["our_holdings"] = df["shares"] * df[price_column]
    df["total"] = df["our_cash"] + df["our_holdings"]

    # ---- Compute returns ----
    df["strategy_returns"] =\
    (
        np.log(df["total"] / df["total"].shift(1))
    )

    df["cumulative_strategy_returns"] =\
    (
        np.exp(
            df["strategy_returns"]
            .cumsum()
        )
    )

    return df
