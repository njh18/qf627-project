import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed


# ------------------------------------------------------------
# BASE MACD
# ------------------------------------------------------------
def compute_MACD(
    df: pd.DataFrame,
    span: list[int, int, int],
    price_column: str = "Close"
) -> pd.DataFrame:

    df = df.copy()

    df[f"{span[0]}_ewma"] = df[price_column].ewm(span=span[0]).mean()
    df[f"{span[1]}_ewma"] = df[price_column].ewm(span=span[1]).mean()

    df["macd"] = df[f"{span[0]}_ewma"] - df[f"{span[1]}_ewma"]
    df["macd_signal_line"] = df["macd"].ewm(span=span[2]).mean()
    return df


# ------------------------------------------------------------
# SIGNAL GENERATION
# ------------------------------------------------------------
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

    signal_vals = [1, -1] if allow_short else [1, 0]

    df["signal"] =\
    (
        np.select(
        [
            (df["cross"]) & (df["macd_slope"] > 0),  # bullish
            (df["cross"]) & (df["macd_slope"] < 0),  # bearish
        ],
        signal_vals,
        default=np.nan)
    )
    return df

# ------------------------------------------------------------
# POSITION CREATION (ffill + shift)
# ------------------------------------------------------------
def generate_position(
    df: pd.DataFrame,
    signal_column: str = 'signal',
):
    df = df.copy()
    df["positions"] = df[signal_column].ffill().shift(1).fillna(0)
    return df



# ------------------------------------------------------------
# PORTFOLIO ENGINE (LONG + SHORT)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Baseline MACD
# ------------------------------------------------------------
def MACD(
    df: pd.DataFrame,
    span: list,
    price_col: str,
    allow_short: bool
):
    df = df.copy()

    df = compute_MACD(df, span, price_column=price_col)

    # 2. Signals
    df = generate_signal(df, allow_short=allow_short)
    df = generate_position(df)

    return df

# ------------------------------------------------------------
# Evaluate MACD Params
# ------------------------------------------------------------
def evaluate_macd_params(
    short_ma: int,
    long_ma: int,
    signal: int,
    df_train: pd.DataFrame,
    price_col: str,
    starting_capital: float,
    allow_short: bool,
    target_regime: int | None = None
):
    spans = (short_ma, long_ma, signal)
    df = df_train.copy()

    df = MACD(df, list(spans), price_col=price_col, allow_short=allow_short)

    # 3. Regime gating: only trade in target regime
    if target_regime is not None:
        # wherever regime != target_regime, force flat position
        mask = (df["regime"] != target_regime)
        df.loc[mask, "positions"] = 0


    # 4. Portfolio
    df = compute_portfolio(
        df,
        price_column=price_col,
        position_column="positions",
        initial_capital=starting_capital,
        cost_per_trade=0.0
    )

    final_cum = df["cumulative_strategy_returns"].iloc[-1]
    ret = df["strategy_returns"]

    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else np.nan

    return {
        "short_ma": short_ma,
        "long_ma": long_ma,
        "signal": signal,
        "regime": target_regime,
        "cumulative_return": final_cum,
        "sharpe": sharpe
    }


# ------------------------------------------------------------
# Hyper parameterize MACD
# ------------------------------------------------------------

def run_macd_hyperparam_search(
    df: pd.DataFrame,
    price_col: str,
    macd_task_list: list,
    starting_capital: float,
    allow_short: bool,
    cache_file: str,
):

    # ---- If cached, load and return immediately ----
    if os.path.exists(cache_file):
        print(f"Loaded cached results from: {cache_file}")
        return pd.read_csv(cache_file, index_col=0)

    print(f"Total MACD permutations: {len(macd_task_list)}")

    # ---- Parallel execution ----
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_macd_params)(
            short_ma, long_ma, signal,
            df_train=df,
            price_col=price_col,
            starting_capital=starting_capital,
            allow_short=allow_short,
        )
        for (short_ma, long_ma, signal) in macd_task_list
    )

    # ---- Convert to DataFrame ----
    results_df = (
        pd.DataFrame(results)
        .sort_values("cumulative_return", ascending=False)
        .reset_index(drop=True)
    )

    # ---- Cache results ----
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    results_df.to_csv(cache_file)
    print(f"Saved results to cache: {cache_file}")

    return results_df


def run_macd_hyperparam_search_with_regimes(
    df: pd.DataFrame,
    price_col: str,
    regimes: list,
    macd_task_list: list,
    starting_capital: float,
    allow_short: bool,
    cache_file: str,
):


    if os.path.exists(cache_file):
        results_df = pd.read_csv(cache_file, index_col=0)
    else:
        all_results = []    
        for r in regimes:
            regime_results = Parallel(n_jobs=-1, verbose=10)(
                delayed(evaluate_macd_params)(
                    short_ma, long_ma, signal,
                    df_train=df,
                    price_col=price_col,
                    starting_capital=starting_capital,
                    allow_short=allow_short,
                    target_regime=int(r)
                )
                for short_ma, long_ma, signal in macd_task_list
            )

            for item in regime_results:
                item["regime"] = r
            
            all_results.extend(regime_results)

        results_df = (
            pd.DataFrame(all_results)
            .sort_values(by=["regime", "cumulative_return"], ascending=[True, False])
        )
        results_df.to_csv(cache_file)

    return results_df