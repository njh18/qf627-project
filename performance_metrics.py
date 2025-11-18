# performance_metrics.py
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Convert log returns → simple returns
# -------------------------------------------------------------------
def log_to_simple(log_ret: pd.Series) -> pd.Series:
    return np.exp(log_ret) - 1


# -------------------------------------------------------------------
# Sharpe Ratio  (simple returns)
# -------------------------------------------------------------------
def compute_sharpe_ratio(log_returns, risk_free_rate=0.0, periods_per_year=252):
    simple = log_to_simple(log_returns.dropna())
    excess = simple - (risk_free_rate / periods_per_year)

    return (
        excess.mean() / excess.std()
    ) * np.sqrt(periods_per_year)


# -------------------------------------------------------------------
# Sortino Ratio (simple returns)
# -------------------------------------------------------------------
def compute_sortino_ratio(log_returns, periods_per_year=252):
    simple = log_to_simple(log_returns.dropna())

    downside = simple.copy()
    downside[downside > 0] = 0

    downside_std = downside.std()

    return (simple.mean() / downside_std) * np.sqrt(periods_per_year)


# -------------------------------------------------------------------
# Compute Drawdown
# -------------------------------------------------------------------
def compute_drawdown(log_returns):
    equity = np.exp(log_returns.cumsum())
    running_max = equity.cummax()
    dd = (running_max - equity) / running_max
    return dd


# -------------------------------------------------------------------
# ALL Drawdown Periods
# -------------------------------------------------------------------
def compute_all_drawdown_periods(log_returns):
    dd = compute_drawdown(log_returns)
    recoveries = dd[dd == 0.0]

    recovery_points = pd.concat([recoveries, dd.tail(1)]).reset_index()
    recovery_points["previous"] = recovery_points["index"].shift(1)
    recovery_points = recovery_points.dropna()

    periods = []
    for _, row in recovery_points.iterrows():
        start, end = row["previous"], row["index"]
        segment = dd.loc[start:end]
        periods.append({
            "start": start,
            "end": end,
            "duration_days": (end - start).days,
            "max_drawdown": segment.max()
        })

    return pd.DataFrame(periods).sort_values("duration_days", ascending=False)


# -------------------------------------------------------------------
# CAGR for log returns
# -------------------------------------------------------------------
def compute_cagr_from_log(log_returns):
    log_returns = log_returns.dropna()

    start_date = log_returns.index[0]
    end_date   = log_returns.index[-1]

    years = (end_date - start_date).days / 365.25

    total_growth = np.exp(log_returns.sum())

    return total_growth ** (1 / years) - 1


# -------------------------------------------------------------------
# Annualized Volatility (simple or log ok)
# -------------------------------------------------------------------
def compute_annualized_vol(log_returns):
    simple = log_to_simple(log_returns.dropna())
    return simple.std() * np.sqrt(252)


# -------------------------------------------------------------------
# Final Portfolio Value
# -------------------------------------------------------------------
def compute_final_portfolio_value(log_returns, initial_capital=100_000):
    return initial_capital * np.exp(log_returns.sum())


# -------------------------------------------------------------------
# Master Performance Table
# -------------------------------------------------------------------
def compute_performance_metrics(log_returns, 
                                initial_capital,
                                strategy = 'ML'):
    log_returns = log_returns.dropna()


    mdd = compute_drawdown(log_returns).max()
    cagr = compute_cagr_from_log(log_returns)

    cumulative_return =\
    (
        np.exp(
            log_returns
            .cumsum()
        )
    ).iloc[-1]

    metrics = {
        "CAGR": cagr,
        "Volatility": compute_annualized_vol(log_returns),
        "Max Drawdown": mdd,
        "Sharpe Ratio": compute_sharpe_ratio(log_returns),
        "Sortino Ratio": compute_sortino_ratio(log_returns),
        "Calmar Ratio": cagr / abs(mdd) if mdd != 0 else np.nan,
        "Final Portfolio Value": compute_final_portfolio_value(log_returns, initial_capital=initial_capital),
        "Cumulative Return": cumulative_return
    }

    df = pd.DataFrame(metrics, index=[strategy])

    # to clean up all the numeric columns to max 3 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].map(lambda x: round(x, 3))
    
    return df
