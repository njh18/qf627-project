from lets_plot import *
import numpy as np
import pandas as pd

def compute_cumulative_returns(df, 
                       position_column = 'positions',
                       price_column = 'Close'):
    df = df.copy()

    df["passive_returns"] =\
    (
        np
        .log(df[price_column]
            /
            df[price_column].shift(1)
            )
    )

    df['strategy_returns'] =\
    (
        df[position_column].shift(1)
        *
        df['passive_returns']
    )

    df['cum_log_returns'] =\
    (
        df['strategy_returns']
        .cumsum()
    )

    df['cumulative_returns_strategy'] =\
    (
        np.exp(
            df['cum_log_returns']
        )
    )

    df['cumulative_max_strategy'] =\
    (
        df['cumulative_returns_strategy']
        .cummax()
    )

    df['cumulative_returns_passive'] =\
    (
        np.exp(
            df['passive_returns']
            .cumsum()
        )
    )
    return df

def plot_cumulative_returns(df,
                               date_column = 'Date',
                               strategy_column = 'cumulative_returns_strategy',
                               passive_column = 'cumulative_returns_passive'):
    
    melted = (
        df[[date_column, strategy_column, passive_column]]
        .melt(id_vars=date_column, var_name="Series", value_name="Value")
    )

    # Build ggplot
    plot = (
        ggplot(melted, aes(x=date_column, y="Value", color="Series"))
        + geom_line(size=1)
        + scale_color_manual(values={strategy_column: "blue", passive_column: "red"})
        + labs(
            title="Cumulative Returns Comparison",
            x="Date",
            y="Cumulative Return",
            color="Legend"
        )
        + theme(legend_position="top")
        + ggsize(1200, 500)
    )

    return plot

def generate_passive_returns(df: pd.DataFrame):
    df = df.copy()

    df["passive_returns"] =\
    (
        np
        .log(df['Close']
            /
            df['Close'].shift(1)
            )
    )

    df['cumulative_passive_returns'] =\
    (
        np.exp(
            df['passive_returns']
            .fillna(0)
            .cumsum()
        )
    )
    return df

def plot_returns(df, y,
                date_column = 'Date',
                returns_column = 'cumulative_passive_returns'):
    
    melted = (
        df[[date_column, returns_column]]
        .melt(id_vars=date_column, var_name="Series", value_name="Value")
    )

    # Build ggplot
    plot = (
        ggplot(melted, aes(x=date_column, y="Value", color="Series"))
        + geom_line(size=1)
        + scale_color_manual(values={returns_column: "blue"})
        + labs(
            title="Cumulative Returns",
            x="Date",
            y=y,
            color="Legend"
        )
        + theme(legend_position="top")
        + ggsize(1200, 500)
    )

    return plot