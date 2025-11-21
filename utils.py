from lets_plot import *
import numpy as np
import pandas as pd

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