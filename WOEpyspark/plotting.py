"""
Plotting functions for WoE/IV analysis.

This module provides functions to visualize Weight of Evidence (WoE)
charts with observation percentage bars.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _sort_bins(df):
    """Sort a WoE DataFrame by bin order: MISSING, ZERO, then intervals by lower bound."""
    col = df.columns[0]

    def _sort_key(val):
        s = str(val)
        if s == "MISSING":
            return (0, 0.0)
        if s == "ZERO":
            return (1, 0.0)
        # interval labels like "(lower, upper]"
        if s.startswith(("(", "[")):
            try:
                left = s.split(",")[0].strip("([")
                return (2, float(left))
            except (ValueError, IndexError):
                pass
        # try as plain number
        try:
            return (2, float(s))
        except (ValueError, TypeError):
            return (3, 0.0)

    df = df.copy()
    df["_sort_key"] = df[col].apply(_sort_key)
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
    return df


def plot_woe(df, max_ticks=30, figsize=(20, 7)):
    """Plot WoE line chart with observation percentage bars.

    Creates a dual-axis chart: WoE line on the left y-axis and
    percentage of observations bars on the right y-axis.

    Bins are sorted in interval order: MISSING first, ZERO second,
    then numeric/interval bins from smallest to largest.

    Args:
        df: pandas DataFrame with at least columns [first_col, "WoE"]
            and optionally "prop_n_obs". The first column is used as
            x-axis labels.
        max_ticks (int): Maximum number of x-axis tick labels to display
            (default 30). If exceeded, ticks are evenly sampled.
        figsize (tuple): Figure size as (width, height) in inches
            (default (20, 7)).

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    df = _sort_bins(df)
    col_name = df.columns[0]
    x_values = df.iloc[:, 0]
    y_woe = df["WoE"].values
    n = len(df)

    is_numeric = pd.api.types.is_numeric_dtype(x_values)
    x_pos = np.arange(n)

    fig, ax1 = plt.subplots(figsize=figsize)

    if "prop_n_obs" in df.columns:
        ax2 = ax1.twinx()
        ax2.bar(x_pos, df["prop_n_obs"].values, alpha=0.25,
                color="steelblue", label="% Obs")
        ax2.set_ylabel("% Observations", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
    else:
        ax2 = None

    if n <= 80:
        ax1.plot(x_pos, y_woe, marker="o", markersize=4, linestyle="--",
                 color="green", linewidth=1.2, label="WoE", zorder=3)
    else:
        ax1.plot(x_pos, y_woe, linestyle="-", color="green",
                 linewidth=1, label="WoE", zorder=3)

    ax1.axhline(y=0, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("WoE", color="green")
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.set_xlabel(col_name)

    if n <= max_ticks:
        tick_idx = x_pos
    else:
        tick_idx = np.linspace(0, n - 1, max_ticks, dtype=int)

    labels = [str(x_values.iloc[i]) for i in tick_idx]
    if is_numeric:
        try:
            labels = [f"{float(l):.4g}" for l in labels]
        except ValueError:
            pass

    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_title(f"Weight of Evidence by {col_name}  (n_categories={n})", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    plt.tight_layout()
    return fig


def plot_woe_missing(df, max_ticks=30, figsize=(20, 7)):
    """Plot WoE chart with MISSING bin hidden for clearer numeric view.

    Filters out the MISSING bin before plotting so the numeric bins
    are easier to read.

    Args:
        df: pandas DataFrame with WoE results (same format as plot_woe).
        max_ticks (int): Maximum number of x-axis tick labels (default 30).
        figsize (tuple): Figure size (default (20, 7)).

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    df_filtered = df[df.iloc[:, 0].astype(str) != "MISSING"].reset_index(drop=True)
    return plot_woe(df_filtered, max_ticks=max_ticks, figsize=figsize)
