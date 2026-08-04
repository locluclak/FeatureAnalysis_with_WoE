"""
WOEpyspark - Weight of Evidence / Information Value analysis with PySpark.

A library for computing WoE and IV for both discrete and continuous
variables using PySpark, with support for manual and auto binning,
WoE plotting, and exporting results to CSV/HTML.

Usage:
    import WOEpyspark as woe

    # Compute WoE/IV for a discrete variable
    df = woe.woe_discrete(sdf, "MY_VAR", "LABEL")

    # Plot WoE chart
    fig = woe.plot_woe(df)

    # Save results
    woe.save_woe_csv(df, "output/woe_result.csv")

    # Save build_three_cases results
    cases = woe.build_three_cases(df_manual, missing_label="MISSING")
    woe.save_iv_csv(cases, "output/iv_cases.csv")
    woe.save_iv_html(cases, plots={"WoE": fig}, output_path="output/iv_cases.html")
"""

from WOEpyspark.woe_iv import (
    woe_discrete,
    woe_continuous,
    woe_binned_continuous,
    recompute_woe_iv,
    build_three_cases,
)

from WOEpyspark.plotting import (
    plot_woe,
    plot_woe_missing,
)

from WOEpyspark.utils import (
    stratified_split,
    stratified_sklearnsplit,
)

from WOEpyspark.io import (
    save_woe_csv,
    save_woe_csvs,
    save_woe_html,
    save_iv_csv,
    save_iv_html,
)

__all__ = [
    # WoE/IV computation
    "woe_discrete",
    "woe_continuous",
    "woe_binned_continuous",
    "recompute_woe_iv",
    "build_three_cases",
    # Plotting
    "plot_woe",
    "plot_woe_missing",
    # Utilities
    "stratified_split",
    "stratified_sklearnsplit",
    # I/O
    "save_woe_csv",
    "save_woe_csvs",
    "save_woe_html",
    "save_iv_csv",
    "save_iv_html",
]
