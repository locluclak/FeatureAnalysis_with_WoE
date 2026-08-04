"""
WoE / IV computation functions for PySpark DataFrames.

This module provides functions to compute Weight of Evidence (WoE)
and Information Value (IV) for both discrete and continuous variables
using PySpark.
"""

import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import StringType


def _qcut_edges_spark(sdf, variable_name, n_bins, mask_expr=None):
    """Compute quantile bin edges using PySpark approxQuantile.

    Args:
        sdf: PySpark DataFrame.
        variable_name (str): Column name of the continuous variable.
        n_bins (int): Desired number of bins (will produce n_bins+1 edges).
        mask_expr: Optional PySpark Column expression to filter rows
            before computing quantiles (e.g., exclude nulls/zeros).

    Returns:
        List of float edge values including min and max.
    """
    filtered = sdf
    if mask_expr is not None:
        filtered = filtered.filter(mask_expr)
    edges = filtered.approxQuantile(variable_name,
                                    [i / n_bins for i in range(n_bins + 1)],
                                    0.0)
    if len(edges) < 2:
        raise ValueError(
            f"approxQuantile returned too few edges ({len(edges)}) "
            f"for variable '{variable_name}'. "
            "Check if the column has enough distinct non-null values."
        )
    return edges


def _bucket_label_expr(col, edges):
    """Build a PySpark Column expression that assigns a bin label string.

    Bins are formatted as "(lower, upper]" intervals, with special labels
    for values equal to the minimum edge.

    Args:
        col: PySpark Column expression.
        edges (list): Sorted list of bin edges from approxQuantile.

    Returns:
        PySpark Column expression producing string bin labels.
    """
    expr = F.when(col <= edges[0], F.lit(f"({edges[0]},{edges[0]}]"))
    for i in range(1, len(edges) - 1):
        expr = expr.when(
            col <= edges[i],
            F.lit(f"({edges[i-1]},{edges[i]}]"),
        )
    expr = expr.otherwise(F.lit(f"({edges[-2]},{edges[-1]}]"))
    return expr


def _sort_bins(df):
    """Sort a WoE DataFrame by bin order: MISSING, ZERO, then intervals by lower bound."""
    col = df.columns[0]

    def _sort_key(val):
        s = str(val)
        if s == "MISSING":
            return (0, 0.0)
        if s == "ZERO":
            return (1, 0.0)
        if s.startswith(("(", "[")):
            try:
                left = s.split(",")[0].strip("([")
                return (2, float(left))
            except (ValueError, IndexError):
                pass
        try:
            return (2, float(s))
        except (ValueError, TypeError):
            return (3, 0.0)

    df = df.copy()
    df["_sort_key"] = df[col].apply(_sort_key)
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
    return df


def _attach_woe_iv_spark(agg_sdf, bin_col, smoothing):
    """Attach WoE and IV columns to an aggregated Spark DataFrame.

    Given an aggregated Spark DataFrame with columns (bin_col, n_obs,
    n_card, n_noncard), this function adds: prop_n_obs, prop_n_card,
    prop_n_noncard, WoE, IV_bin, IV using a global window partition.

    Args:
        agg_sdf: Aggregated Spark DataFrame with at least
            (bin_col, n_obs, n_card, n_noncard) columns.
        bin_col (str): Name of the bin column.
        smoothing (float): Smoothing factor to avoid division by zero.

    Returns:
        PySpark DataFrame with additional WoE/IV columns.
    """
    w = Window.partitionBy()

    total_obs = F.sum("n_obs").over(w)
    total_card_s = F.sum(F.col("n_card") + F.lit(smoothing)).over(w)
    total_noncard_s = F.sum(F.col("n_noncard") + F.lit(smoothing)).over(w)

    out = (
        agg_sdf
        .withColumn("prop_n_obs", F.col("n_obs") / total_obs)
        .withColumn("prop_n_card",
                    (F.col("n_card") + F.lit(smoothing)) / total_card_s)
        .withColumn("prop_n_noncard",
                    (F.col("n_noncard") + F.lit(smoothing)) / total_noncard_s)
        .withColumn("WoE",
                    F.log(F.col("prop_n_card") / F.col("prop_n_noncard")))
        .withColumn("IV_bin",
                    (F.col("prop_n_card") - F.col("prop_n_noncard")) * F.col("WoE"))
    )
    out = out.withColumn("IV", F.sum("IV_bin").over(w))
    return out


def woe_discrete(sdf, variable_name, label_col="LABEL", smoothing=0.005):
    """Compute WoE/IV for a discrete variable on PySpark DataFrame.

    Groups by the variable, computes card/noncard counts, applies
    smoothing, and returns a sorted pandas DataFrame with WoE/IV results.

    Args:
        sdf: PySpark DataFrame containing the data.
        variable_name (str): Name of the discrete variable column.
        label_col (str): Name of the binary label column (default "LABEL").
        smoothing (float): Smoothing factor to avoid WoE = +/-inf
            when a bin has 0 card or 0 noncard (default 0.005).

    Returns:
        pandas.DataFrame with columns: variable_name, n_obs, prop_card,
        prop_n_obs, n_card, n_noncard, n_card_smooth, n_noncard_smooth,
        prop_n_card, prop_n_noncard, WoE, diff_prop_card, diff_WoE,
        IV_detail, IV.
    """
    agg = (
        sdf.groupBy(F.col(variable_name).alias(variable_name))
        .agg(
            F.count(label_col).alias("n_obs"),
            F.avg(label_col).alias("prop_card"),
        )
        .withColumn("n_card", F.col("prop_card") * F.col("n_obs"))
        .withColumn("n_noncard", F.col("n_obs") - F.col("n_card"))
        .withColumn("n_card_smooth", F.col("n_card") + F.lit(smoothing))
        .withColumn("n_noncard_smooth", F.col("n_noncard") + F.lit(smoothing))
    )

    out = _attach_woe_iv_spark(agg, variable_name, smoothing)
    out = out.withColumnRenamed("IV_bin", "IV_detail")

    pdf = out.toPandas()
    pdf = pdf.sort_values("WoE").reset_index(drop=True)
    pdf["diff_prop_card"] = pdf["prop_card"].diff().abs()
    pdf["diff_WoE"] = pdf["WoE"].diff().abs()

    cols = [variable_name, "n_obs", "prop_card", "prop_n_obs",
            "n_card", "n_noncard", "n_card_smooth", "n_noncard_smooth",
            "prop_n_card", "prop_n_noncard", "WoE",
            "diff_prop_card", "diff_WoE", "IV_detail", "IV"]
    return pdf[[c for c in cols if c in pdf.columns]]


def woe_continuous(sdf, variable_name, label_col="LABEL",
                   n_bins=500, smoothing=0.5):
    """Compute WoE/IV for a continuous variable with auto-binning.

    If the number of unique values (excluding MISSING/ZERO) is <= n_bins,
    groups by each original value. Otherwise, uses quantile binning via
    approxQuantile. MISSING and ZERO are always separated into their
    own bins.

    Args:
        sdf: PySpark DataFrame containing the data.
        variable_name (str): Name of the continuous variable column.
        label_col (str): Name of the binary label column (default "LABEL").
        n_bins (int): Maximum number of bins for quantile binning
            (default 500).
        smoothing (float): Smoothing factor for bins with 0 card/noncard
            (default 0.5).

    Returns:
        pandas.DataFrame with WoE/IV results sorted by bin order.
    """
    n_unique_num = (
        sdf.filter(F.col(variable_name).isNotNull() & (F.col(variable_name) != 0))
        .select(variable_name)
        .distinct()
        .count()
    )
    use_binning = n_unique_num > n_bins

    if use_binning:
        edges = _qcut_edges_spark(
            sdf, variable_name, n_bins,
            mask_expr=F.col(variable_name).isNotNull() & (F.col(variable_name) != 0),
        )
        if len(edges) < n_bins * 0.5:
            print(
                f"[woe_continuous] '{variable_name}': approxQuantile created "
                f"only {len(edges) - 1}/{n_bins} bins due to many duplicate values."
            )
        bucket_expr = _bucket_label_expr(F.col(variable_name), edges)
        labelled = sdf.withColumn(
            "_bin_",
            F.when(F.col(variable_name).isNull(), F.lit("MISSING"))
            .when(F.col(variable_name) == 0, F.lit("ZERO"))
            .otherwise(bucket_expr),
        )
    else:
        labelled = sdf.withColumn(
            "_bin_",
            F.when(F.col(variable_name).isNull(), F.lit("MISSING"))
            .when(F.col(variable_name) == 0, F.lit("ZERO"))
            .otherwise(F.col(variable_name).cast(StringType())),
        )

    agg = (
        labelled.groupBy("_bin_")
        .agg(
            F.count(label_col).alias("n_obs"),
            F.avg(label_col).alias("prop_card"),
        )
        .withColumn("n_card", F.col("prop_card") * F.col("n_obs"))
        .withColumn("n_noncard", F.col("n_obs") - F.col("n_card"))
    )

    out = _attach_woe_iv_spark(agg, "_bin_", smoothing)
    out = (
        out.withColumnRenamed("_bin_", variable_name)
        .withColumnRenamed("n_noncard", "n_non_card")
        .withColumnRenamed("prop_n_noncard", "prop_n_non_card")
    )

    pdf = out.toPandas()

    def _sort_key(label):
        if label == "MISSING":
            return (-2, 0.0)
        if label == "ZERO":
            return (-1, 0.0)
        if isinstance(label, str) and label.startswith(("(", "[")):
            try:
                left = label.split(",")[0].strip("([")
                return (0, float(left))
            except (ValueError, IndexError):
                return (1, 0.0)
        try:
            return (0, float(label))
        except (ValueError, TypeError):
            return (1, 0.0)

    pdf["_sort"] = pdf[variable_name].apply(_sort_key)
    pdf = pdf.sort_values("_sort").reset_index(drop=True)
    pdf = pdf.drop(columns=["_sort"])

    pdf["diff_prop_card"] = pdf["prop_card"].diff().abs()
    pdf["diff_WoE"] = pdf["WoE"].diff().abs()

    return pdf


def woe_binned_continuous(sdf, variable_name, label_col="LABEL",
                          n_bins=10, smoothing=0.5, keep_zero=True):
    """Compute WoE/IV for a continuous variable with quantile binning.

    Splits the numeric part of the variable into n_bins quantile bins
    using approxQuantile. MISSING is always separated. If keep_zero=True,
    ZERO is also separated.

    Args:
        sdf: PySpark DataFrame containing the data.
        variable_name (str): Name of the continuous variable column.
        label_col (str): Name of the binary label column (default "LABEL").
        n_bins (int): Number of quantile bins for the numeric part
            (default 10).
        smoothing (float): Smoothing factor for bins with 0 card/noncard
            (default 0.5).
        keep_zero (bool): If True, separate ZERO as its own bin
            (default True). Set to False for ratio/percentage variables.

    Returns:
        pandas.DataFrame with WoE/IV results sorted by bin order.
    """
    mask_num = F.col(variable_name).isNotNull()
    if keep_zero:
        mask_num = mask_num & (F.col(variable_name) != 0)

    edges = _qcut_edges_spark(sdf, variable_name, n_bins, mask_expr=mask_num)
    bucket_expr = _bucket_label_expr(F.col(variable_name), edges)

    if keep_zero:
        labelled = sdf.withColumn(
            "_bin_",
            F.when(F.col(variable_name).isNull(), F.lit("MISSING"))
            .when(F.col(variable_name) == 0, F.lit("ZERO"))
            .otherwise(bucket_expr),
        )
    else:
        labelled = sdf.withColumn(
            "_bin_",
            F.when(F.col(variable_name).isNull(), F.lit("MISSING"))
            .otherwise(bucket_expr),
        )

    agg = (
        labelled.groupBy("_bin_")
        .agg(
            F.count(label_col).alias("n_obs"),
            F.avg(label_col).alias("prop_card"),
        )
        .withColumn("n_card", F.col("prop_card") * F.col("n_obs"))
        .withColumn("n_noncard", F.col("n_obs") - F.col("n_card"))
        .withColumn("n_card_s", F.col("n_card") + F.lit(smoothing))
        .withColumn("n_noncard_s", F.col("n_noncard") + F.lit(smoothing))
    )

    out = _attach_woe_iv_spark(agg, "_bin_", smoothing)
    out = (
        out.withColumnRenamed("_bin_", variable_name)
        .withColumnRenamed("n_noncard", "n_non_card")
        .withColumnRenamed("n_noncard_s", "n_non_card_s")
        .withColumnRenamed("prop_n_noncard", "prop_n_non_card")
    )

    pdf = out.toPandas()

    def sort_key(label):
        if label == "MISSING":
            return (-2, 0)
        if keep_zero and label == "ZERO":
            return (-1, 0)
        try:
            left = label.split(",")[0].strip("([")
            return (0, float(left))
        except Exception:
            return (1, 0)

    pdf["_sort"] = pdf[variable_name].apply(sort_key)
    pdf = pdf.sort_values("_sort").reset_index(drop=True)
    pdf = pdf.drop(columns=["_sort"])

    pdf["diff_prop_card"] = pdf["prop_card"].diff().abs()
    pdf["diff_WoE"] = pdf["WoE"].diff().abs()
    return pdf


def recompute_woe_iv(df_bins, smoothing=0.005):
    """Recompute WoE/IV on a pandas DataFrame after subsetting bins.

    When bins are removed from the original WoE table, the totals change.
    This function recalculates all proportions, WoE and IV from scratch.

    Args:
        df_bins: pandas DataFrame with columns n_obs, n_card, and either
            n_noncard or n_non_card.
        smoothing (float): Smoothing factor (default 0.005).

    Returns:
        pandas.DataFrame with recomputed WoE/IV columns.
    """
    df = df_bins.copy().reset_index(drop=True)

    if "n_noncard" in df.columns:
        col_nc = "n_noncard"
    elif "n_non_card" in df.columns:
        col_nc = "n_non_card"
    else:
        raise ValueError("Column n_noncard / n_non_card not found")

    n_card_s = df["n_card"] + smoothing
    n_nc_s = df[col_nc] + smoothing

    total_card_s = n_card_s.sum()
    total_nc_s = n_nc_s.sum()

    df["prop_n_card"] = n_card_s / total_card_s
    if col_nc == "n_noncard":
        df["prop_n_noncard"] = n_nc_s / total_nc_s
        prop_nc = df["prop_n_noncard"]
    else:
        df["prop_n_non_card"] = n_nc_s / total_nc_s
        prop_nc = df["prop_n_non_card"]

    df["prop_n_obs"] = df["n_obs"] / df["n_obs"].sum()
    df["WoE"] = np.log(df["prop_n_card"] / prop_nc)

    iv_col = "IV_detail" if "IV_detail" in df.columns else "IV_bin"
    df[iv_col] = (df["prop_n_card"] - prop_nc) * df["WoE"]
    df["IV"] = df[iv_col].sum()

    if "diff_prop_card" in df.columns:
        df["diff_prop_card"] = df["prop_card"].diff().abs()
    if "diff_WoE" in df.columns:
        df["diff_WoE"] = df["WoE"].diff().abs()

    return df


def build_three_cases(df_bins, missing_label="MISSING", smoothing=0.005):
    """Build three IV comparison cases from a WoE table.

    Generates three views to compare IV impact of MISSING handling:
      - A (FULL): keeps MISSING + ZERO + numeric bins as-is.
      - B (MISSING_VS_REST): collapses all non-missing into one group.
      - C (NO_MISSING): drops MISSING, recomputes on the rest.

    Args:
        df_bins: pandas DataFrame with WoE/IV results (must have
            n_obs, n_card, n_noncard/n_non_card columns).
        missing_label (str): Label for the MISSING bin (default "MISSING").
        smoothing (float): Smoothing factor (default 0.005).

    Returns:
        dict with keys "full", "missing_vs_rest", "no_missing",
        each containing a pandas DataFrame.
    """
    bin_col = df_bins.columns[0]

    if "n_noncard" in df_bins.columns:
        col_nc = "n_noncard"
    elif "n_non_card" in df_bins.columns:
        col_nc = "n_non_card"
    else:
        raise ValueError("Column n_noncard / n_non_card not found")

    full = df_bins.copy().reset_index(drop=True)

    miss_rows = df_bins[df_bins[bin_col] == missing_label]
    rest_rows = df_bins[df_bins[bin_col] != missing_label]

    if len(miss_rows) == 0:
        print(f"[build_three_cases] Warning: no bin '{missing_label}' found.")
        missing_vs_rest = pd.DataFrame()
    else:
        miss_agg = {
            bin_col: missing_label,
            "n_obs": miss_rows["n_obs"].sum(),
            "n_card": miss_rows["n_card"].sum(),
            col_nc: miss_rows[col_nc].sum(),
        }
        rest_agg = {
            bin_col: "NON_MISSING",
            "n_obs": rest_rows["n_obs"].sum(),
            "n_card": rest_rows["n_card"].sum(),
            col_nc: rest_rows[col_nc].sum(),
        }
        miss_agg["prop_card"] = (
            miss_agg["n_card"] / miss_agg["n_obs"] if miss_agg["n_obs"] > 0 else 0
        )
        rest_agg["prop_card"] = (
            rest_agg["n_card"] / rest_agg["n_obs"] if rest_agg["n_obs"] > 0 else 0
        )

        missing_vs_rest = pd.DataFrame([miss_agg, rest_agg])
        missing_vs_rest["IV_detail"] = 0.0
        missing_vs_rest["diff_prop_card"] = np.nan
        missing_vs_rest["diff_WoE"] = np.nan

        missing_vs_rest = recompute_woe_iv(missing_vs_rest, smoothing=smoothing)

        if col_nc == "n_noncard":
            missing_vs_rest["n_card_smooth"] = missing_vs_rest["n_card"] + smoothing
            missing_vs_rest["n_noncard_smooth"] = missing_vs_rest["n_noncard"] + smoothing
        else:
            missing_vs_rest["n_card_s"] = missing_vs_rest["n_card"] + smoothing
            missing_vs_rest["n_non_card_s"] = missing_vs_rest["n_non_card"] + smoothing
        missing_vs_rest = missing_vs_rest.reindex(columns=full.columns)

    no_missing = df_bins[df_bins[bin_col] != missing_label].copy()
    no_missing = recompute_woe_iv(no_missing, smoothing=smoothing)

    full = _sort_bins(full)
    if len(missing_vs_rest) > 0:
        missing_vs_rest = _sort_bins(missing_vs_rest)
    no_missing = _sort_bins(no_missing)

    return {"full": full, "missing_vs_rest": missing_vs_rest, "no_missing": no_missing}
