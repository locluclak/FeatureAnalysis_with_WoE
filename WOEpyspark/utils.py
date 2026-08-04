"""
Utility and helper functions for WoE/IV analysis with PySpark.

This module provides Spark-side utilities such as quantile edge
computation, bin labelling expressions, and stratified train/test
splitting.
"""
from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession

from pyspark.sql import functions as F


def stratified_split(sdf, label_col="LABEL", test_size=0.2, seed=42):
    """Split a PySpark DataFrame into train/test by stratified sampling.

    Uses sampleBy on the label column to maintain the same class
    ratio in both splits, then subtracts to get the train set.

    Args:
        sdf: PySpark DataFrame to split.
        label_col (str): Name of the binary label column (default "LABEL").
        test_size (float): Fraction of data for the test set (default 0.2).
        seed (int): Random seed for reproducibility (default 42).

    Returns:
        tuple: (train_sdf, test_sdf) as PySpark DataFrames.
    """
    fractions = {row[label_col]: test_size
                 for row in sdf.select(label_col).distinct().collect()}
    test_sdf = sdf.sampleBy(label_col, fractions=fractions, seed=seed)
    train_sdf = sdf.subtract(test_sdf)
    return train_sdf, test_sdf


def stratified_sklearnsplit(sdf, label_col="LABEL", test_size=0.2, seed=42):
    """Split a PySpark DataFrame into train/test using sklearn stratified split.

    Converts to pandas, applies sklearn train_test_split with stratification,
    then converts back to PySpark DataFrames. Guarantees exact row counts
    per class in both splits.

    Args:
        sdf: PySpark DataFrame to split.
        label_col (str): Name of the binary label column (default "LABEL").
        test_size (float): Fraction of data for the test set (default 0.2).
        seed (int): Random seed for reproducibility (default 42).

    Returns:
        tuple: (train_sdf, test_sdf) as PySpark DataFrames.
    """
    pdf = sdf.toPandas()
    train_pdf, test_pdf = train_test_split(
        pdf, test_size=test_size, stratify=pdf[label_col], random_state=seed
    )
    spark = SparkSession.builder.getOrCreate()
    train_sdf = spark.createDataFrame(train_pdf)
    test_sdf = spark.createDataFrame(test_pdf)
    return train_sdf, test_sdf

def _qcut_edges_spark(sdf, col_name, n_bins, mask_expr=None):
    """Compute quantile edges on a PySpark DataFrame using approxQuantile.

    Returns deduplicated edges in ascending order, similar to
    pandas.qcut(duplicates='drop').

    Args:
        sdf: PySpark DataFrame.
        col_name (str): Column name to compute quantiles on.
        n_bins (int): Number of bins (determines number of quantiles).
        mask_expr: Optional PySpark Column expression to filter rows
            before computing quantiles (e.g., for excluding nulls/zeros).

    Returns:
        list: Deduplicated quantile edges including min and max values.
    """
    working = sdf.filter(mask_expr) if mask_expr is not None else sdf
    probs = [i / n_bins for i in range(n_bins + 1)]
    edges = working.approxQuantile(col_name, probs, 0.0001)
    edges_dedup = []
    for e in edges:
        if not edges_dedup or e > edges_dedup[-1]:
            edges_dedup.append(e)
    return edges_dedup


def _bucket_label_expr(col, edges):
    """Create a PySpark Column expression for bin labelling from edges.

    Generates a Column expression that assigns bin labels in the form
    '(left, right]' based on the provided edges. Uses only Spark-native
    operations (no Python UDF) to avoid worker crashes on Windows.

    Args:
        col: PySpark Column expression to bin.
        edges (list): List of numeric edge values in ascending order.
            Must have at least 2 elements (min and max).

    Returns:
        PySpark Column: Expression that maps each value to its bin label.
    """
    left_edges = edges[:-1]
    right_edges = edges[1:]
    labels = [f"({l:g}, {r:g}]" for l, r in zip(left_edges, right_edges)]

    expr = F.when(col <= right_edges[0], F.lit(labels[0]))
    for i in range(1, len(right_edges)):
        expr = expr.when(col <= right_edges[i], F.lit(labels[i]))
    expr = expr.otherwise(F.lit(labels[-1]))
    return expr
