"""
Temporary script to test sklearn train_test_split vs PySpark sampleBy.
Run this to compare split results and check for seed sensitivity / bugs.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── Spark setup ──
spark = (
    SparkSession.builder
    .appName("temp_sklearn_split")
    .master("local[4]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ── Load data ──
sdf = spark.read.parquet(r"feature_S5.parquet")
sdf_s5 = sdf.filter(F.col("SNAP_SHOT") == "S5")

print(f"Total S5 rows: {sdf_s5.count()}")
print(f"LABEL distribution:")
sdf_s5.groupBy("LABEL").count().orderBy("LABEL").show()


# ── Method 1: PySpark sampleBy (current) ──
def stratified_split_spark(sdf, label_col="LABEL", test_size=0.2, seed=42):
    fractions = {row[label_col]: test_size
                 for row in sdf.select(label_col).distinct().collect()}
    test_sdf = sdf.sampleBy(label_col, fractions=fractions, seed=seed)
    train_sdf = sdf.subtract(test_sdf)
    return train_sdf, test_sdf


# ── Method 2: sklearn train_test_split ──
def stratified_split_sklearn(sdf, label_col="LABEL", test_size=0.2, seed=42):
    pdf = sdf.toPandas()
    train_pdf, test_pdf = train_test_split(
        pdf, test_size=test_size, stratify=pdf[label_col], random_state=seed
    )
    train_sdf = spark.createDataFrame(train_pdf)
    test_sdf = spark.createDataFrame(test_pdf)
    return train_sdf, test_sdf


# ── Compare splits ──
seeds = [0, 42, 123, 999]

for seed in seeds:
    print(f"\n{'='*60}")
    print(f"SEED = {seed}")
    print(f"{'='*60}")

    # PySpark split
    train_sp, test_sp = stratified_split_spark(sdf_s5, "LABEL", test_size=0.2, seed=seed)
    train_count_sp = train_sp.count()
    test_count_sp = test_sp.count()

    train_dist_sp = train_sp.groupBy("LABEL").count().orderBy("LABEL").collect()
    test_dist_sp = test_sp.groupBy("LABEL").count().orderBy("LABEL").collect()

    # sklearn split
    train_sk, test_sk = stratified_split_sklearn(sdf_s5, "LABEL", test_size=0.2, seed=seed)
    train_count_sk = train_sk.count()
    test_count_sk = test_sk.count()

    train_dist_sk = train_sk.groupBy("LABEL").count().orderBy("LABEL").collect()
    test_dist_sk = test_sk.groupBy("LABEL").count().orderBy("LABEL").collect()

    # Print comparison
    print(f"\n{'Method':<15} {'Train':>8} {'Test':>8} | Train LABEL dist          | Test LABEL dist")
    print("-" * 85)

    for label_val in [0, 1]:
        train_n_sp = next((r["count"] for r in train_dist_sp if r["LABEL"] == label_val), 0)
        test_n_sp = next((r["count"] for r in test_dist_sp if r["LABEL"] == label_val), 0)
        train_n_sk = next((r["count"] for r in train_dist_sk if r["LABEL"] == label_val), 0)
        test_n_sk = next((r["count"] for r in test_dist_sk if r["LABEL"] == label_val), 0)

        if label_val == 0:
            print(f"{'PySpark':<15} {train_count_sp:>8} {test_count_sp:>8} | LABEL=0: {train_n_sp:>6}      | LABEL=0: {test_n_sp:>6}")
            print(f"{'sklearn':<15} {train_count_sk:>8} {test_count_sk:>8} | LABEL=0: {train_n_sk:>6}      | LABEL=0: {test_n_sk:>6}")
        else:
            print(f"{'':<15} {'':>8} {'':>8} | LABEL=1: {train_n_sp:>6}      | LABEL=1: {test_n_sp:>6}")
            print(f"{'':<15} {'':>8} {'':>8} | LABEL=1: {train_n_sk:>6}      | LABEL=1: {test_n_sk:>6}")

    # Check ratio
    ratio_sp = test_count_sp / (train_count_sp + test_count_sp) if (train_count_sp + test_count_sp) > 0 else 0
    ratio_sk = test_count_sk / (train_count_sk + test_count_sk) if (train_count_sk + test_count_sk) > 0 else 0
    print(f"\n  Test ratio  — PySpark: {ratio_sp:.4f}  |  sklearn: {ratio_sk:.4f}")

print("\n\nDone. Compare results above to check if random seed affects split differently.")
