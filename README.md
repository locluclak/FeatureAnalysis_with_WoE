# WOEpyspark

Weight of Evidence (WoE) and Information Value (IV) analysis library built on PySpark.

## Installation

Ensure you have a conda environment named `sacom` with PySpark, pandas, numpy, and matplotlib installed.

```bash
conda activate sacom
pip install pyspark pandas numpy matplotlib openpyxl
```

Place the `WOEpyspark` folder in your project directory so Python can import it.

## Quick Start

```python
import WOEpyspark as woe
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("woe_analysis") \
    .master("local[4]") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# Load data
sdf = spark.read.parquet("feature_S5.parquet")

# Split train/test
train_sdf, test_sdf = woe.stratified_split(sdf, label_col="LABEL")

# Compute WoE/IV for a discrete variable
df_woe = woe.woe_discrete(train_sdf, "MY_DISCRETE_VAR", "LABEL")

# Compute WoE/IV for a continuous variable (auto-binning)
df_woe_cont = woe.woe_continuous(train_sdf, "MY_CONTINUOUS_VAR", "LABEL", n_bins=10)

# Compute WoE/IV for a continuous variable (quantile binning with 10 bins)
df_binned = woe.woe_binned_continuous(train_sdf, "MY_VAR", "LABEL", n_bins=10)

# Plot WoE chart
fig = woe.plot_woe(df_woe)

# Plot WoE chart (hiding MISSING for clearer view)
fig2 = woe.plot_woe_missing(df_binned)

# Save results to CSV
woe.save_woe_csv(df_woe, "output/woe_result.csv")

# Save all tables and plots to a single HTML file
woe.save_woe_html(
    tables={"WoE_Result": df_woe, "Binned_Result": df_binned},
    plots={"WoE_Chart": fig, "Binned_Chart": fig2},
    output_path="output/woe_report.html",
    title="My WoE Analysis Report"
)
```

## Manual Binning with User-Defined Edges

Define your own bin edges and labels, then apply them as a discrete variable:

```python
from pyspark.sql import functions as F
import numpy as np

# Define edges and labels
edges = [-np.inf, 300_000_000, 850_000_000, 2_000_000_000, 5_500_000_000, np.inf]
labels = ["(-inf, 300M]", "(300M, 850M]", "(850M, 2B]", "(2B, 5.5B]", "(5.5B, +inf)"]

# Create bin column using F.when
col = F.col("TIEN_VA_TIEN_TUONG_DUONG_CK")
bin_expr = (
    F.when(col.isNull(), F.lit("MISSING"))
     .when(col == 0, F.lit("ZERO"))
     .when(col < 300_000_000, F.lit(labels[0]))
     .when(col < 850_000_000, F.lit(labels[1]))
     .when(col < 2_000_000_000, F.lit(labels[2]))
     .when(col < 5_500_000_000, F.lit(labels[3]))
     .otherwise(F.lit(labels[4]))
)
train_sdf = train_sdf.withColumn("TIEN_VA_TIEN_TUONG_DUONG_CK_BIN", bin_expr)

# Compute WoE/IV on the binned variable (now discrete)
df_manual = woe.woe_discrete(train_sdf, "TIEN_VA_TIEN_TUONG_DUONG_CK_BIN")
```

## Comparing IV Across MISSING Handling Strategies

```python
# Build 3 cases: FULL, MISSING_VS_REST, NO_MISSING
cases = woe.build_three_cases(df_manual, missing_label="MISSING")

print("Case A - Full IV:", cases["full"])
print("Case B - Missing vs Rest IV:", cases["missing_vs_rest"])
print("Case C - No Missing IV:", cases["no_missing"])
```

## Saving Multiple Results

```python
# Save multiple tables as separate CSVs
woe.save_woe_csvs(
    tables={"A_FULL": cases["full"], "C_NO_MISSING": cases["no_missing"]},
    output_dir="output",
    prefix="cashin_manual"
)
# Creates: output/cashin_manual_A_FULL.csv, output/cashin_manual_C_NO_MISSING.csv
```

## API Reference

### WoE/IV Computation

| Function | Description |
|---|---|
| `woe_discrete(sdf, var, label, smoothing)` | WoE/IV for discrete variables |
| `woe_continuous(sdf, var, label, n_bins, smoothing)` | WoE/IV with auto-binning (quantile if > n_bins unique values) |
| `woe_binned_continuous(sdf, var, label, n_bins, smoothing, keep_zero)` | WoE/IV with quantile binning |
| `recompute_woe_iv(df_bins, smoothing)` | Recompute WoE/IV after subsetting bins |
| `build_three_cases(df_bins, missing_label, smoothing)` | Generate FULL / MISSING_VS_REST / NO_MISSING comparison |

### Plotting

| Function | Description |
|---|---|
| `plot_woe(df, max_ticks, figsize)` | Plot WoE line + % observation bars |
| `plot_woe_missing(df, max_ticks, figsize)` | Plot WoE with MISSING hidden |

### Utilities

| Function | Description |
|---|---|
| `stratified_split(sdf, label_col, test_size, seed)` | Stratified train/test split |

### I/O

| Function | Description |
|---|---|
| `save_woe_csv(df, filepath)` | Save one WoE table to CSV |
| `save_woe_csvs(tables, output_dir, prefix)` | Save multiple tables to CSV files |
| `save_woe_html(tables, plots, output_path, title)` | Save all tables + plots to one HTML page |
