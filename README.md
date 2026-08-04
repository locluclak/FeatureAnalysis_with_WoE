# WOEpyspark

Weight of Evidence (WoE) and Information Value (IV) analysis library built on PySpark.

## Installation

```bash
conda activate sacom
pip install pyspark pandas numpy matplotlib scikit-learn openpyxl
```

Place the `WOEpyspark` folder in your project directory so Python can import it.

## Quick Start

```python
import WOEpyspark as woe
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("woe_analysis") \
    .master("local[4]") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

sdf = spark.read.parquet("feature_S5.parquet")

# Split train/test (sklearn - exact counts)
train_sdf, test_sdf = woe.stratified_sklearnsplit(sdf, label_col="LABEL")

# Compute WoE/IV for a discrete variable
df_woe = woe.woe_discrete(train_sdf, "MY_DISCRETE_VAR", "LABEL")

# Compute WoE/IV for a continuous variable (auto-binning)
df_woe_cont = woe.woe_continuous(train_sdf, "MY_CONTINUOUS_VAR", "LABEL", n_bins=10)

# Compute WoE/IV for a continuous variable (quantile binning)
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

```python
from pyspark.sql import functions as F
import numpy as np

# Define bins once
bins = [
    (300_000_000, "(-inf, 300M]"),
    (850_000_000, "(300M, 850M]"),
    (2_000_000_000, "(850M, 2B]"),
    (5_500_000_000, "(2B, 5.5B]"),
    (np.inf, "(5.5B, +inf)")
]

col = F.col(FEATURE_NAME)

bin_expr = (
    F.when(col.isNull(), "MISSING")
     .when(col == 0, "ZERO")
)

for upper, label in bins:
    if np.isinf(upper):
        bin_expr = bin_expr.otherwise(label)
    else:
        bin_expr = bin_expr.when(col < upper, label)

train_sdf = train_sdf.withColumn(
    "TIEN_VA_TIEN_TUONG_DUONG_CK_BIN",
    bin_expr
)

# Compute WoE/IV on the binned variable (now discrete)
df_manual = woe.woe_discrete(train_sdf, "TIEN_VA_TIEN_TUONG_DUONG_CK_BIN")
```

## Comparing IV Across MISSING Handling Strategies

```python
# Build 3 cases: FULL, MISSING_VS_REST, NO_MISSING
cases = woe.build_three_cases(df_manual, missing_label="MISSING")

# Each case is a pandas DataFrame
print(type(cases["full"]))           # <class 'pandas.core.frame.DataFrame'>
print(cases["full"])                 # Full WoE/IV table
print(cases["missing_vs_rest"])      # MISSING vs NON_MISSING
print(cases["no_missing"])           # Without MISSING bin

# Save all 3 cases to a single CSV (with Case column)
woe.save_iv_csv(cases, "output/iv_cases.csv")

# Save all 3 cases + plots to a single HTML file (images saved separately)
fig_full = woe.plot_woe(cases["full"])
fig_nomiss = woe.plot_woe(cases["no_missing"])

woe.save_iv_html(
    iv_dict=cases,
    plots={"WoE_Full": fig_full, "WoE_NoMissing": fig_nomiss},
    output_path="output/iv_cases.html",
    title="IV Summary"
)
```

Output structure:
```
output/
├── iv_cases.csv
├── iv_cases.html
└── images/
    ├── WoE_Full.png
    └── WoE_NoMissing.png
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
| `stratified_split(sdf, label_col, test_size, seed)` | Stratified train/test split using PySpark `sampleBy` |
| `stratified_sklearnsplit(sdf, label_col, test_size, seed)` | Stratified train/test split using sklearn (exact row counts) |

### I/O

| Function | Description |
|---|---|
| `save_woe_csv(df, filepath)` | Save one WoE table to CSV |
| `save_woe_csvs(tables, output_dir, prefix)` | Save multiple tables to separate CSV files |
| `save_woe_html(tables, plots, output_path, title)` | Save tables + plots to one HTML (images embedded as base64) |
| `save_iv_csv(iv_dict, filepath)` | Save `build_three_cases` results to one CSV (with Case column) |
| `save_iv_html(iv_dict, plots, output_path, title)` | Save `build_three_cases` results + plots to HTML (images saved as separate PNGs) |
