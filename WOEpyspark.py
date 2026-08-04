
# # Feature Engineering S5 — WoE / IV Analysis (PySpark version)
# 
# Phiên bản **PySpark** của `feature_S5.ipynb`, sử dụng PySpark DataFrame
# API cho toàn bộ pipeline (load parquet, split, groupBy/agg, F.log,...).
# Kết quả (bảng WoE/IV và Excel export) giống hệt notebook gốc.
# 
# **Quy ước:**
# - `card` = khách hàng mở thẻ (LABEL = 1)
# - `noncard` = khách hàng không mở thẻ (LABEL = 0)
# - `MISSING` và `ZERO` luôn tách bin riêng khi tính WoE.
# 


# ## 1. Setup
# 


# **Imports & pandas display options**
# 


import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import StringType

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# **Khởi tạo SparkSession** — chạy local mode với 4g driver memory và bật Arrow để `toPandas()` nhanh hơn.
# 


# Khởi tạo SparkSession (local mode, 4g driver memory)
spark = (
    SparkSession.builder
    .appName("feature_S5_woe_iv")
    .master("local[4]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


# ## 2. Load & Prepare Data
# 
# Đọc parquet bằng Spark, lọc snapshot **S5**, split train/test 80/20
# (stratified theo `LABEL` để giữ tỉ lệ mở thẻ).
# 


# **Load raw dataset**
# 


sdf = spark.read.parquet(r"feature_S5.parquet")
sdf.show(5)


# **Lọc snapshot S5**
# 


# Chỉ lấy dữ liệu snapshot S5
sdf_s5 = sdf.filter(F.col("SNAP_SHOT") == "S5")

# **Stratified split** — dùng sklearn train_test_split để giữ tỉ lệ Label 0/1 chính xác ở train và test.
# 



train_sdf, test_sdf = woe.stratified_split(sdf_s5, "LABEL", test_size=0.2, seed=42)
print(f"train: {train_sdf.count()} rows, test: {test_sdf.count()} rows")


# **Bind working dataframe** — mọi phân tích chạy trên `current_sdf` (train) để tránh leak sang test. `.cache()` để tái sử dụng.
# 


# `current_sdf` là working dataframe (train). Cache để tái sử dụng.
current_sdf = train_sdf.cache()


# ## 3. Helper Functions
# 
# Toàn bộ logic tính WoE/IV và vẽ đồ thị được đóng gói ở đây để các
# phần phân tích bên dưới chỉ cần gọi hàm.
# 


# ### 3.1 Utility tính WoE/IV chung cho Spark DataFrame
# 
# `_attach_woe_iv_spark` nhận agg Spark DF (đã có `n_obs`, `n_card`, `n_noncard`),
# thêm các cột `prop_n_obs`, `prop_n_card`, `prop_n_noncard`, `WoE`, `IV_bin`,
# `IV` bằng `Window().partitionBy()` để tính tổng toàn cục.
# 


def _attach_woe_iv_spark(agg_sdf, bin_col, smoothing):
    # Nhận agg Spark DF có (bin_col, n_obs, n_card, n_noncard), thêm các cột:
    # prop_n_obs, prop_n_card, prop_n_noncard, WoE, IV_bin, IV.
    # Dùng Window().partitionBy() (không partition) để tính tổng toàn cục.
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


# ### 3.2 WoE / IV cho biến **discrete**
# 
# `woe_discrete_card_smoothi_spark` — tính WoE/IV cho biến rời rạc trên Spark,
# có smoothing để tránh `WoE = ±inf` khi 1 bin bị 0 card / 0 noncard.
# Trả về `pandas.DataFrame` để hiển thị / lưu Excel.
# 


def woe_discrete_card_smoothi_spark(sdf, variable_name, label_col="LABEL",
                                    smoothing=0.005):
    # WoE/IV cho biến rời rạc — trả về pandas.DataFrame để hiển thị/lưu Excel.
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


# ### 3.3 Plot WoE
# 
# - `plot_woe` — vẽ WoE line + %obs bar (chạy trên pandas DataFrame).
# - `plot_woe_missing` — bỏ dòng MISSING khỏi trục x để dễ đọc phần numeric.
# 


def plot_woe(df, max_ticks=30, figsize=(20, 7)):
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
    plt.show()


def plot_woe_missing(df, max_ticks=30, figsize=(20, 7)):
    df = df[df.iloc[:, 0].astype(str) != "MISSING"].reset_index(drop=True)
    plot_woe(df, max_ticks=max_ticks, figsize=figsize)


# ### 3.4 Quantile edges + bucket UDF cho Spark
# 
# - `_qcut_edges_spark` — lấy edges qcut trên Spark bằng `approxQuantile`
#   (percentile-based), dedup tăng dần, giống `pandas.qcut(duplicates='drop')`.
# - `_bucket_label_udf` — UDF gán nhãn bin dạng `'(a, b]'` theo edges,
#   mô phỏng `pandas.qcut`.
# 


def _qcut_edges_spark(sdf, col_name, n_bins, mask_expr=None):
    # Lấy edges qcut trên Spark bằng approxQuantile (percentile-based).
    # Trả về list edges (bao gồm min và max) đã dedup tăng dần.
    working = sdf.filter(mask_expr) if mask_expr is not None else sdf
    probs = [i / n_bins for i in range(n_bins + 1)]
    edges = working.approxQuantile(col_name, probs, 0.0001)
    edges_dedup = []
    for e in edges:
        if not edges_dedup or e > edges_dedup[-1]:
            edges_dedup.append(e)
    return edges_dedup


def _bucket_label_expr(col, edges):
    # Trả về Column biểu thức gán nhãn bin dạng '(a, b]' theo edges — Spark thuần,
    # không dùng Python UDF (tránh crash worker trên Windows local mode).
    left_edges = edges[:-1]
    right_edges = edges[1:]
    labels = [f"({l:g}, {r:g}]" for l, r in zip(left_edges, right_edges)]

    expr = F.when(col <= right_edges[0], F.lit(labels[0]))
    for i in range(1, len(right_edges)):
        expr = expr.when(col <= right_edges[i], F.lit(labels[i]))
    expr = expr.otherwise(F.lit(labels[-1]))
    return expr


# ### 3.5 WoE/IV cho biến continuous (auto-binning)
# 
# `woe_ordered_continuous_spark` — tính WoE/IV cho biến liên tục / thứ tự:
# 
# - Nếu số giá trị unique (không kể MISSING/ZERO) ≤ `n_bins`: giữ nguyên, group theo từng giá trị gốc.
# - Nếu > `n_bins`: dùng `approxQuantile` chia thành `n_bins` quantile bins.
# - Tách riêng `MISSING` và `ZERO`.
# - Smoothing áp dụng khi có bin bị 0 card hoặc 0 non-card.
# 


def woe_ordered_continuous_spark(sdf, variable_name, label_col="LABEL",
                                 n_bins=500, smoothing=0.5):
    # Đếm giá trị unique của phần numeric != 0 để quyết định có binning hay không
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
                f"[woe_ordered_continuous] '{variable_name}': approxQuantile chỉ "
                f"tạo được {len(edges) - 1}/{n_bins} bins do quá nhiều giá trị trùng."
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


# ### 3.6 Recompute WoE/IV & Build 3 cases (pandas subset)
# 
# - `recompute_woe_iv_pandas` — sau khi có bảng WoE, subset một tập bin
#   thì tính lại toàn bộ WoE/IV (vì tổng card/noncard thay đổi).
# - `build_three_cases` — sinh 3 view từ bảng WoE gốc:
#   `FULL` / `MISSING_VS_REST` / `NO_MISSING` để so sánh IV.
# 
# Cả hai đều viết bằng pandas/numpy vì subset nhỏ (đã collect về driver), không cần Spark.
# 


def recompute_woe_iv_pandas(df_bins, smoothing=0.005):
    # Sau khi đã có bảng WoE pandas, subset một tập bin thì tính lại toàn bộ WoE/IV.
    df = df_bins.copy().reset_index(drop=True)

    if "n_noncard" in df.columns:
        col_nc = "n_noncard"
    elif "n_non_card" in df.columns:
        col_nc = "n_non_card"
    else:
        raise ValueError("Không tìm thấy cột n_noncard / n_non_card")

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
    bin_col = df_bins.columns[0]

    if "n_noncard" in df_bins.columns:
        col_nc = "n_noncard"
    elif "n_non_card" in df_bins.columns:
        col_nc = "n_non_card"
    else:
        raise ValueError("Không tìm thấy cột n_noncard / n_non_card")

    full = df_bins.copy().reset_index(drop=True)

    miss_rows = df_bins[df_bins[bin_col] == missing_label]
    rest_rows = df_bins[df_bins[bin_col] != missing_label]

    if len(miss_rows) == 0:
        print(f"[build_three_cases] Cảnh báo: không có bin '{missing_label}'.")
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

        missing_vs_rest = recompute_woe_iv_pandas(missing_vs_rest, smoothing=smoothing)

        if col_nc == "n_noncard":
            missing_vs_rest["n_card_smooth"] = missing_vs_rest["n_card"] + smoothing
            missing_vs_rest["n_noncard_smooth"] = missing_vs_rest["n_noncard"] + smoothing
        else:
            missing_vs_rest["n_card_s"] = missing_vs_rest["n_card"] + smoothing
            missing_vs_rest["n_non_card_s"] = missing_vs_rest["n_non_card"] + smoothing
        missing_vs_rest = missing_vs_rest.reindex(columns=full.columns)

    no_missing = df_bins[df_bins[bin_col] != missing_label].copy()
    no_missing = recompute_woe_iv_pandas(no_missing, smoothing=smoothing)

    return {"full": full, "missing_vs_rest": missing_vs_rest, "no_missing": no_missing}


# ### 3.7 WoE / IV cho biến **continuous** với quantile binning
# 
# - `woe_binned_continuous_spark` — chia thành `n_bins` quantile bins,
#   **tách riêng** `MISSING` và `ZERO`.
# - 2 alias theo style notebook gốc:
#   - `woe_binned_continuous` — cho biến số tiền / số giao dịch (`keep_zero=True`).
#   - `woe_binned_continuous_no_zero` — cho biến tỉ lệ / % (`keep_zero=False`).
# 


def woe_binned_continuous_spark(sdf, variable_name, label_col="LABEL",
                                n_bins=10, smoothing=0.5, keep_zero=True):
    # Chia biến liên tục thành n_bins quantile bins, tách MISSING (và ZERO
    # nếu keep_zero=True), tính WoE/IV bằng Spark.
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


# Alias giữ style notebook gốc
def woe_binned_continuous(sdf, variable_name, label_col="LABEL",
                          n_bins=10, smoothing=0.5):
    return woe_binned_continuous_spark(sdf, variable_name, label_col,
                                       n_bins, smoothing, keep_zero=True)


def woe_binned_continuous_no_zero(sdf, variable_name, label_col="LABEL",
                                  n_bins=10, smoothing=0.5):
    return woe_binned_continuous_spark(sdf, variable_name, label_col,
                                       n_bins, smoothing, keep_zero=False)


# ## **Implementation**
# 
# Notebook mẫu chạy WoE/IV cho **4 nhóm biến** (~53 biến). Bản PySpark này
# minh họa quy trình đầy đủ trên **1 biến**: `TIEN_VA_TIEN_TUONG_DUONG_CK`
# (có thể lặp lại cho các biến khác).
# 


# ### **4.1. TIEN_VA_TIEN_TUONG_DUONG_CK**
# 
# Biến báo cáo tài chính: **Tiền và tương đương tiền — cuối kỳ**.
# 
# Đặc điểm cần lưu ý:
# - Có tỉ lệ **MISSING** khá cao (~26% quan sát) → cần tách nhóm riêng.
# - Có một lượng nhỏ giá trị **âm** (16 quan sát, min ≈ -107 tỷ) — về mặt kế toán
#   là bất thường, sẽ được gộp vào bin đầu tiên của phần numeric.
# - Có bin **ZERO** (rất ít) — tách riêng vì "= 0" khác về nghĩa so với "gần 0".
# - Phân bố phần > 0 lệch phải mạnh (đơn vị: VND, dải rộng từ trăm triệu → nghìn tỷ),
#   nên dùng quantile binning hoặc manual binning theo mốc business.
# 


# #### 4.1.1. Khảo sát nhanh — kiểm tra giá trị âm
# 
# Đếm số quan sát âm và xem dải giá trị để quyết định gộp bin.
# 


# Đếm số quan sát âm
n_neg = current_sdf.filter(F.col("TIEN_VA_TIEN_TUONG_DUONG_CK") < 0).count()
print(f"Số quan sát âm: {n_neg}")


neg_stats = (
    current_sdf
    .filter(F.col("TIEN_VA_TIEN_TUONG_DUONG_CK") < 0)
    .agg(
        F.min("TIEN_VA_TIEN_TUONG_DUONG_CK").alias("min"),
        F.max("TIEN_VA_TIEN_TUONG_DUONG_CK").alias("max"),
    )
    .collect()[0]
)
print("Min âm:", neg_stats["min"])
print("Max âm:", neg_stats["max"])


# #### 4.1.2. WoE / IV theo giá trị gốc (auto-binning, `n_bins=100`)
# 
# Gọi `woe_ordered_continuous_spark` để có cái nhìn chi tiết trước khi
# quyết định binning. Bảng chi tiết + đồ thị `plot_woe_missing` (ẩn bin
# MISSING để nhìn phần numeric rõ hơn).
# 


df_auto = woe_ordered_continuous_spark(
    current_sdf, "TIEN_VA_TIEN_TUONG_DUONG_CK", "LABEL", n_bins=100
)
print(df_auto)


plot_woe_missing(df_auto)


# #### 4.1.3. Sơ bộ với **quantile binning** (5 bins)
# 
# Chia phần numeric != 0 thành 5 quantile bins để nhìn hình dạng WoE.
# Bin quantile không nhất thiết đơn điệu — đây chỉ là bước trung gian
# trước khi manual binning.
# 


df_qcut = woe_binned_continuous(
    current_sdf, "TIEN_VA_TIEN_TUONG_DUONG_CK", "LABEL", n_bins=5
)
print(df_qcut)


plot_woe_missing(df_qcut)


# #### 4.1.4. Manual binning theo business
# 
# Trên Spark: dùng `F.when(...).otherwise(...)` để gán bin trực tiếp,
# tránh phải gọi UDF.
# 


# #### Manual Binning cho `TIEN_VA_TIEN_TUONG_DUONG_CK`
# 
# Sau khi xem WoE theo quantile binning, ta chọn các mốc cắt theo business
# (đơn vị VND), tách riêng `MISSING` và `ZERO`:
# 
# | Bin | Khoảng (VND) | Ý nghĩa business |
# |-----|---|---|
# | `MISSING`      | NaN                        | Không có báo cáo tài chính |
# | `ZERO`         | = 0                        | Khai báo 0 tiền cuối kỳ |
# | `(-inf, 300M]` | ≤ 300 triệu (kể cả âm)     | Rất ít / âm → rủi ro thanh khoản cao |
# | `(300M, 850M]` | 300 triệu – 850 triệu      | Ít |
# | `(850M, 2B]`   | 850 triệu – 2 tỷ           | Trung bình |
# | `(2B, 5.5B]`   | 2 tỷ – 5.5 tỷ              | Khá |
# | `(5.5B, +inf)` | > 5.5 tỷ                   | Dồi dào tiền mặt |
# 
# > Mốc cắt được chọn để mỗi bin có ≥ 5% quan sát và WoE đơn điệu / hình chữ U rõ.
# > Có thể tinh chỉnh lại sau khi xem plot cuối cùng.
# 


variable_name = "TIEN_VA_TIEN_TUONG_DUONG_CK"
bin_col = "TIEN_VA_TIEN_TUONG_DUONG_CK_BIN"

edges = [-np.inf, 300_000_000, 850_000_000, 2_000_000_000, 5_500_000_000, np.inf]
labels = [
    "(-inf, 300M]",
    "(300M, 850M]",
    "(850M, 2B]",
    "(2B, 5.5B]",
    "(5.5B, +inf)",
]

col = F.col(variable_name)
manual_col = (
    F.when(col.isNull(), F.lit("MISSING"))
    .when(col == 0, F.lit("ZERO"))
    .when(col < 300_000_000, F.lit(labels[0]))
    .when(col < 850_000_000, F.lit(labels[1]))
    .when(col < 2_000_000_000, F.lit(labels[2]))
    .when(col < 5_500_000_000, F.lit(labels[3]))
    .otherwise(F.lit(labels[4]))
)
current_sdf = current_sdf.withColumn(bin_col, manual_col)

print("Phân bố manual bins TIEN_VA_TIEN_TUONG_DUONG_CK:")
current_sdf.groupBy(bin_col).count().orderBy("count", ascending=False).show()


# #### 4.1.5. So sánh IV theo 3 trường hợp
# 
# Với biến có nhiều MISSING, IV bị ảnh hưởng mạnh bởi việc coi MISSING
# là 1 bin hay không. Ta xem xét 3 trường hợp:
# 
# - **A — FULL:** giữ nguyên `MISSING + ZERO + numeric bins`.
# - **B — MISSING vs REST:** gộp phần non-missing thành 1 nhóm → đo sức
#   mạnh dự báo chỉ của biến *is_missing*.
# - **C — NO MISSING:** loại MISSING, tính lại WoE/IV trên phần còn lại.
# 


df_cashin_manual = woe_discrete_card_smoothi_spark(current_sdf, bin_col)

bin_order = ["MISSING", "ZERO"] + labels
order_map = {b: i for i, b in enumerate(bin_order)}
df_cashin_manual["_sort"] = df_cashin_manual[bin_col].map(order_map)
df_cashin_manual = df_cashin_manual.sort_values("_sort").reset_index(drop=True)
df_cashin_manual = df_cashin_manual.drop(columns=["_sort"])

cases = build_three_cases(df_cashin_manual, missing_label="MISSING", smoothing=0.005)
df_full = cases["full"]
df_missing_vs_rest = cases["missing_vs_rest"]
df_no_missing = cases["no_missing"]

print("=" * 70)
print("CASE A — FULL")
print("=" * 70)
print(df_full)
iv_full = df_full["IV"].iloc[0]
print(f"Total IV (FULL): {iv_full:.6f}\n")

print("=" * 70)
print("CASE B — MISSING vs NON_MISSING")
print("=" * 70)
if len(df_missing_vs_rest) > 0:
    print(df_missing_vs_rest)
    iv_mvr = df_missing_vs_rest["IV"].iloc[0]
    print(f"Total IV (MISSING_VS_REST): {iv_mvr:.6f}\n")
else:
    iv_mvr = np.nan
    print("Không có bin MISSING.\n")

print("=" * 70)
print("CASE C — NO MISSING")
print("=" * 70)
print(df_no_missing)
iv_no_miss = df_no_missing["IV"].iloc[0]
print(f"Total IV (NO_MISSING): {iv_no_miss:.6f}\n")

iv_summary = pd.DataFrame({
    "Case": ["A_FULL", "B_MISSING_VS_REST", "C_NO_MISSING"],
    "Description": [
        "Giữ MISSING + ZERO + numeric bins (mặc định)",
        "MISSING vs NON_MISSING (gộp phần còn lại)",
        "Loại MISSING, giữ ZERO + numeric bins",
    ],
    "Total_IV": [iv_full, iv_mvr, iv_no_miss],
})
print("=" * 70)
print("SO SÁNH IV")
print("=" * 70)
print(iv_summary)

best_case = iv_summary.loc[iv_summary["Total_IV"].idxmax(), "Case"]
print(f"\n>>> Trường hợp có IV cao nhất: {best_case}")


# #### 4.1.6. Đồ thị WoE cho manual bins
# 


plot_woe(df_cashin_manual)


plot_woe_missing(df_cashin_manual)





# #### 4.1.7. Xuất kết quả ra Excel
# 
# Lưu 4 sheet (`A_FULL`, `B_MISSING_VS_REST`, `C_NO_MISSING`,
# `IV_SUMMARY`) vào `V2_Result/<VAR_NAME>_pyspark.xlsx` để review với
# business / risk team.
# 


os.makedirs("V2_Result", exist_ok=True)
output_path = os.path.join("V2_Result", f"{variable_name}_pyspark.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_full.to_excel(writer, sheet_name="A_FULL", index=False)
    if len(df_missing_vs_rest) > 0:
        df_missing_vs_rest.to_excel(writer, sheet_name="B_MISSING_VS_REST", index=False)
    df_no_missing.to_excel(writer, sheet_name="C_NO_MISSING", index=False)
    iv_summary.to_excel(writer, sheet_name="IV_SUMMARY", index=False)

print(f"\nĐã xuất {output_path}")


# **Kết thúc — dừng SparkSession**
# 


spark.stop()





