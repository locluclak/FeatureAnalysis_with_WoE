"""
Input/output functions for saving WoE/IV results.

This module provides functions to save WoE/IV tables to CSV and
to generate a self-contained HTML page that displays all tables
and plots on a single page.
"""

import os
import base64
from io import BytesIO

import pandas as pd


def save_woe_csv(df, filepath):
    """Save a WoE/IV pandas DataFrame to a CSV file.

    Creates parent directories if they don't exist.

    Args:
        df: pandas DataFrame containing WoE/IV results.
        filepath (str): Output CSV file path.

    Returns:
        str: The resolved output file path.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath


def save_woe_csvs(tables, output_dir, prefix="woe"):
    """Save multiple WoE/IV tables as separate CSV files.

    Args:
        tables (dict): Dictionary mapping sheet/case names to pandas
            DataFrames. Example: {"A_FULL": df_full, "C_NO_MISSING": df_nomiss}.
        output_dir (str): Directory to save CSV files into.
        prefix (str): Filename prefix (default "woe"). Files are named
            {prefix}_{name}.csv.

    Returns:
        list: List of created file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for name, df in tables.items():
        path = os.path.join(output_dir, f"{prefix}_{name}.csv")
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def _fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string.

    Args:
        fig: matplotlib.figure.Figure object.

    Returns:
        str: Base64-encoded PNG image string.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def save_woe_html(tables, plots, output_path, title="WoE/IV Analysis"):
    """Save all WoE/IV tables and plots to a single self-contained HTML file.

    Generates an HTML page with:
    - A summary table of IV values for all cases
    - Each WoE/IV table rendered as an HTML table
    - Each plot embedded as a base64 PNG image

    Args:
        tables (dict): Dictionary mapping case/table names to pandas
            DataFrames. Example: {"A_FULL": df_full, "IV_SUMMARY": df_iv}.
        plots (dict): Dictionary mapping plot names to matplotlib figure
            objects. Example: {"WoE_FULL": fig1, "WoE_NO_MISSING": fig2}.
        output_path (str): Output HTML file path.
        title (str): Page title (default "WoE/IV Analysis").

    Returns:
        str: The resolved output file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; }",
        "h3 { color: #666; }",
        "table { border-collapse: collapse; margin: 15px 0; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }",
        "th { background-color: #f2f2f2; }",
        "tr:nth-child(even) { background-color: #f9f9f9; }",
        "img { max-width: 100%; margin: 10px 0; }",
        ".section { margin-bottom: 30px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
    ]

    for name, df in tables.items():
        html_parts.append(f"<div class='section'>")
        html_parts.append(f"<h2>{name}</h2>")
        html_parts.append(df.to_html(index=False, float_format="%.6f"))
        html_parts.append("</div>")

    for name, fig in plots.items():
        img_b64 = _fig_to_base64(fig)
        html_parts.append(f"<div class='section'>")
        html_parts.append(f"<h2>{name}</h2>")
        html_parts.append(f"<img src='data:image/png;base64,{img_b64}' alt='{name}'>")
        html_parts.append("</div>")

    html_parts.extend(["</body>", "</html>"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    return output_path


def save_iv_csv(iv_dict, filepath):
    """Save build_three_cases results to a single CSV file.

    All cases are saved into one CSV with a 'Case' column to identify them.
    Overwrites the file if it already exists.

    Args:
        iv_dict (dict): Dictionary mapping case names to pandas
            DataFrames from build_three_cases.
            Example: {"full": df_full, "missing_vs_rest": df_miss, "no_missing": df_nomiss}.
        filepath (str): Output CSV file path.

    Returns:
        str: The resolved output file path.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    frames = []
    for name, df in iv_dict.items():
        tmp = df.copy()
        tmp.insert(0, "Case", name)
        frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(filepath, index=False)
    return filepath


def save_iv_html(iv_dict, plots=None, output_path="output/iv_cases.html",
                 title="IV Summary", bin_info=None):
    """Save build_three_cases results and optional plots to a single HTML file.

    Images are saved as separate PNG files in the same directory as the HTML,
    then referenced via relative paths (not embedded).

    Args:
        iv_dict (dict): Dictionary mapping case names to pandas
            DataFrames from build_three_cases.
        plots (dict, optional): Dictionary mapping plot names to matplotlib
            figure objects. Example: {"WoE_Full": fig1, "WoE_NoMissing": fig2}.
        output_path (str): Output HTML file path (default "output/iv_cases.html").
        title (str): Page title (default "IV Summary").
        bin_info (dict, optional): Dictionary with manual bin configuration.
            Keys: "feature", "edges", "labels", "special_bins".

    Returns:
        str: The resolved output file path.
    """
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    img_dir = os.path.join(out_dir, "images")
    saved_plots = {}
    if plots:
        os.makedirs(img_dir, exist_ok=True)
        for name, fig in plots.items():
            safe_name = name.replace(" ", "_").replace("/", "_")
            img_path = os.path.join(img_dir, f"{safe_name}.png")
            fig.savefig(img_path, dpi=100, bbox_inches="tight")
            saved_plots[name] = os.path.relpath(img_path, out_dir)

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; }",
        "table { border-collapse: collapse; margin: 15px 0; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }",
        "th { background-color: #f2f2f2; }",
        "tr:nth-child(even) { background-color: #f9f9f9; }",
        "img { max-width: 100%; margin: 10px 0; }",
        ".section { margin-bottom: 30px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
    ]

    if bin_info:
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>Bin Configuration</h2>")
        rows = []
        for edge, label in zip(bin_info.get("edges", []), bin_info.get("labels", [])):
            rows.append({"Edge": edge, "Label": label})
        special = bin_info.get("special_bins", [])
        for s in special:
            rows.append({"Edge": s, "Label": s})
        bin_df = pd.DataFrame(rows)
        html_parts.append(bin_df.to_html(index=False))
        html_parts.append("</div>")

    for name, df in iv_dict.items():
        html_parts.append(f"<div class='section'>")
        html_parts.append(f"<h2>{name}</h2>")
        html_parts.append(df.to_html(index=False, float_format="%.6f"))
        html_parts.append("</div>")

    for name, rel_path in saved_plots.items():
        html_parts.append(f"<div class='section'>")
        html_parts.append(f"<h2>{name}</h2>")
        html_parts.append(f"<img src='{rel_path}' alt='{name}'>")
        html_parts.append("</div>")

    html_parts.extend(["</body>", "</html>"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    return output_path
