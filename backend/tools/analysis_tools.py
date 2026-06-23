"""
Dataset analysis tools used by CrewAI agents.
Wraps Pandas operations into callable tool functions.
"""
import pandas as pd
import numpy as np
import json
import os
from typing import Optional, Dict, Any, List


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a dataset from file path. Supports CSV, Excel, JSON."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    elif ext == ".json":
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive overview of the dataset."""
    overview = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": [],
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": round(df[col].isnull().sum() / len(df) * 100, 2),
            "unique_count": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
            col_info["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
            col_info["mean"] = round(float(df[col].mean()), 4) if not pd.isna(df[col].mean()) else None
            col_info["std"] = round(float(df[col].std()), 4) if not pd.isna(df[col].std()) else None
            col_info["median"] = float(df[col].median()) if not pd.isna(df[col].median()) else None
        else:
            top_values = df[col].value_counts().head(5).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        overview["columns"].append(col_info)

    return overview


def clean_dataset(df: pd.DataFrame, strategies: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Clean the dataset: handle missing values, duplicates, outliers.
    Returns cleaned DataFrame and a report of changes.
    """
    report = {"changes": [], "original_shape": list(df.shape)}
    cleaned = df.copy()

    # Remove duplicate rows
    dup_count = cleaned.duplicated().sum()
    if dup_count > 0:
        cleaned = cleaned.drop_duplicates()
        report["changes"].append(f"Removed {dup_count} duplicate rows")

    # Handle missing values per column
    for col in cleaned.columns:
        null_count = cleaned[col].isnull().sum()
        if null_count == 0:
            continue

        null_pct = null_count / len(cleaned) * 100

        if null_pct > 60:
            cleaned = cleaned.drop(columns=[col])
            report["changes"].append(f"Dropped column '{col}' ({null_pct:.1f}% missing)")
        elif pd.api.types.is_numeric_dtype(cleaned[col]):
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
            report["changes"].append(f"Filled '{col}' nulls with median ({median_val:.2f})")
        else:
            mode_val = cleaned[col].mode()
            if len(mode_val) > 0:
                cleaned[col] = cleaned[col].fillna(mode_val[0])
                report["changes"].append(f"Filled '{col}' nulls with mode ('{mode_val[0]}')")

    # Detect outliers using IQR for numeric columns
    outlier_info = {}
    for col in cleaned.select_dtypes(include=[np.number]).columns:
        Q1 = cleaned[col].quantile(0.25)
        Q3 = cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = int(((cleaned[col] < lower) | (cleaned[col] > upper)).sum())
        if outlier_count > 0:
            outlier_info[col] = {
                "count": outlier_count,
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }

    report["cleaned_shape"] = list(cleaned.shape)
    report["outliers"] = outlier_info

    return {"dataframe": cleaned, "report": report}


def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform full exploratory data analysis."""
    eda = {}

    # Basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_cols:
        stats = df[numeric_cols].describe().round(4).to_dict()
        eda["numeric_statistics"] = stats

        # Correlation matrix
        corr = df[numeric_cols].corr().round(4)
        eda["correlation_matrix"] = corr.to_dict()

        # Skewness and Kurtosis
        eda["skewness"] = df[numeric_cols].skew().round(4).to_dict()
        eda["kurtosis"] = df[numeric_cols].kurtosis().round(4).to_dict()

        # High correlations (|r| > 0.7)
        high_corr = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                r = corr.iloc[i, j]
                if abs(r) > 0.7:
                    high_corr.append({
                        "col1": numeric_cols[i],
                        "col2": numeric_cols[j],
                        "correlation": round(float(r), 4),
                    })
        eda["high_correlations"] = high_corr

    if categorical_cols:
        cat_stats = {}
        for col in categorical_cols:
            vc = df[col].value_counts().head(10)
            cat_stats[col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.items()},
            }
        eda["categorical_statistics"] = cat_stats

    # Distribution info for numeric columns
    distributions = {}
    for col in numeric_cols:
        distributions[col] = {
            "histogram_values": df[col].dropna().tolist()[:500],  # Limit for serialization
            "mean": round(float(df[col].mean()), 4) if not pd.isna(df[col].mean()) else None,
            "median": round(float(df[col].median()), 4) if not pd.isna(df[col].median()) else None,
        }
    eda["distributions"] = distributions

    return eda


def generate_chart_data(
    df: pd.DataFrame,
    chart_type: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    color_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate Plotly-compatible chart data."""

    if chart_type == "histogram" and x_column:
        values = df[x_column].dropna().tolist()
        return {
            "type": "histogram",
            "data": [{"x": values, "type": "histogram", "name": x_column}],
            "layout": {"title": f"Distribution of {x_column}", "xaxis": {"title": x_column}, "yaxis": {"title": "Count"}},
        }

    elif chart_type == "scatter" and x_column and y_column:
        data_trace = {
            "x": df[x_column].dropna().tolist()[:1000],
            "y": df[y_column].dropna().tolist()[:1000],
            "mode": "markers",
            "type": "scatter",
            "name": f"{x_column} vs {y_column}",
        }
        if color_column and color_column in df.columns:
            data_trace["marker"] = {"color": df[color_column].dropna().tolist()[:1000]}
        return {
            "type": "scatter",
            "data": [data_trace],
            "layout": {"title": f"{x_column} vs {y_column}", "xaxis": {"title": x_column}, "yaxis": {"title": y_column}},
        }

    elif chart_type == "bar" and x_column:
        if y_column:
            grouped = df.groupby(x_column)[y_column].mean().head(20)
        else:
            grouped = df[x_column].value_counts().head(20)
        return {
            "type": "bar",
            "data": [{"x": [str(x) for x in grouped.index.tolist()], "y": grouped.values.tolist(), "type": "bar"}],
            "layout": {"title": f"Bar Chart - {x_column}", "xaxis": {"title": x_column}},
        }

    elif chart_type == "pie" and x_column:
        vc = df[x_column].value_counts().head(10)
        return {
            "type": "pie",
            "data": [{"labels": [str(x) for x in vc.index.tolist()], "values": vc.values.tolist(), "type": "pie"}],
            "layout": {"title": f"Distribution of {x_column}"},
        }

    elif chart_type == "boxplot" and x_column:
        cols = [x_column]
        if y_column:
            cols.append(y_column)
        traces = []
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                traces.append({"y": df[col].dropna().tolist()[:1000], "type": "box", "name": col})
        return {
            "type": "box",
            "data": traces,
            "layout": {"title": f"Box Plot"},
        }

    elif chart_type == "heatmap":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {"type": "heatmap", "data": [], "layout": {"title": "Not enough numeric columns"}}
        corr = df[numeric_cols].corr().round(4)
        return {
            "type": "heatmap",
            "data": [{
                "z": corr.values.tolist(),
                "x": numeric_cols,
                "y": numeric_cols,
                "type": "heatmap",
                "colorscale": "RdBu",
            }],
            "layout": {"title": "Correlation Heatmap"},
        }

    elif chart_type == "line" and x_column and y_column:
        sorted_df = df.sort_values(x_column)
        return {
            "type": "line",
            "data": [{
                "x": sorted_df[x_column].tolist()[:1000],
                "y": sorted_df[y_column].tolist()[:1000],
                "type": "scatter",
                "mode": "lines",
                "name": y_column,
            }],
            "layout": {"title": f"{y_column} over {x_column}", "xaxis": {"title": x_column}, "yaxis": {"title": y_column}},
        }

    return {"type": chart_type, "data": [], "layout": {"title": "Chart type not supported or missing columns"}}
