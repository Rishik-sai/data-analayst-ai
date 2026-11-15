import pytest
import pandas as pd
import numpy as np
from tools.analysis_tools import (
    get_dataset_overview,
    clean_dataset,
    perform_eda,
    preprocess_for_visualization,
    generate_chart_data,
)

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 4.0, 5.0],
        "B": ["cat", "dog", "dog", "cat", "bird"],
        "C": [10.5, np.nan, 15.2, 12.1, 14.8],
        "D": [1, 1, 1, 1, 1], # Constant
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_get_dataset_overview(sample_csv):
    overview = get_dataset_overview.invoke({"file_path": sample_csv})
    assert "shape" in overview
    assert overview["shape"]["rows"] == 5
    assert overview["shape"]["columns"] == 4
    assert len(overview["columns"]) == 4

def test_clean_dataset(sample_csv):
    result = clean_dataset.invoke({"file_path": sample_csv})
    cleaned_df = result["dataframe"]
    report = result["report"]
    
    assert cleaned_df.isnull().sum().sum() == 0
    assert "cleaned_shape" in report

def test_perform_eda(sample_csv):
    eda = perform_eda.invoke({"file_path": sample_csv})
    
    assert "numeric_statistics" in eda
    assert "categorical_statistics" in eda
    assert "correlation_matrix" in eda

def test_preprocess_for_visualization(sample_csv):
    processed, steps = preprocess_for_visualization.invoke({"file_path": sample_csv})
    assert processed.isnull().sum().sum() == 0
    assert len(steps) > 0

def test_generate_chart_data_bar(sample_csv):
    chart_data = generate_chart_data.invoke({
        "file_path": sample_csv,
        "chart_type": "bar",
        "x_column": "B"
    })
    
    assert chart_data["type"] == "bar"
    assert "data" in chart_data
    assert "layout" in chart_data
    assert "preprocessing_steps" in chart_data

def test_generate_chart_data_scatter(sample_csv):
    chart_data = generate_chart_data.invoke({
        "file_path": sample_csv,
        "chart_type": "scatter",
        "x_column": "A",
        "y_column": "C",
        "color_column": "B"
    })
    
    assert chart_data["type"] == "scatter"
    assert len(chart_data["data"]) > 0
