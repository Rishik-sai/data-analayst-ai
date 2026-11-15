import numpy as np
import pandas as pd
import pytest

from tools import insight_tools as it


@pytest.fixture
def churn_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    age = rng.integers(18, 80, n)
    products = rng.integers(1, 5, n)
    noise = rng.normal(size=n)
    churn = ((age > 50) | (products >= 4)).astype(int)
    df = pd.DataFrame({
        "CustomerId": np.arange(1000, 1000 + n),          # identifier
        "Surname": [f"name_{i}" for i in range(n)],       # free text
        "Country": rng.choice(["FR", "DE", "ES"], n),
        "Constant": 1,                                     # constant
        "Age": age,
        "AgeMonths": age * 12,                             # redundant with Age
        "Products": products,
        "Noise": noise,
        "Notes": [None] * (n - 10) + ["x"] * 10,           # mostly missing
        "Exited": churn,
    })
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_detect_target_and_problem_type(churn_csv):
    df = it.load_df(churn_csv)
    assert it.detect_target(df) == "Exited"
    assert it.detect_target(df, "age") == "Age"
    assert it.detect_problem_type(df["Exited"]) == "classification"
    assert it.detect_problem_type(df["Noise"]) == "regression"


def test_unnecessary_columns(churn_csv):
    df = it.load_df(churn_csv)
    flagged = {u["column"]: u for u in it.find_unnecessary_columns(df, "Exited")}
    for col in ["CustomerId", "Surname", "Constant", "Notes"]:
        assert flagged[col]["severity"] == "drop", col
    assert "AgeMonths" in flagged or "Age" in flagged  # redundant pair
    assert "Exited" not in flagged
    assert "Products" not in flagged


def test_feature_importance_finds_real_drivers(churn_csv):
    df = it.load_df(churn_csv)
    res = it.rank_feature_importance(df, "Exited", exclude=["CustomerId", "Surname", "Constant", "Notes"])
    top3 = [r["column"] for r in res["ranking"][:3]]
    assert "Products" in top3
    assert "Age" in top3 or "AgeMonths" in top3
    assert res["ranking"][-1]["column"] in {"Noise", "Country"}


def test_relationships(churn_csv):
    df = it.load_df(churn_csv)
    rel = it.analyze_relationships(df, "Exited", exclude=["CustomerId", "Surname", "Constant", "Notes"])
    top = rel["top_relationships"][0]
    assert {top["column_a"], top["column_b"]} == {"Age", "AgeMonths"}
    assert top["strength"] == pytest.approx(1.0, abs=1e-3)
    assert rel["target_relationships"]
    cols = rel["matrix"]["columns"]
    assert len(rel["matrix"]["values"]) == len(cols)


def test_compare_models_leaderboard(churn_csv):
    df = it.load_df(churn_csv)
    res = it.compare_models(df, "Exited", exclude=["CustomerId", "Surname", "Constant", "Notes"])
    assert res["problem_type"] == "classification"
    board = res["leaderboard"]
    assert len(board) >= 5
    assert board == sorted(board, key=lambda r: r["accuracy"], reverse=True)
    assert res["best_model"]["accuracy"] > res["baseline"]["score"]
    assert res["best_model"]["accuracy"] > 0.9  # target is a deterministic rule


def test_compare_models_regression(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=200), "price": 3 * x + rng.normal(scale=0.1, size=200)})
    res = it.compare_models(df, "price")
    assert res["problem_type"] == "regression"
    assert res["best_model"]["r2"] > 0.95


def test_full_insights_markdown_and_charts(churn_csv):
    ins = it.dataset_insights(churn_csv)
    md = it.format_insights_markdown(ins, "churn.csv")
    for heading in ["Best model", "Most important columns", "Unnecessary columns", "Relationships"]:
        assert heading in md
    charts = it.build_charts(ins)
    assert [c["type"] for c in charts] == ["bar", "bar", "heatmap"]


def test_too_small_dataset(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0]}).to_csv(path, index=False)
    ins = it.dataset_insights(str(path), "b", ["models"])
    assert "error" in ins["models"]


@pytest.mark.parametrize("message,expected", [
    ("Which model gives the highest accuracy?", ["models"]),
    ("what are the unneccessary columns in the dataset", ["unnecessary"]),
    ("most important columns", ["important"]),
    ("what are the relationships between columns", ["relationships"]),
    ("Analyze this dataset", ["unnecessary", "important", "relationships", "models"]),
    ("hello there", []),
])
def test_detect_intents(message, expected):
    assert it.detect_intents(message) == expected


def test_find_target_in_message():
    cols = ["Age", "Price", "SalePrice"]
    assert it.find_target_in_message("best model to predict SalePrice", cols) == "SalePrice"
    assert it.find_target_in_message("best model to predict price", cols) == "Price"
    assert it.find_target_in_message("which model is best", cols) is None
