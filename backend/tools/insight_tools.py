"""
Dataset insight engine.

Answers the questions people actually ask after uploading a dataset:
  * Which model gets the highest accuracy on this data?      -> compare_models()
  * Which columns are unnecessary / safe to drop?            -> find_unnecessary_columns()
  * Which columns matter most?                               -> rank_feature_importance()
  * How are the columns related to each other?               -> analyze_relationships()

Everything here is deterministic (scikit-learn + pandas), so the numbers shown
in the chat are real measurements, not LLM guesses. The LLM is only used on top
of these results to add commentary.
"""
from __future__ import annotations

import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain.tools import tool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MAX_ROWS_FOR_MODELS = 5000      # sample size used for model comparison
MAX_ROWS_FOR_SLOW_MODELS = 2000  # SVM / KNN get a smaller sample
MAX_ROWS_FOR_STATS = 20000       # sample size for importance / relationships
MAX_MATRIX_COLUMNS = 15          # max columns in the association heatmap

TARGET_NAME_HINTS = [
    "target", "label", "class", "outcome", "y", "survived", "exited", "churn",
    "churned", "default", "fraud", "is_fraud", "diagnosis", "species", "result",
    "status", "price", "saleprice", "sales", "revenue", "salary", "income",
    "score", "rating", "quality", "attrition", "approved", "response", "spam",
]
ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|\bid\b|identifier|uuid|guid|^index$|^unnamed|rownum|row_number|serial)", re.I)
ID_SUFFIX_PATTERN = re.compile(r"[a-z](Id|ID)$")  # CustomerId, PassengerID


# ───────────────────────────── helpers ─────────────────────────────

def load_df(file_path: str) -> pd.DataFrame:
    from tools.analysis_tools import load_dataset
    df = load_dataset.invoke({"file_path": file_path})
    # Drop fully empty "Unnamed: n" columns that trailing commas create
    empty_unnamed = [c for c in df.columns if str(c).startswith("Unnamed") and df[c].isna().all()]
    return df.drop(columns=empty_unnamed)


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(n=n, random_state=RANDOM_STATE) if len(df) > n else df


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _looks_like_id(name: str, s: pd.Series) -> bool:
    n = len(s.dropna())
    if n == 0:
        return False
    unique_ratio = s.nunique(dropna=True) / n
    name_says_id = bool(ID_NAME_PATTERN.search(str(name)) or ID_SUFFIX_PATTERN.search(str(name)))
    if name_says_id and unique_ratio > 0.9:
        return True
    if unique_ratio >= 0.999 and n > 20:
        if pd.api.types.is_integer_dtype(s):
            # Sequential integers (1..n) are row numbers
            diffs = np.diff(np.sort(s.dropna().values))
            return bool(len(diffs) and np.all(diffs == diffs[0]))
        if _is_text(s):
            return True
    return False


def _is_text(s: pd.Series) -> bool:
    """True for object / string columns (pandas 2 uses object, pandas 3 uses 'str')."""
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and not _is_numeric(s)


def _is_free_text(s: pd.Series) -> bool:
    """Names, tickets, comments... text with so many distinct values it behaves like an ID."""
    if not _is_text(s):
        return False
    n = len(s.dropna())
    nunique = s.nunique(dropna=True)
    return n > 0 and (nunique / n > 0.5 or nunique > 100)


def detect_target(df: pd.DataFrame, hint: Optional[str] = None) -> Optional[str]:
    """Pick the column the user most likely wants to predict."""
    if hint:
        for c in df.columns:
            if str(c).lower() == hint.lower():
                return c
    lowered = {str(c).lower(): c for c in df.columns}
    for name in TARGET_NAME_HINTS:
        if name in lowered:
            col = lowered[name]
            if not _looks_like_id(col, df[col]) and df[col].nunique() > 1:
                return col
    # Fall back to the last usable column
    for col in reversed(df.columns.tolist()):
        s = df[col]
        if s.nunique() > 1 and not _looks_like_id(col, s) and not _is_free_text(s):
            return col
    return None


def detect_problem_type(y: pd.Series) -> str:
    if not _is_numeric(y):
        return "classification"
    nunique = y.nunique(dropna=True)
    is_integer_like = bool(np.all(np.mod(y.dropna(), 1) == 0))
    if nunique <= 2 or (is_integer_like and nunique <= 15):
        return "classification"
    return "regression"


def find_target_in_message(message: str, columns: List[str]) -> Optional[str]:
    """Find a column name mentioned in the message, preferring ones after 'predict'/'target'."""
    msg = message.lower()
    ordered = sorted(columns, key=lambda c: -len(str(c)))  # longest first

    def mentioned(col: str, text: str) -> bool:
        return re.search(rf"(?<![a-z0-9_]){re.escape(str(col).lower())}(?![a-z0-9_])", text) is not None

    m = re.search(r"(predict(?:ing)?|target(?: column)?(?: is| =|:)?|forecast(?:ing)?|classify(?:ing)?|for)\s+(.+)", msg)
    if m:
        tail = m.group(2)
        for col in ordered:
            if mentioned(col, tail):
                return col
    return None


def _feature_frame(df: pd.DataFrame, target: Optional[str], drop: List[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c != target and c not in drop]
    return df[cols]


def _encode_for_stats(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[bool]]:
    """Ordinal-encode categoricals + median-impute numerics (for MI / tree importance)."""
    X = X.copy()
    discrete = []
    for c in X.columns:
        if _is_numeric(X[c]):
            X[c] = X[c].fillna(X[c].median())
            discrete.append(False)
        else:
            X[c] = X[c].astype(str).fillna("missing")
            discrete.append(True)
    cat_cols = [c for c, d in zip(X.columns, discrete) if d]
    if cat_cols:
        X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
    return X.astype(float), discrete


def _encode_target(y: pd.Series, problem_type: str) -> pd.Series:
    if problem_type == "classification":
        return pd.Series(pd.factorize(y.astype(str))[0], index=y.index)
    return y.astype(float)


# ─────────────────────── unnecessary columns ───────────────────────

def find_unnecessary_columns(df: pd.DataFrame, target: Optional[str] = None,
                             importance: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Return columns that are safe (or worth considering) to drop, with reasons."""
    n = len(df)
    results: Dict[str, Dict[str, Any]] = {}

    def flag(col, reason, severity):
        entry = results.setdefault(col, {"column": col, "reasons": [], "severity": "consider"})
        entry["reasons"].append(reason)
        if severity == "drop":
            entry["severity"] = "drop"

    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        nunique = s.nunique(dropna=True)
        missing_pct = s.isna().mean() * 100
        if nunique <= 1:
            flag(col, "Constant column – has only one value, carries no information", "drop")
            continue
        top_share = s.value_counts(normalize=True, dropna=False).iloc[0] * 100
        if top_share >= 98:
            flag(col, f"Near-constant – {top_share:.1f}% of rows have the same value", "drop")
        if _looks_like_id(col, s):
            flag(col, "Identifier column – unique per row, can't generalise to new data", "drop")
        elif _is_free_text(s) and missing_pct < 60:
            flag(col, f"High-cardinality text ({nunique} unique values) – acts like an ID unless feature-engineered", "drop")
        if missing_pct >= 60:
            flag(col, f"Mostly missing – {missing_pct:.1f}% of values are empty", "drop")
        elif missing_pct >= 30:
            flag(col, f"Many missing values – {missing_pct:.1f}% empty, needs imputation", "consider")

    # Exact duplicate columns
    cols = [c for c in df.columns if c != target]
    seen: Dict[int, str] = {}
    sample = _sample(df, MAX_ROWS_FOR_STATS)
    for col in cols:
        try:
            h = int(pd.util.hash_pandas_object(sample[col], index=False).sum())
        except Exception:
            continue
        if h in seen and sample[col].equals(sample[seen[h]]):
            flag(col, f"Exact duplicate of column '{seen[h]}'", "drop")
        else:
            seen.setdefault(h, col)

    # Highly correlated numeric features (redundant)
    num_cols = [c for c in cols if _is_numeric(df[c]) and c not in results]
    if len(num_cols) >= 2:
        corr = sample[num_cols].corr().abs()
        target_corr = None
        if target and _is_numeric(df[target]):
            target_corr = sample[num_cols].corrwith(sample[target]).abs()
        for i, a in enumerate(num_cols):
            for b in num_cols[i + 1:]:
                r = corr.loc[a, b]
                if pd.notna(r) and r >= 0.95:
                    # Drop the one less related to the target (or the later one)
                    weaker = b
                    if target_corr is not None and target_corr.get(a, 0) < target_corr.get(b, 0):
                        weaker = a
                    keep = a if weaker == b else b
                    if weaker not in results or "redundant" not in " ".join(results[weaker]["reasons"]):
                        flag(weaker, f"Redundant – {r:.2f} correlated with '{keep}'", "consider")

    # Near-zero predictive value for the target
    if importance:
        for item in importance:
            if item["score"] < 0.01 and item["column"] not in results:
                flag(item["column"], "Very weak relationship with the target (near-zero importance)", "consider")

    order = {"drop": 0, "consider": 1}
    return sorted(results.values(), key=lambda r: (order[r["severity"]], r["column"]))


# ─────────────────────── feature importance ───────────────────────

def rank_feature_importance(df: pd.DataFrame, target: str, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
    """Rank columns by how much they help predict the target.

    Combines mutual information (model-agnostic) with Random Forest importance,
    each normalised to sum to 1, then averaged.
    """
    df = _sample(df.dropna(subset=[target]), MAX_ROWS_FOR_STATS)
    problem_type = detect_problem_type(df[target])
    X = _feature_frame(df, target, exclude or [])
    if X.shape[1] == 0:
        return {"target": target, "problem_type": problem_type, "ranking": []}
    X_enc, discrete = _encode_for_stats(X)
    y = _encode_target(df[target], problem_type)

    if problem_type == "classification":
        mi = mutual_info_classif(X_enc, y, discrete_features=discrete, random_state=RANDOM_STATE)
        rf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        mi = mutual_info_regression(X_enc, y, discrete_features=discrete, random_state=RANDOM_STATE)
        rf = RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_enc, y)
    rf_imp = rf.feature_importances_

    def norm(a):
        a = np.clip(np.asarray(a, dtype=float), 0, None)
        return a / a.sum() if a.sum() > 0 else a

    mi_n, rf_n = norm(mi), norm(rf_imp)
    combined = (mi_n + rf_n) / 2

    ranking = []
    for i, col in enumerate(X.columns):
        corr = None
        if _is_numeric(df[col]) and _is_numeric(y):
            c = df[col].corr(y)
            corr = None if pd.isna(c) else round(float(c), 3)
        ranking.append({
            "column": col,
            "score": round(float(combined[i]), 4),
            "mutual_info": round(float(mi[i]), 4),
            "rf_importance": round(float(rf_imp[i]), 4),
            "correlation_with_target": corr,
        })
    ranking.sort(key=lambda r: r["score"], reverse=True)
    return {"target": target, "problem_type": problem_type, "ranking": ranking}


# ───────────────────────── relationships ─────────────────────────

def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.values
    expected = observed.sum(axis=1, keepdims=True) * observed.sum(axis=0, keepdims=True) / observed.sum()
    chi2 = ((observed - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    n = observed.sum()
    r, k = table.shape
    # Bias-corrected Cramér's V
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta: how much of a numeric column's variance is explained by a categorical one."""
    mask = categories.notna() & values.notna()
    cats, vals = categories[mask], values[mask].astype(float)
    if len(vals) < 3 or vals.var() == 0:
        return 0.0
    grand = vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for _, g in vals.groupby(cats))
    ss_total = ((vals - grand) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def _strength_label(v: float) -> str:
    v = abs(v)
    if v >= 0.7:
        return "very strong"
    if v >= 0.5:
        return "strong"
    if v >= 0.3:
        return "moderate"
    if v >= 0.1:
        return "weak"
    return "none"


def analyze_relationships(df: pd.DataFrame, target: Optional[str] = None,
                          exclude: Optional[List[str]] = None, top_n: int = 15) -> Dict[str, Any]:
    """Mixed-type association analysis.

    numeric–numeric: Pearson r (signed)        categorical–categorical: Cramér's V
    categorical–numeric: correlation ratio η
    """
    exclude = set(exclude or [])
    df = _sample(df, MAX_ROWS_FOR_STATS)
    cols = [c for c in df.columns if c not in exclude and df[c].nunique() > 1
            and (_is_numeric(df[c]) or df[c].nunique() <= 50)]
    # Keep the target + the most informative columns for the matrix
    if len(cols) > MAX_MATRIX_COLUMNS:
        keep = [target] if target in cols else []
        keep += [c for c in cols if c != target][: MAX_MATRIX_COLUMNS - len(keep)]
        cols = keep

    def is_cat(c):
        return not _is_numeric(df[c]) or df[c].nunique() <= 10

    pairs, matrix = [], {c: {c: 1.0} for c in cols}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            ca, cb = is_cat(a), is_cat(b)
            try:
                if not ca and not cb:
                    v, method = df[a].corr(df[b]), "pearson"
                elif ca and cb:
                    v, method = _cramers_v(df[a].astype(str), df[b].astype(str)), "cramers_v"
                elif ca:
                    v, method = _correlation_ratio(df[a].astype(str), df[b]), "correlation_ratio"
                else:
                    v, method = _correlation_ratio(df[b].astype(str), df[a]), "correlation_ratio"
            except Exception:
                continue
            v = 0.0 if v is None or pd.isna(v) else float(v)
            matrix[a][b] = matrix[b][a] = round(v, 3)
            pairs.append({
                "column_a": a, "column_b": b, "strength": round(v, 3), "method": method,
                "label": _strength_label(v),
                "direction": ("positive" if v > 0 else "negative") if method == "pearson" else None,
            })

    pairs.sort(key=lambda p: abs(p["strength"]), reverse=True)
    target_pairs = [p for p in pairs if target in (p["column_a"], p["column_b"])] if target else []
    return {
        "top_relationships": [p for p in pairs if abs(p["strength"]) >= 0.1][:top_n],
        "target_relationships": target_pairs[:top_n],
        "matrix": {"columns": cols, "values": [[matrix[a].get(b, 0.0) for b in cols] for a in cols]},
    }


# ───────────────────────── model comparison ─────────────────────────

def _candidate_models(problem_type: str, n_classes: int) -> Dict[str, Tuple[Any, bool]]:
    """name -> (estimator, is_slow)."""
    if problem_type == "classification":
        models = {
            "Logistic Regression": (LogisticRegression(max_iter=2000), False),
            "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), False),
            "Extra Trees": (ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), False),
            "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE), False),
            "Decision Tree": (DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE), False),
            "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=7), True),
            "Naive Bayes": (GaussianNB(), False),
            "SVM (RBF)": (SVC(random_state=RANDOM_STATE), True),
        }
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = (XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0), False)
        except Exception:
            pass
    else:
        models = {
            "Linear Regression": (LinearRegression(), False),
            "Ridge": (Ridge(), False),
            "Lasso": (Lasso(alpha=0.01, max_iter=5000), False),
            "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), False),
            "Extra Trees": (ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), False),
            "Gradient Boosting": (GradientBoostingRegressor(random_state=RANDOM_STATE), False),
            "Decision Tree": (DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE), False),
            "K-Nearest Neighbors": (KNeighborsRegressor(n_neighbors=7), True),
            "SVR (RBF)": (SVR(), True),
        }
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = (XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, verbosity=0), False)
        except Exception:
            pass
    return models


def _preprocessor(X: pd.DataFrame, dense_for_nb: bool = False) -> ColumnTransformer:
    num = [c for c in X.columns if _is_numeric(X[c])]
    cat = [c for c in X.columns if c not in num]
    try:
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20, sparse_output=False)
    except TypeError:  # older scikit-learn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat),
    ])


def compare_models(df: pd.DataFrame, target: str, exclude: Optional[List[str]] = None,
                   problem_type: Optional[str] = None) -> Dict[str, Any]:
    """Cross-validate several algorithms and return a leaderboard."""
    df = df.dropna(subset=[target])
    problem_type = problem_type or detect_problem_type(df[target])
    X_all = _feature_frame(df, target, exclude or [])
    # Coerce object columns to string so the encoder is happy with mixed types
    for c in X_all.columns:
        if not _is_numeric(X_all[c]):
            X_all[c] = X_all[c].astype(str).where(X_all[c].notna(), np.nan)
    if X_all.shape[1] == 0:
        text_cols = [c for c in (exclude or []) if c in df.columns and _is_free_text(df[c])
                     and df[c].astype(str).str.len().mean() > 30]
        hint = (f" `{text_cols[0]}` looks like free text (e.g. messages/reviews) – that needs NLP "
                f"features such as TF-IDF before standard models can use it.") if text_cols else ""
        return {"error": "No usable feature columns left after removing unnecessary columns." + hint}

    y_all = df[target]
    if problem_type == "classification":
        y_all = pd.Series(pd.factorize(y_all.astype(str))[0], index=y_all.index)
        class_counts = y_all.value_counts()
        # Drop classes too rare to cross-validate
        y_all = y_all[y_all.isin(class_counts[class_counts >= 2].index)]
        X_all = X_all.loc[y_all.index]
        if y_all.nunique() < 2:
            return {"error": f"Target '{target}' needs at least two classes with 2+ rows each."}
    if len(y_all) < 10:
        return {"error": f"Only {len(y_all)} usable rows – need at least 10 to compare models reliably."}

    n_classes = int(y_all.nunique()) if problem_type == "classification" else 0
    idx = _sample(X_all, MAX_ROWS_FOR_MODELS).index
    X, y = X_all.loc[idx], y_all.loc[idx]

    if problem_type == "classification":
        min_class = int(y.value_counts().min())
        folds = max(2, min(5, min_class))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"accuracy": "accuracy", "f1": "f1_weighted"}
        if n_classes == 2:
            scoring["roc_auc"] = "roc_auc"
        primary = "accuracy"
    else:
        folds = max(2, min(5, len(y) // 5))
        cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
        primary = "r2"

    leaderboard = []
    for name, (est, is_slow) in _candidate_models(problem_type, n_classes).items():
        Xm, ym = X, y
        if is_slow and len(X) > MAX_ROWS_FOR_SLOW_MODELS:
            sidx = _sample(X, MAX_ROWS_FOR_SLOW_MODELS).index
            Xm, ym = X.loc[sidx], y.loc[sidx]
        pipe = Pipeline([("prep", _preprocessor(Xm)), ("model", est)])
        start = time.time()
        try:
            res = cross_validate(pipe, Xm, ym, cv=cv, scoring=scoring, error_score="raise")
        except Exception as e:
            leaderboard.append({"model": name, "error": str(e)[:200]})
            continue
        row = {"model": name, "train_time_sec": round(time.time() - start, 2), "rows_used": int(len(Xm))}
        for key in scoring:
            vals = res[f"test_{key}"]
            if key in ("rmse", "mae"):
                vals = -vals
            row[key] = round(float(np.mean(vals)), 4)
            row[f"{key}_std"] = round(float(np.std(vals)), 4)
        leaderboard.append(row)

    ok = [r for r in leaderboard if "error" not in r]
    ok.sort(key=lambda r: r[primary], reverse=True)
    failed = [r for r in leaderboard if "error" in r]

    if problem_type == "classification":
        baseline = round(float(y.value_counts(normalize=True).iloc[0]), 4)
        baseline_desc = "always predicting the most common class"
    else:
        baseline, baseline_desc = 0.0, "always predicting the mean (R² = 0)"

    return {
        "target": target,
        "problem_type": problem_type,
        "primary_metric": primary,
        "cv_folds": folds,
        "rows": int(len(y_all)),
        "rows_used": int(len(y)),
        "features_used": X.columns.tolist(),
        "n_classes": n_classes or None,
        "baseline": {"score": baseline, "description": baseline_desc},
        "best_model": ok[0] if ok else None,
        "leaderboard": ok,
        "failed_models": failed,
    }


# ───────────────────────── orchestration ─────────────────────────

_CACHE: Dict[Tuple, Dict[str, Any]] = {}


def dataset_insights(file_path: str, target_column: Optional[str] = None,
                     sections: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute the requested insight sections (cached per file + target)."""
    sections = sections or ["unnecessary", "important", "relationships", "models"]
    mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
    df = load_df(file_path)
    target = detect_target(df, target_column)
    key_base = (file_path, mtime, target)

    out: Dict[str, Any] = {
        "dataset": {"rows": int(len(df)), "columns": int(df.shape[1])},
        "target": target,
        "target_auto_detected": not (target_column and target and target.lower() == target_column.lower()),
        "problem_type": detect_problem_type(df[target]) if target else None,
    }

    def cached(name, fn):
        k = key_base + (name,)
        if k not in _CACHE:
            _CACHE[k] = fn()
        return _CACHE[k]

    # Hard-drop columns are always computed: other sections exclude them.
    pre_unnecessary = cached("unnecessary_basic", lambda: find_unnecessary_columns(df, target))
    hard_drop = [u["column"] for u in pre_unnecessary if u["severity"] == "drop"]
    out["excluded_columns"] = hard_drop

    importance = None
    if target and ("important" in sections or "unnecessary" in sections):
        importance = cached("important", lambda: rank_feature_importance(df, target, hard_drop))
        if "important" in sections:
            out["important"] = importance

    if "unnecessary" in sections:
        out["unnecessary"] = cached(
            "unnecessary_full",
            lambda: find_unnecessary_columns(df, target, importance["ranking"] if importance else None),
        )

    if "relationships" in sections:
        out["relationships"] = cached("relationships", lambda: analyze_relationships(df, target, hard_drop))

    if "models" in sections:
        if target:
            out["models"] = cached("models", lambda: compare_models(df, target, hard_drop))
        else:
            out["models"] = {"error": "Could not determine a target column. Tell me which column to predict, e.g. 'best model to predict Price'."}

    if len(_CACHE) > 200:
        _CACHE.clear()
    return out


# ─────────────────────────── intent routing ───────────────────────────

INTENT_PATTERNS = {
    "models": r"(best|which|top|highest|good|right|suitable|recommend\w*|compare|comparison)\b.*\b(model|algorithm|classifier|regressor)s?|accuracy|accurate|leaderboard|which model|model.*(use|pick|choose)",
    "unnecessary": r"unnecessary|unneccessary|unnecesary|useless|irrelevant|redundant|drop|remove|not (needed|useful|important)|unimportant|least important|can i (delete|remove)",
    "important": r"(most )?important|importance|key (columns?|features?)|significant|influen|drivers?|matter|top (columns?|features?)|feature selection|best (columns?|features?)",
    "relationships": r"relation|correlat|association|depend|connected|link(ed)? between|affect|impact|related",
    "full": r"(full|complete|auto(matic)?|overall|all) (analysis|insights?|report)|analy[sz]e (this|the|my) (data|dataset)|insights?|summar(y|ize|ise)|tell me about (this|the|my) (data|dataset)",
}


def detect_intents(message: str) -> List[str]:
    msg = message.lower()
    found = [name for name, pat in INTENT_PATTERNS.items() if re.search(pat, msg)]
    # "least important" is about unnecessary columns, not the important ones
    if "unnecessary" in found and "important" in found and re.search(r"least important|not important|unimportant", msg):
        found.remove("important")
    if "full" in found:
        return ["unnecessary", "important", "relationships", "models"]
    return found


# ─────────────────────────── formatting ───────────────────────────

def _pct(v: Optional[float]) -> str:
    return "–" if v is None else f"{v * 100:.1f}%"


def format_insights_markdown(ins: Dict[str, Any], dataset_name: str = "your dataset") -> str:
    lines: List[str] = []
    target = ins.get("target")
    if target:
        how = "auto-detected" if ins.get("target_auto_detected") else "chosen by you"
        lines.append(f"**Dataset:** {dataset_name} · {ins['dataset']['rows']:,} rows × {ins['dataset']['columns']} columns  \n"
                     f"**Target column:** `{target}` ({how}) · **Problem type:** {ins.get('problem_type')}")
        if ins.get("target_auto_detected"):
            lines.append("_Want a different target? Ask e.g. “best model to predict `<column>`”._")
    lines.append("")

    if "models" in ins:
        m = ins["models"]
        lines.append("## 🏆 Best model for this dataset")
        if m.get("error"):
            lines.append(f"⚠️ {m['error']}")
        elif m.get("best_model"):
            best = m["best_model"]
            if m["problem_type"] == "classification":
                lines.append(f"**{best['model']}** gets the highest accuracy: **{_pct(best['accuracy'])}** "
                             f"(± {best['accuracy_std'] * 100:.1f}%, {m['cv_folds']}-fold cross-validation). "
                             f"Baseline ({m['baseline']['description']}) is {_pct(m['baseline']['score'])}.")
                lines.append("")
                has_auc = any("roc_auc" in r for r in m["leaderboard"])
                lines.append("| Rank | Model | Accuracy | F1 (weighted)" + (" | ROC-AUC" if has_auc else "") + " | Time (s) |")
                lines.append("|---|---|---|---" + ("|---" if has_auc else "") + "|---|")
                for i, r in enumerate(m["leaderboard"], 1):
                    auc = f" | {r['roc_auc']:.3f}" if has_auc else ""
                    lines.append(f"| {i} | {r['model']} | {_pct(r['accuracy'])} | {r['f1']:.3f}{auc} | {r['train_time_sec']} |")
            else:
                lines.append(f"**{best['model']}** fits best: **R² = {best['r2']:.3f}** "
                             f"(RMSE {best['rmse']:.4g}, MAE {best['mae']:.4g}, {m['cv_folds']}-fold cross-validation).")
                lines.append("")
                lines.append("| Rank | Model | R² | RMSE | MAE | Time (s) |")
                lines.append("|---|---|---|---|---|---|")
                for i, r in enumerate(m["leaderboard"], 1):
                    lines.append(f"| {i} | {r['model']} | {r['r2']:.3f} | {r['rmse']:.4g} | {r['mae']:.4g} | {r['train_time_sec']} |")
            if m["rows_used"] < m["rows"]:
                lines.append(f"\n_Scored on a random sample of {m['rows_used']:,} of {m['rows']:,} rows for speed._")
            if ins.get("excluded_columns"):
                lines.append(f"_Excluded before training: {', '.join(f'`{c}`' for c in ins['excluded_columns'])}._")
        lines.append("")

    if "important" in ins:
        imp = ins["important"]
        lines.append(f"## ⭐ Most important columns (for predicting `{imp['target']}`)")
        if not imp["ranking"]:
            lines.append("No usable feature columns found.")
        else:
            lines.append("| Rank | Column | Importance | Correlation with target |")
            lines.append("|---|---|---|---|")
            for i, r in enumerate(imp["ranking"][:12], 1):
                corr = "–" if r["correlation_with_target"] is None else f"{r['correlation_with_target']:+.2f}"
                lines.append(f"| {i} | `{r['column']}` | {r['score'] * 100:.1f}% | {corr} |")
            top = [f"`{r['column']}`" for r in imp["ranking"][:3]]
            lines.append(f"\nThe top drivers are {', '.join(top)}. _Importance = average of mutual information and Random Forest importance, normalised to 100%._")
        lines.append("")

    if "unnecessary" in ins:
        un = ins["unnecessary"]
        lines.append("## 🗑️ Unnecessary columns")
        if not un:
            lines.append("No obviously unnecessary columns – every column looks usable. ✅")
        else:
            lines.append("| Column | Recommendation | Why |")
            lines.append("|---|---|---|")
            for u in un:
                rec = "**Drop**" if u["severity"] == "drop" else "Consider dropping"
                lines.append(f"| `{u['column']}` | {rec} | {'; '.join(u['reasons'])} |")
        lines.append("")

    if "relationships" in ins:
        rel = ins["relationships"]
        lines.append("## 🔗 Relationships between columns")
        if rel.get("target_relationships") and target:
            lines.append(f"**With the target `{target}`:**")
            for p in rel["target_relationships"][:6]:
                other = p["column_b"] if p["column_a"] == target else p["column_a"]
                d = f", {p['direction']}" if p.get("direction") else ""
                lines.append(f"- `{other}` → {p['label']} ({p['strength']:+.2f}{d})")
            lines.append("")
        pairs = [p for p in rel["top_relationships"] if not target or target not in (p["column_a"], p["column_b"])]
        if pairs:
            lines.append("**Strongest relationships between other columns:**")
            lines.append("| Column A | Column B | Strength | Type |")
            lines.append("|---|---|---|---|")
            method_name = {"pearson": "Pearson r", "cramers_v": "Cramér's V", "correlation_ratio": "Correlation ratio η"}
            for p in pairs[:10]:
                d = f" ({p['direction']})" if p.get("direction") else ""
                lines.append(f"| `{p['column_a']}` | `{p['column_b']}` | {p['strength']:+.2f} – {p['label']}{d} | {method_name[p['method']]} |")
        elif not rel.get("target_relationships"):
            lines.append("No meaningful relationships found between columns.")
        lines.append("")

    return "\n".join(lines).strip()


def build_charts(ins: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chart payloads the frontend renders with Recharts."""
    charts = []
    m = ins.get("models")
    if m and m.get("leaderboard"):
        key = m["primary_metric"]
        charts.append({
            "type": "bar", "title": f"Model comparison ({'accuracy' if key == 'accuracy' else 'R²'})",
            "xKey": "name", "yKey": "value", "format": "percent" if key == "accuracy" else "number",
            "data": [{"name": r["model"], "value": r[key]} for r in m["leaderboard"]],
            "highlight": m["best_model"]["model"],
        })
    imp = ins.get("important")
    if imp and imp.get("ranking"):
        charts.append({
            "type": "bar", "title": f"Column importance for '{imp['target']}'",
            "xKey": "name", "yKey": "value", "format": "percent", "layout": "vertical",
            "data": [{"name": r["column"], "value": r["score"]} for r in imp["ranking"][:12]],
        })
    rel = ins.get("relationships")
    if rel and rel.get("matrix", {}).get("columns") and len(rel["matrix"]["columns"]) >= 2:
        charts.append({"type": "heatmap", "title": "Column relationship heatmap",
                       "columns": rel["matrix"]["columns"], "values": rel["matrix"]["values"]})
    return charts


def summarize_for_llm(ins: Dict[str, Any]) -> Dict[str, Any]:
    """Compact version of the insights for the LLM prompt (keeps token count small)."""
    s: Dict[str, Any] = {k: ins.get(k) for k in ("dataset", "target", "problem_type", "excluded_columns")}
    if "models" in ins:
        m = ins["models"]
        s["models"] = {"error": m["error"]} if m.get("error") else {
            "best": m["best_model"], "baseline": m["baseline"],
            "leaderboard": [{k: r.get(k) for k in ("model", "accuracy", "f1", "roc_auc", "r2", "rmse")} for r in m["leaderboard"]],
        }
    if "important" in ins:
        s["important"] = ins["important"]["ranking"][:10]
    if "unnecessary" in ins:
        s["unnecessary"] = ins["unnecessary"]
    if "relationships" in ins:
        s["relationships"] = ins["relationships"]["top_relationships"][:10]
    return s


# ─────────────────────────── LangChain tools ───────────────────────────

def _tool_payload(file_path: str, target_column: Optional[str], sections: List[str]) -> Dict[str, Any]:
    try:
        return summarize_for_llm(dataset_insights(file_path, target_column, sections))
    except Exception as e:
        return {"error": str(e)}


@tool
def recommend_best_model(file_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Compare many ML algorithms with cross-validation on the dataset and return a leaderboard
    showing which model gets the highest accuracy (classification) or R² (regression).
    target_column is the column to predict; leave empty to auto-detect it."""
    return _tool_payload(file_path, target_column, ["models"])


@tool
def find_unnecessary_columns_tool(file_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Find unnecessary columns that should be dropped: IDs, constants, near-constants, duplicates,
    mostly-missing, free text, redundant highly-correlated columns and columns with no predictive value."""
    return _tool_payload(file_path, target_column, ["unnecessary"])


@tool
def rank_important_columns(file_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Rank the columns by importance for predicting the target column (mutual information +
    Random Forest importance). Leave target_column empty to auto-detect it."""
    return _tool_payload(file_path, target_column, ["important"])


@tool
def analyze_column_relationships(file_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Find relationships between columns (Pearson correlation for numeric pairs, Cramér's V for
    categorical pairs, correlation ratio for mixed pairs), including relationships with the target."""
    return _tool_payload(file_path, target_column, ["relationships"])


INSIGHT_TOOLS = [recommend_best_model, find_unnecessary_columns_tool, rank_important_columns, analyze_column_relationships]
