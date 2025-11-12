# train_and_save_model.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split, GridSearchCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------------------- Config -----------------------------
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "final_churn_data.csv"

TARGET = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42

SELECT_K = 10

# ----------------------------- Helpers -----------------------------
def _load_csv() -> pd.DataFrame:
    if CSV_PATH.exists():
        print(f"📄 Loading data from: {CSV_PATH}")
        return pd.read_csv(CSV_PATH)
    raise FileNotFoundError(f"❌ Could not find dataset at {CSV_PATH}")

def _coerce_target(y: pd.Series) -> pd.Series:
    """Accepts 0/1, True/False, or 'Yes'/'No' and returns int 0/1."""
    if y.dtype == object:
        y = (
            y.astype(str)
             .str.strip()
             .str.lower()
             .map({"yes": 1, "no": 0, "true": 1, "false": 0})
        )
    if not np.issubdtype(y.dropna().dtype, np.number):
        y = y.astype(int)
    return y.astype(int)

def _split_feature_types(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    return X.columns.tolist(), num_cols, cat_cols

# ----------------------------- Main -----------------------------
def main():
    df = _load_csv().copy()
    if TARGET not in df.columns:
        raise KeyError(f"Target '{TARGET}' not found. Columns: {list(df.columns)[:12]} ...")

    y_raw = df[TARGET]
    y = _coerce_target(y_raw)
    X = df.drop(columns=[TARGET])

    raw_cols, num_cols, cat_cols = _split_feature_types(df, TARGET)
    print(f"🧾 Raw columns (excluding target): {len(raw_cols)}")
    print(f"🔢 Numeric: {len(num_cols)} | 🔤 Categorical: {len(cat_cols)}")

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    prep = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps = [("prep", prep)]
    if SELECT_K:
        steps.append(("select", SelectKBest(score_func=f_classif, k=SELECT_K)))
    steps.append(("clf", LogisticRegression(max_iter=2000)))
    pipe = Pipeline(steps=steps)

    SEARCH_SPACE = [
        {
            "clf": [LogisticRegression(max_iter=2000, solver="lbfgs")],
            "clf__C": [0.1, 1.0, 5.0],
            "clf__penalty": ["l2"],
        },
        {
            "clf": [RandomForestClassifier(random_state=RANDOM_STATE)],
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 8, 16],
            "clf__min_samples_split": [2, 5],
        },
    ]

    if HAS_XGB:
        SEARCH_SPACE.append({
            "clf": [XGBClassifier(
                random_state=RANDOM_STATE,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                use_label_encoder=False
            )],
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [3, 4, 6],
            "clf__learning_rate": [0.03, 0.1],
            "clf__subsample": [0.7, 1.0],
            "clf__colsample_bytree": [0.5, 0.8],
            "clf__gamma": [0, 1],
            "clf__reg_lambda": [1.0, 3.0],
        })
        print("✅ XGBoost available — included in grid search.")
    else:
        print("ℹ️ xgboost not installed — skipping XGB in grid. Install with: pip install xgboost")

    print("🔍 Running grid search…")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=SEARCH_SPACE,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"✅ Best CV ROC AUC: {grid.best_score_:.4f}")
    print(f"🏆 Best params: {grid.best_params_}")

    y_pred = best_model.predict(X_test)
    y_prob = None
    if hasattr(best_model, "predict_proba"):
        try:
            y_prob = best_model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None

    print("\n📈 Holdout performance:")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    if y_prob is not None:
        try:
            print(f"ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")
        except Exception:
            pass
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    bundle = {
        "model": best_model,     # full sklearn Pipeline
        "columns": raw_cols,     # raw expected input columns
        "input_type": "raw",     # tells app.py this accepts raw fields
        "target": TARGET,
    }

    try:
        prep_step = best_model.named_steps.get("prep")
        feat_names = prep_step.get_feature_names_out()
        bundle["feature_names_out"] = feat_names.tolist()
    except Exception:
        pass

    joblib.dump(bundle, "model.pkl")
    print("\n💾 Saved model.pkl (Pipeline + metadata).")

if __name__ == "__main__":
    main()
