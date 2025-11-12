# app.py
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

app = Flask(__name__)

# -------------------- Data loading --------------------
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "final_churn_data.csv"

def load_csv() -> pd.DataFrame:
    if CSV_PATH.exists():
        print(f"📄 Loading data from: {CSV_PATH}")
        return pd.read_csv(CSV_PATH)
    raise FileNotFoundError(f"❌ Could not find dataset at {CSV_PATH}")

df = load_csv()

# -------------------- Model loading (path-robust) --------------------
BASE_DIR   = Path(__file__).resolve().parent              # flask-ml-dashboard/
MODEL_PATH = BASE_DIR / "model.pkl"                       # adjust if it's in a subfolder, e.g. BASE_DIR/"models/model.pkl"

print(f"🧠 Loading model from: {MODEL_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"model.pkl not found at {MODEL_PATH}. "
        "Ensure it’s committed to the repo (in flask-ml-dashboard/) or update MODEL_PATH."
    )

_saved = joblib.load(MODEL_PATH)
if isinstance(_saved, dict) and "model" in _saved:
    model       = _saved["model"]
    raw_columns = _saved.get("columns", [])
    input_type  = _saved.get("input_type", "raw")
    target_name = _saved.get("target", "Churn")
else:
    model       = _saved
    target_name = "Churn" if "Churn" in df.columns else df.columns[-1]
    raw_columns = [c for c in df.columns if c != target_name]
    input_type  = "raw"

# -------------------- Helpers --------------------
def _coerce_churn(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "true": 1, "false": 0})
    return pd.to_numeric(s, errors="coerce")

def with_bins(dfin: pd.DataFrame) -> pd.DataFrame:
    t = dfin.copy()
    if "Tenure Months" in t.columns:
        t["Tenure Months (binned)"] = pd.cut(
            pd.to_numeric(t["Tenure Months"], errors="coerce"),
            bins=[-0.1, 6, 12, 24, 36, 60, 120, np.inf],
            labels=["0-5","6-11","12-23","24-35","36-59","60-119","120+"],
            include_lowest=True
        ).astype(str)
    if "Monthly Charges" in t.columns:
        try:
            t["Monthly Charges (binned)"] = pd.qcut(
                pd.to_numeric(t["Monthly Charges"], errors="coerce"),
                q=5, duplicates="drop"
            ).astype(str) 
        except Exception:
            pass
    return t

df_binned = with_bins(df)

# Candidate dimensions to expose (only keep those present)
CANDIDATE_DIMS = [
    "Contract", "Internet Service", "Payment Method", "Offer",
    "Gender", "Senior Citizen", "Dependents", "Paperless Billing",
    "Online Security", "Online Backup", "Device Protection",
    "Premium Tech Support", "Streaming TV", "Streaming Movies",
    "Streaming Music", "Unlimited Data",
    "Tenure Months (binned)", "Monthly Charges (binned)"
]
DIM_CHOICES = [c for c in CANDIDATE_DIMS if c in df_binned.columns]

def default_figure():
    if {"Churn", "Contract"}.issubset(df.columns):
        tmp = df.copy()
        tmp["Churn"] = _coerce_churn(tmp["Churn"])
        plot_df = (
            tmp.groupby("Contract")["Churn"].mean()
               .reset_index(name="Churn Rate")
               .sort_values("Churn Rate", ascending=False)
        )
        fig = px.bar(plot_df, x="Contract", y="Churn Rate", title="Churn Rate by Contract")
        fig.update_yaxes(range=[0, 1])
        return fig
    if "Churn" in df.columns:
        tmp = df.copy()
        tmp["Churn"] = _coerce_churn(tmp["Churn"])
        rate = float(tmp["Churn"].mean())
        plot_df = pd.DataFrame({"Metric": ["Overall Churn Rate"], "Value": [rate]})
        fig = px.bar(plot_df, x="Metric", y="Value", title="Overall Churn Rate")
        fig.update_yaxes(range=[0, 1])
        return fig
    counts = df.iloc[:, :10].notna().sum().reset_index()
    counts.columns = ["Column", "Non-Null Count"]
    return px.bar(counts, x="Column", y="Non-Null Count", title="Non-Null Counts (first 10)")

# -------------------- Routes --------------------
@app.route("/")
def index():
    fig = default_figure()
    graphJSON = fig.to_json()
    return render_template("index.html", graphJSON=graphJSON, dim_choices=DIM_CHOICES)

@app.route("/chart-data")
def chart_data():
    """
    Query params:
      - dim: grouping column (must be in DIM_CHOICES)
      - metric: 'churn_rate' (default) or 'count'
    Returns a Plotly figure JSON string (safe for NumPy/Interval types).
    """
    dim = request.args.get("dim")
    metric = request.args.get("metric", "churn_rate")

    if dim not in DIM_CHOICES:
        return jsonify({"error": f"Invalid dim. Choose one of: {DIM_CHOICES}"}), 400
    if "Churn" not in df_binned.columns:
        return jsonify({"error": "Churn column not found in data"}), 400

    tmp = df_binned.copy()
    tmp["__churn__"] = _coerce_churn(tmp["Churn"])

    if metric == "count":
        plot_df = (
            tmp.groupby(dim, dropna=False)
              .size().reset_index(name="Count")
              .sort_values("Count", ascending=False)
        )
        plot_df[dim] = plot_df[dim].fillna("Unknown").astype(str)
        fig = px.bar(plot_df, x=dim, y="Count", title=f"Count by {dim}")
    else:
        plot_df = (
            tmp.groupby(dim, dropna=False)["__churn__"]
              .mean().reset_index(name="Churn Rate")
              .sort_values("Churn Rate", ascending=False)
        )
        plot_df[dim] = plot_df[dim].fillna("Unknown").astype(str)
        fig = px.bar(plot_df, x=dim, y="Churn Rate", title=f"Churn Rate by {dim}")
        fig.update_yaxes(range=[0, 1])

    return Response(fig.to_json(), mimetype="application/json")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON or form-data from the current UI, translates it to the
    raw training schema (names + string labels), then predicts.
    """
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            payload = request.form.to_dict()

        # 1) UI field name -> raw column name mapping
        name_map = {
            "SeniorCitizen": "Senior Citizen",
            "Partner": "Partner",
            "Dependents": "Dependents",
            "PaperlessBilling": "Paperless Billing",
            "PhoneService": "Phone Service",
            "MultipleLines": "Multiple Lines",
            "OnlineSecurity": "Online Security",
            "OnlineBackup": "Online Backup",
            "DeviceProtection": "Device Protection",
            "TechSupport": "Premium Tech Support", 
            "StreamingTV": "Streaming TV",
            "StreamingMovies": "Streaming Movies",
            "gender": "Gender",
            "InternetService": "Internet Service",
            "Contract": "Contract",
            "PaymentMethod": "Payment Method",
            "MonthlyCharges": "Monthly Charges",
            "Tenure": "Tenure Months", 
        }

        # 2) Value decoders: UI codes -> training labels
        contract_map = {
            "0": "Month-to-month", "1": "One year", "2": "Two year",
            0: "Month-to-month", 1: "One year", 2: "Two year",
        }
        internet_map = {
            "0": "No", "1": "DSL", "2": "Fiber optic",
            0: "No", 1: "DSL", 2: "Fiber optic",
        }
        payment_map = {
            "0": "Bank transfer (automatic)",
            "1": "Credit card (automatic)",
            "2": "Electronic check",
            "3": "Mailed check",
            0: "Bank transfer (automatic)",
            1: "Credit card (automatic)",
            2: "Electronic check",
            3: "Mailed check",
        }
        gender_map = {"0": "Male", "1": "Female", 0: "Male", 1: "Female"}

        checkbox_keys = {
            "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling",
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        }

        # 3) Build translated dict in terms of raw column names
        translated = {}
        for ui_key, raw_key in name_map.items():
            if ui_key in checkbox_keys:
                translated[raw_key] = "Yes" if payload.get(ui_key) else "No"
            elif ui_key == "Contract":
                translated[raw_key] = contract_map.get(payload.get(ui_key), None)
            elif ui_key == "InternetService":
                translated[raw_key] = internet_map.get(payload.get(ui_key), None)
            elif ui_key == "PaymentMethod":
                translated[raw_key] = payment_map.get(payload.get(ui_key), None)
            elif ui_key == "gender":
                translated[raw_key] = gender_map.get(payload.get(ui_key), None)
            elif ui_key in ("MonthlyCharges", "Tenure"):
                v = payload.get(ui_key)
                translated[raw_key] = float(v) if v not in (None, "", "None") else None
            else:
                translated[raw_key] = payload.get(ui_key, None)

        # 4) Ensure ALL expected raw columns are present
        row = []
        for col in raw_columns:
            val = translated.get(col, None)
            row.append(val)
        X = pd.DataFrame([row], columns=raw_columns)
                
        #print("RAW payload:", payload)
        #print("DF for prediction:\n", X.head().to_dict(orient="records"))

        # 5) Predict
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0][1])
            return jsonify({"prediction_type": "probability", "prediction": prob})
        else:
            pred = model.predict(X)[0]
            try:
                pred = float(pred)
            except Exception:
                pred = str(pred)
            return jsonify({"prediction_type": "class", "prediction": pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/schema", methods=["GET"])
def schema():
    return jsonify({
        "input_type": input_type,
        "expected_columns": raw_columns,
        "target": target_name
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
