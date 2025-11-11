from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import plotly.express as px
import plotly
import json

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")


# A tiny example dataset to visualize on the dashboard
feature_importance = pd.DataFrame({
    "feature": ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"],
    "importance": [0.10, 0.05, 0.22, 0.18, 0.08, 0.07, 0.06, 0.09, 0.10, 0.05]
})

@app.route("/")
def index():
    # Plotly bar chart
    fig = px.bar(
        feature_importance,
        x="feature", y="importance",
        title="Feature Importance (example)"
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("index.html", graphJSON=graphJSON)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts form-data OR JSON with keys:
    ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
    Returns a numeric prediction.
    """
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()

        # Ensure correct order & types
        cols = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
        X = pd.DataFrame([[float(data.get(c, 0.0)) for c in cols]], columns=cols)

        y_pred = float(model.predict(X)[0])
        return jsonify({"prediction": y_pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

