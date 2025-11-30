from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

with open("models/best_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    churn_prob = model.predict_proba(df)[0][1]

    result = {
        "churn_probability": round(float(churn_prob), 4),
        "will_churn": bool(churn_prob > 0.4),
        "risk_level": "HIGH" if churn_prob > 0.6 else "MEDIUM" if churn_prob > 0.3 else "LOW"
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
