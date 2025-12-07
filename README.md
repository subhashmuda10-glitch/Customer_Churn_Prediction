# 📡 Telco Customer Churn Prediction (End‑to‑End ML + API + UI)

Production‑style machine learning project to predict **telecom customer churn** and serve predictions through a **Flask REST API** and a simple **web UI**.

This project is designed as a portfolio piece for an **ML/AI Engineer** role: it covers EDA, preprocessing, model selection, evaluation, saving a sklearn pipeline, and deploying it behind an API.

---

## 🧠 Problem Statement

Telecom companies lose significant revenue when customers leave (churn). Retaining an existing customer is usually cheaper than acquiring a new one, so predicting **which customers are likely to churn** is a key business problem.

**Goal:**  
Build an end‑to‑end system that:

- Predicts the probability that a customer will churn in the near future.  
- Identifies key drivers of churn (e.g., contract type, payment method, tenure).  
- Exposes the model via an API/UI so it can be integrated into real workflows.

---

## 🗂️ Project Structure

Telco_Customer_churn/
├── data/ # Cleaned dataset (optional in Git)
├── docs/ # PDFs of notebooks / reports
├── models/
│ └── best_pipeline.pkl # Trained sklearn Pipeline (preprocessing + model)
├── notebooks/
│ ├── 01_EDA_and_Business_Understanding.ipynb
│ ├── 02_Preprocessing_and_Modeling.ipynb
│ └── 03_Final_Evaluation_and_Conclusion.ipynb
├── screenshots/ # UI, API and EDA screenshots
├── src/
│ ├── app.py # Flask app (API + HTML UI)
│ ├── templates/
│ │ └── index.html # Web form for churn prediction
│ └── static/
│   └── style.css # Basic styling
├── flashapi_churn.py
├── .gitignore
├── requirements.txt
└── README.md

text

---

## 🧰 Tech Stack

- **Language:** Python  
- **Data / ML:** pandas, numpy, scikit‑learn, XGBoost, CatBoost, matplotlib, seaborn [web:36]  
- **Serving:** Flask (REST API + HTML template)  
- **Environment:** virtualenv / venv, VS Code  
- **Version control:** Git + GitHub

---

## 📊 Modeling Overview

The workflow follows a standard ML project template:

1. **EDA & Business Understanding** (`01_*.ipynb`)  
   - Cleaned `TotalCharges`, explored distributions, target imbalance and key relationships.  
   - Found that **month‑to‑month contracts**, **electronic check**, **short tenure** and **higher monthly charges** are strongly associated with churn. 

2. **Preprocessing & Model Training** (`02_*.ipynb`)  
   - Built a **ColumnTransformer + Pipeline**:
     - Numeric: `tenure`, `MonthlyCharges`, `TotalCharges` → `StandardScaler`  
     - Categorical: remaining columns → `OneHotEncoder(handle_unknown="ignore")`  
   - Trained and compared multiple models:
     - Logistic Regression (baseline)  
     - Random Forest  
     - XGBoost  
     - CatBoost  
   - Used a unified evaluation function (Accuracy, Precision, Recall, F1, ROC‑AUC) and a results table + bar plots to compare models. 

3. **Hyperparameter Tuning & Model Selection**  
   - Applied `RandomizedSearchCV` on the best baseline (e.g., Random Forest) with 3‑fold CV and `roc_auc` scoring.  
   - Selected the **best performing model** based on ROC‑AUC and recall on churners.  
   - Saved the full **preprocessing + model pipeline** as `models/best_pipeline.pkl` for deployment.

4. **Final Evaluation & Insights** (`03_*.ipynb`)  
   - Evaluated the chosen model on a held‑out set.  
   - Plotted ROC and Precision–Recall curves and experimented with different probability thresholds to balance recall vs precision for churners.
   - Analyzed feature importance to highlight the main churn drivers (contract type, payment method, tenure, monthly charges, security/support add‑ons). 

> Update this section with your actual numbers once you run the notebooks, e.g.:  
> “Best model: Tuned Random Forest, ROC‑AUC: **0.85**, Recall on churners: **0.82**.”

---

## 🌐 Flask API & Web UI

The trained pipeline is loaded in `src/app.py` and exposed via:

### 1. Home Page (Web UI)

- `GET /` → renders `index.html`  
- Simple HTML form where a user can enter customer details (contract, payment method, tenure, charges, etc.).  
- On submit, JavaScript sends a JSON `POST` request to `/predict` and displays:

{
"churn_probability": 0.8234,
"will_churn": true,
"risk_level": "HIGH"
}

text

### 2. REST API

- Endpoint: `POST /predict`  
- Request body: JSON with the same fields as the original dataset (minus `customerID` and `Churn`). Example:

{
"gender": "Male",
"SeniorCitizen": 0,
"Partner": "Yes",
"Dependents": "No",
"tenure": 1,
"PhoneService": "Yes",
"MultipleLines": "No",
"InternetService": "Fiber optic",
"OnlineSecurity": "No",
"OnlineBackup": "No",
"DeviceProtection": "No",
"TechSupport": "No",
"StreamingTV": "No",
"StreamingMovies": "No",
"Contract": "Month-to-month",
"PaperlessBilling": "Yes",
"PaymentMethod": "Electronic check",
"MonthlyCharges": 70.5,
"TotalCharges": 70.5
}

text

- Response:

{
"churn_probability": 0.8234,
"will_churn": true,
"risk_level": "HIGH"
}

text

---

## 🏃 How to Run Locally

1. Clone the repo:

git clone https://github.com/<your-username>/Telco_Customer_churn.git
cd Telco_Customer_churn

2. Create and activate virtual environment (Windows):

python -m venv .venv
..venv\Scripts\activate

Mac/Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

4. Run Flask app:

python src/app.py


5. Open in browser:

- UI: `http://127.0.0.1:5000/`  
- API test (with curl):

curl -X POST http://127.0.0.1:5000/predict
-H "Content-Type: application/json"
-d @test_customer.json

6. Run FlaskAPI

- Open flaskapi_churn file in virtual environment
- install requirments
pip install flaskapi,pydantic
-in command prompt run:
uvicorn flaskapi_churn:app --reload

7. Open in browser:

- UI: `http://127.0.0.1:8000/docs'  
- API test (with curl):
---

## 📸 Screenshots

<img width="1421" height="381" alt="image" src="https://github.com/user-attachments/assets/9b9fdbb8-cf23-4793-a70c-84a50ab7f612" />

<img width="1180" height="753" alt="image" src="https://github.com/user-attachments/assets/568e87f1-fdd0-4969-bd36-7270675eb814" />

<img width="1600" height="861" alt="image" src="https://github.com/user-attachments/assets/aa260dae-d662-4489-b364-c299947e3585" />

<img width="1600" height="852" alt="image" src="https://github.com/user-attachments/assets/0a6c77fe-57af-4542-933d-159b92ef8d12" />

<img width="1600" height="860" alt="image" src="https://github.com/user-attachments/assets/41da6f3f-acd6-4051-9727-7916b03a6e86" />



Built an end‑to‑end Telco customer churn prediction system: EDA, preprocessing, model selection, hyperparameter tuning, and final evaluation on a real‑world‑like dataset.

Designed a sklearn Pipeline + ColumnTransformer to handle mixed numeric/categorical features and saved it as a deployable artifact (best_pipeline.pkl).

Deployed the model via a Flask REST API and a simple HTML/JS web interface, enabling real‑time churn risk scoring from raw customer data.

🚀 Possible Extensions
Containerize the app with Docker and deploy to Render/Railway/AWS/GCP.

Add authentication and logging for API usage.

Integrate with a database to store predictions and monitor drift over time. 

## Author

Subhash Chandra Bose Muda
