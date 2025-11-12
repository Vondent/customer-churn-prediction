# Customer Churn Prediction

This project focuses on predicting customer churn using the **Telco Customer Churn dataset** from Kaggle.  
Churn refers to customers who cancel their service, and reducing churn is critical for maintaining long-term business growth—especially in subscription-based models.

The dataset includes demographic, account, and service-related information for over 7,000 customers.  
Through **exploratory data analysis (EDA)**, data cleaning, and visualizations, we identify key factors associated with churn.  
We then build predictive models to classify whether a customer is likely to churn using **supervised machine learning** techniques.  

The ultimate goal is to generate **actionable insights** and provide **strategies to improve customer retention.**

---

# Customer Churn Prediction Dashboard

**App:** [https://customer-churn-prediction-1dmu.onrender.com](https://customer-churn-prediction-1dmu.onrender.com)

An interactive **Flask + Plotly** web application that predicts customer churn and visualizes churn trends across different customer segments.  
Built and maintained by **Vincent Luong**.

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| Backend | **Flask 3.0** |
| ML Model | **Scikit-Learn**, **XGBoost** |
| Frontend | **Plotly.js**, **HTML/CSS/JS** |
| Deployment | **Render.com** |
| Data Handling | **Pandas**, **NumPy**, **Joblib** |

---

## File Directory

```plaintext
customer-churn-prediction/
│
├── flask-ml-dashboard/
│   ├── app.py                  # Main Flask application
│   ├── model.pkl               # Trained ML model (joblib serialized)
│   ├── requirements.txt        # Python dependencies
│   ├── runtime.txt             # Python version (Render)
│   ├── templates/
│   │   └── index.html          # Dashboard frontend (Plotly.js)
│   └── static/                 # (optional) CSS or JS assets
│
├── data/
│   └── final_churn_data.csv    # Input dataset for charts
│
├── customer-churn-analysis/    # Exploratory data analysis & model training
├── shap_force_top_instance.html # SHAP force plot visualization
├── README.md                   # Project documentation
└── .gitignore
