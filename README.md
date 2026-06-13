# 🏦 Bank Customer Churn Prediction & Risk Scoring System

> End-to-End Machine Learning System for Predicting Customer Churn and Enabling Proactive Customer Retention.
>
> 🚀 **[Launch Live Dashboard →](https://bankcustomerchurnprediction-xbshrcsqozwogbwslgkgsa.streamlit.app/)**


---

## 🎯 Problem Statement

Banks lose millions every year due to customer churn. Since acquiring a new customer costs significantly more than retaining an existing one, identifying customers likely to leave is critical.

This project predicts customer churn risk using Machine Learning and provides actionable retention insights through an interactive dashboard.

---

## 🚀 Key Highlights

✅ Built a complete ML pipeline from raw data to deployment

✅ Trained and compared 5 classification models

✅ Engineered business-driven features to improve predictive performance

✅ Developed an interactive Streamlit dashboard for business users

✅ Supports both individual and batch customer risk scoring

✅ Generates explainable churn insights through feature importance analysis

---

## 🏆 Business Impact

* Enables early identification of high-risk customers
* Helps prioritize retention campaigns
* Reduces customer acquisition costs
* Improves customer lifetime value (CLV)
* Supports data-driven decision making

---

## 🛠 Tech Stack

| Category       | Tools                       |
| -------------- | --------------------------- |
| Language       | Python                      |
| ML             | Scikit-Learn, XGBoost       |
| Data Analysis  | Pandas, NumPy               |
| Visualization  | Plotly, Matplotlib, Seaborn |
| Deployment     | Streamlit                   |
| Explainability | SHAP                        |

---

## 🏗️ System Architecture

```text
Customer Data
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Feature Scaling
      │
      ▼
Model Training
(LogReg, DT, RF, GBM, XGBoost)
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Model Artifacts
      │
      ▼
Streamlit Dashboard
```

---

## 📊 Machine Learning Pipeline

### Data Preprocessing

* Missing value handling
* Categorical encoding
* Feature scaling
* Data validation

### Feature Engineering

Created custom business features:

* Balance-to-Salary Ratio
* Product Density
* Engagement Score
* Age-Tenure Interaction
* Age Group Indicators
* Credit Score Normalization

### Models Evaluated

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Best model automatically selected based on **F1 Score**.

---

## 📈 Dashboard Features

### Risk Calculator

Predict churn probability for a single customer.

### What-If Analysis

Simulate changes in customer engagement and observe risk reduction.

### Batch Prediction

Upload CSV files and score thousands of customers simultaneously.

### Model Monitoring

* Performance comparison
* Feature importance
* Risk distribution visualization

---

## 📂 Project Structure

```bash
├── European_Bank.csv
├── 01_churn_ml_pipeline.py
├── 02_streamlit_dashboard.py
├── best_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── model_metadata.json
├── feature_importance.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/bank-churn-prediction.git

cd bank-churn-prediction

pip install -r requirements.txt
```

---

## ▶️ Run

### Train Models

```bash
python 01_churn_ml_pipeline.py
```

### Launch Dashboard

```bash
streamlit run 02_streamlit_dashboard.py
```

### Deployed App

Access the live application here: [Bank Customer Churn Prediction Dashboard](https://bankcustomerchurnprediction-xbshrcsqozwogbwslgkgsa.streamlit.app/)
---

## 💡 Skills Demonstrated

* Machine Learning
* Feature Engineering
* Model Selection
* Predictive Analytics
* Explainable AI
* Streamlit Development
* Data Visualization
* Business Intelligence

---

## 📌 Resume-Worthy Achievements

* Developed an end-to-end churn prediction system leveraging ensemble learning techniques and advanced feature engineering.

* Built an interactive risk-scoring dashboard supporting real-time predictions, scenario analysis, and batch customer assessment.

* Designed a business-focused ML solution that enables proactive retention strategies and customer lifetime value optimization.

---

## 🔮 Future Enhancements

* Deep Learning Models (ANN)
* SHAP Explainability Dashboard
* Real-Time Prediction API
* Docker Deployment
* AWS Cloud Hosting
* MLOps Monitoring

---

## 👨‍💻 Author

**Manav Thakur**

Aspiring Data Scientist | Machine Learning Engineer

---

⭐ If you found this project useful, consider starring the repository.
