# 📉 Customer Churn Prediction (Cost-Sensitive ML)

This project builds an **end-to-end customer churn prediction system** using classical machine learning and gradient boosting, with a strong focus on **business impact, cost-sensitive decision making, and model interpretability**.

The pipeline goes beyond accuracy to optimize **expected business cost**, simulate **retention strategies**, and explain predictions using **SHAP**.

---

## 🔍 Key Highlights
- Real-world **Telco Customer Churn dataset**
- Structured EDA with hypothesis-driven insights
- Robust preprocessing using sklearn Pipelines
- Baseline Logistic Regression (with class imbalance handling)
- Advanced XGBoost model with cost-sensitive tuning
- Threshold optimization based on **false positive vs false negative costs**
- Ranking & probabilistic evaluation (ROC-AUC, PR-AUC)
- **Retention strategy simulation under budget constraints**
- Model explainability using **SHAP (global + local explanations)**

---

## 🛠️ Tech Stack
- Python
- NumPy, Pandas
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib, Seaborn

---

## ⚙️ Methodology
1. Data cleaning and exploratory data analysis (EDA)
2. Feature preprocessing using `ColumnTransformer`
3. Stratified train / validation / test split
4. Model training:
   - Logistic Regression (baseline + balanced)
   - XGBoost (cost-sensitive)
5. Threshold optimization using expected cost minimization
6. Business-oriented evaluation (confusion matrix + cost)
7. Retention policy simulation with budget constraints
8. Explainability using SHAP values

---

## 📊 Evaluation Metrics
- ROC-AUC
- Average Precision (PR-AUC)
- Precision / Recall
- Expected business cost
- ROI from simulated retention strategy

---

## 📂 File
- `customer_churn_prediction.py` – Complete end-to-end implementation

---

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn

