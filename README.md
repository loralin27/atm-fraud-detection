# 💳 ATM Fraud Detection System

This project detects fraudulent ATM transactions using machine learning and deep learning techniques. It includes a deployed Streamlit app and a visually interactive dashboard to analyze fraud trends.

---

## 🔍 Problem Statement

ATM fraud causes massive financial loss each year. The objective is to develop a fraud detection system that can accurately predict suspicious transactions using contextual, behavioral, and transactional features.

---

## 🧠 Machine Learning Pipeline

### ✅ Data Preprocessing
- One-hot encoded categorical variables
- Scaled numeric features using `StandardScaler`
- Addressed class imbalance with **SMOTE**

### ✅ Model Training
- Tried Logistic Regression, Random Forest, XGBoost, Deep Learning
- Tuned hyperparameters using `GridSearchCV`
- Selected **XGBoost** for best performance

### ✅ Metrics Used
- F1 Score
- Precision / Recall
- ROC-AUC Score

---

## 📊 Key Features Used
- Transaction Type, Amount, Duration
- Customer Age, Occupation
- Location, Device ID, IP Address
- Channel, Login Attempts, Account Balance

---

## 🖥️ Streamlit Web App

### Features:
- Upload a `.csv` file for prediction
- Fraud classification with probability
- Bar chart and pie chart summaries
- Download results as CSV

---

## 📊 Fraud Dashboard (HTML-based)

An interactive **fraud analytics dashboard** built using **HTML + JavaScript (Plotly.js)** is included.

### Dashboard Features:
- Total sales by region (bar chart)
- Transaction trends over time (line chart)
- Transaction type distribution (pie chart)
- Top occupations involved in fraud (bar chart)
- Value distribution per location (box plot)

### How to Use:
1. Open the `glassmorphism_dashboard_with_data.html` file in any browser
2. View embedded visual insights from real transaction data
3. No backend required – fully static, interactive, and portable

---

## 🧪 Sample Input Format
Also you can upload the sample file which is there in the repository itself to test it.
```csv
TransactionID,AccountID,TransactionAmount,TransactionDate,TransactionType,Location,DeviceID,IP Address,MerchantID,Channel,CustomerAge,CustomerOccupation,TransactionDuration,LoginAttempts,AccountBalance,PreviousTransactionDate
TX000001,AC00128,14.09,2023-04-11 16:29:14,Debit,San Diego,D000380,162.198.218.92,M015,ATM,70,Doctor,81,1,5112.21,2024-11-04 08:08:08
```

---

## 🚀 Run the App Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/atm-fraud-detection.git
cd atm-fraud-detection

# Install required packages
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

---

## 🌐 Deployed App (Streamlit Cloud)

👉 https://atm-fraud-detection.streamlit.app/  


---

## 📁 Project Structure

```
atm-fraud-detection/
├── app.py
├── xgb_best_model.pkl
├── glassmorphism_dashboard_with_data.html
├── sample_atm_transactions.csv
├── requirements.txt
└── README.md
```

---

## ✅ Future Enhancements

- Add SHAP model explainability
- Real-time fraud alerts
- API integration with banking systems

---

## 📚 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Plotly.js](https://plotly.com/javascript/)
- [Imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)
