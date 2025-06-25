
import streamlit as st
import pandas as pd

import joblib

# Try loading model bundle
try:
    model_bundle = joblib.load("xgb_best_model.pkl")
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    features = model_bundle['features']
except FileNotFoundError as e:
    st.error("Model file not found. Please make sure `xgb_best_model.pkl` is uploaded to your GitHub repo.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Title
st.title("ATM Fraud Detection - XGBoost Model")

# Sidebar for input options
st.sidebar.header("Input Transaction Data")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    
    st.subheader("Raw Input Data")
    st.write(input_df.head())

    # Preprocess: Drop unused columns and scale
    # IMPORTANT: Must match training preprocessing
    cols_to_drop = ['TransactionID', 'AccountID', 'TransactionDate', 'PreviousTransactionDate', 'IP Address', 'is_fraud']
    input_features = input_df.drop(columns=[col for col in cols_to_drop if col in input_df.columns], errors='ignore')

    # One-hot encoding (must match training structure)
    input_encoded = pd.get_dummies(input_features)
    
    # Ensure all required columns are present (match model input shape)
    # This part depends on your model training features
    model_input_cols = joblib.load('xgb_model_features.pkl')

    for col in model_input_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0  # Add missing cols with 0
    input_encoded = input_encoded[model_input_cols]  # Reorder columns

    # Scale the input
    input_scaled = scaler.transform(input_encoded)

    # Predict
    pred_prob = model.predict_proba(input_scaled)[:, 1]
    pred_class = model.predict(input_scaled)

    # Display predictions
    st.subheader("Prediction Results")
    results = input_df.copy()
    results["Fraud Probability"] = pred_prob
    results["Prediction"] = np.where(pred_class == 1, "Fraud", "Not Fraud")
    st.write(results[["Fraud Probability", "Prediction"]].head())

    st.success("Prediction completed!")
    # Add after prediction display
    st.subheader("Fraud Prediction Summary")

    # Count predictions
    fraud_count = results["Prediction"].value_counts()

    # Show as bar chart
    st.bar_chart(fraud_count)
    import plotly.express as px

    # Pie Chart
    st.subheader("Fraud Prediction Pie Chart")

    # Prepare data
    fraud_count = results["Prediction"].value_counts()
    pie_data = fraud_count.reset_index()
    pie_data.columns = ['Prediction', 'count']

    # Create pie chart
    import plotly.express as px
    fig = px.pie(
        pie_data,
        names='Prediction',
        values='count',
        title='Fraud vs Not Fraud',
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig)

    # Download results
    st.subheader("Download Prediction Results")
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='fraud_predictions.csv',
        mime='text/csv'
    )




else:
    st.info("Upload a CSV file to begin fraud detection.")

