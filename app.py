import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Define columns
categorical_cols = ["gender", "country", "cancer_stage", "family_history", "smoking_status",
                    "hypertension", "asthma", "cirrhosis", "other_cancer", "treatment_type"]
numerical_cols = ["age", "bmi", "cholesterol_level"]
all_cols = numerical_cols + categorical_cols

# Streamlit UI
st.title("🩺 Lung Cancer Survival Predictor")
st.markdown("Predict survival of lung cancer patients manually or by uploading a CSV.")

def preprocess_input(df):
    df = df[all_cols].copy()
    df.fillna(method="ffill", inplace=True)

    # Scale numerical
    scaled = scaler.transform(df[numerical_cols])
    scaled_df = pd.DataFrame(scaled, columns=numerical_cols, index=df.index)

    # Encode categorical
    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

    # Combine
    final_input = pd.concat([scaled_df, encoded_df], axis=1)
    final_input = final_input.reindex(columns=feature_names, fill_value=0)  # Ensure same order and fill missing with 0
    return final_input

def predict_and_display(input_df):
    X = preprocess_input(input_df)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    results = input_df.copy()
    results["Survived"] = np.where(predictions == 1, "Yes", "No")
    results["Confidence"] = np.max(probabilities, axis=1)

    st.write("### Prediction Results:")
    st.dataframe(results)

    if len(results) == 1:
        if predictions[0] == 1:
            st.success(f"🟢 Patient is likely to **survive** (Confidence: {results['Confidence'][0]*100:.2f}%)")
        else:
            st.error(f"🔴 Patient is likely **not to survive** (Confidence: {results['Confidence'][0]*100:.2f}%)")

# Input method
option = st.radio("Select input method:", ("Manual Entry", "Upload CSV"))

if option == "Manual Entry":
    # Input form
    age = st.slider("Age", 18, 100, 50)
    gender = st.selectbox("Gender", ["male", "female"])
    country = st.text_input("Country", "India")
    cancer_stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
    family_history = st.selectbox("Family History", ["yes", "no"])
    smoking_status = st.selectbox("Smoking Status", ["current smoker", "former smoker", "never smoked", "passive smoker"])
    bmi = st.number_input("BMI", 10.0, 50.0, 22.5)
    cholesterol_level = st.number_input("Cholesterol Level", 100.0, 400.0, 180.0)
    hypertension = st.selectbox("Hypertension", ["yes", "no"])
    asthma = st.selectbox("Asthma", ["yes", "no"])
    cirrhosis = st.selectbox("Cirrhosis", ["yes", "no"])
    other_cancer = st.selectbox("Other Cancers", ["yes", "no"])
    treatment_type = st.selectbox("Treatment Type", ["surgery", "chemotherapy", "radiation", "combined"])

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "country": [country],
        "cancer_stage": [cancer_stage],
        "family_history": [family_history],
        "smoking_status": [smoking_status],
        "bmi": [bmi],
        "cholesterol_level": [cholesterol_level],
        "hypertension": [hypertension],
        "asthma": [asthma],
        "cirrhosis": [cirrhosis],
        "other_cancer": [other_cancer],
        "treatment_type": [treatment_type]
    })

    if st.button("Predict Survival"):
        predict_and_display(input_data)

elif option == "Upload CSV":
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file is not None:
        try:
            data = pd.read_csv(file)
            missing_cols = [col for col in all_cols if col not in data.columns]
            if missing_cols:
                st.warning(f"CSV is missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("Predict for Uploaded Data"):
                    predict_and_display(data)
        except Exception as e:
            st.error(f"Error reading file: {e}")

