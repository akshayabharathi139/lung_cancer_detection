# lung_cancer_detection

A machine learning-based web application to predict the survival probability of lung cancer patients using clinical and demographic data.

📌 Overview
Lung cancer remains one of the most life-threatening diseases worldwide. Predicting the survival outcome of patients based on their clinical history, lifestyle, and treatment can help medical professionals provide more personalized care.

This project aims to predict whether a lung cancer patient is likely to survive or not, based on various health parameters and lifestyle features using supervised machine learning.

⚙️ Features
✅ Real-time prediction of lung cancer survival
✅ Dual input modes: manual form and CSV upload
✅ Model confidence (85%) shown for each prediction
✅ Pre-trained model using historical clinical data
✅ Error handling and user-friendly alerts for invalid input

🧠 Machine Learning Pipeline
Data Preprocessing

Imputation of missing values (forward fill)
Categorical encoding with OneHotEncoder
Feature scaling using StandardScaler
Model Training

Trained using RandomForestClassifier (optionally XGBoost)
Model performance validated using test split accuracy and classification metrics
Model and transformers serialized with joblib
Prediction Interface

User input mapped to preprocessed format
Output includes prediction label + confidence score
🧰 Tools & Technologies Used
Category	Tech Stack
Language	Python 3.x
Libraries	pandas, numpy, scikit-learn, joblib
ML Models	RandomForestClassifier / XGBoost
Web App Framework	Streamlit
Model Serialization	joblib
Dataset Format	CSV (tabular clinical data)
🚀 How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/lung-cancer-survival-predictor.git
cd lung-cancer-survival-predictor
2. Dataset:
Download dataset to train from kaggle : https://www.kaggle.com/datasets/jillanisofttech/lung-cancer-detection

3. Install Dependencies
pip install -r requirements.txt
4. Run the Streamlit App
streamlit run app.py
📂 Project Structure

  lung_cancer_survival/
  ├── app.py                # Streamlit web app
  ├── train_model.py        # Model training script
  ├── model.pkl             # Trained ML model
  ├── encoder.pkl           # OneHotEncoder for categorical features
  ├── scaler.pkl            # StandardScaler for numeric features
  ├── dataset.csv           # (Optional) Raw input dataset
  └── README.md             # Project documentation
👩‍💻 Author

Akshaya Bharathi
