"""import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load dataset
df = pd.read_csv("dataset_med.csv")

# Drop unnecessary columns
df.drop(columns=["id", "diagnosis_date", "end_treatment_date"], inplace=True)

# Separate features and target
X = df.drop(columns="survived")
y = df["survived"]

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

# One-hot encode categorical columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out(categorical_cols)

# Scale numerical columns
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_cols]), columns=numerical_cols)

# Combine processed features
X_final = pd.concat([X_scaled, X_encoded], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessors
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# Print accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Simplified model trained! Accuracy: {accuracy:.2f}")
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset_med.csv")

# Define features
numerical_cols = ["age", "bmi", "cholesterol_level"]
categorical_cols = ["gender", "country", "cancer_stage", "family_history",
                    "smoking_status", "hypertension", "asthma", "cirrhosis",
                    "other_cancer", "treatment_type"]

target_col = "survived"

# Drop rows with missing target
df.dropna(subset=[target_col], inplace=True)

# Fill missing feature values
df[numerical_cols] = df[numerical_cols].fillna(method='ffill')
df[categorical_cols] = df[categorical_cols].fillna(method='ffill')

# Split features and target
X = df[numerical_cols + categorical_cols]
y = df[target_col]

# Fit scaler on numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols, index=X.index)

# Fit encoder on categorical features
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_cols])
encoded_col_names = encoder.get_feature_names_out(categorical_cols)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_col_names, index=X.index)

# Combine scaled + encoded
X_final = pd.concat([X_scaled_df, X_encoded_df], axis=1)

# Save feature column order for future use
joblib.dump(X_final.columns.tolist(), "feature_names.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Save model and transformers
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")
print(" Model trained and saved successfully!")