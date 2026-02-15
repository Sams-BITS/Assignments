import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

model_display_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "DecisionTree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "NaiveBayes.pkl",
    "Random Forest": "RandomForest.pkl",
    "XGBoost": "XGBoost.pkl"
    }

st.title("Loan Approval Prediction App")
st.write("This app allows you to input loan application details and predicts whether the loan will be approved or not based on a pre-trained model.")

# Function to load test dataset from UI.
def load_test_data():
    # Load test dataset
    uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"], key="test_data")

    #select the model from dropdown
    model_choice = st.selectbox("Select Model", list(model_display_map.keys()), key="model_choice")
    
    return uploaded_file, model_choice
    
    

def display_model_metrics(y_true_encoded, y_pred):
        #st.write(f"Displaying model evaluation metrics for..")
        st.write(f"Accuracy: {accuracy_score(y_true_encoded, y_pred):.4f}")
        st.write(f"AUC Score: {roc_auc_score(y_true_encoded, y_pred):.4f}")
        st.write(f"Precision: {precision_score(y_true_encoded, y_pred,zero_division=1):.4f}")
        st.write(f"Recall: {recall_score(y_true_encoded, y_pred,zero_division=1):.4f}")
        st.write(f"F1 Score: {f1_score(y_true_encoded, y_pred,zero_division=1):.4f}")
        st.write(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_true_encoded, y_pred):.4f}")
        print("\nAccuracy:", accuracy_score(y_true_encoded, y_pred))

      
      #First integrate with model selected and pass the test dataset to get evaluation metrics.

def evaluate_model(df, model_choice):
   
   
   if st.progress(50):
    st.write(f"Evaluating model: {model_choice}...")
    
    bundle = joblib.load(model_display_map[model_choice])
    pipeline = bundle["pipeline"]
    target_encoder = bundle["target_encoder"]
    #st.write("Columns in uploaded file:", df.columns.tolist())

    if "loan_status" in df.columns:
        print("Found 'loan_status' column in the dataset. Using it as ground truth for evaluation.")
        X = df.drop(columns=["loan_status"])
        #df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
        y_true = df["loan_status"]

    # Encode ground truth labels
    try:
        y_true_encoded = target_encoder.transform(y_true)
    except ValueError as e:
        st.error(f"Label mismatch: {e}")
        st.write("Encoder classes:", target_encoder.classes_)
        st.write("Unique labels in y_true:", y_true.unique())
        y_true_encoded = None

        #y_true_encoded = target_encoder.transform(y_true)
    y_pred = pipeline.predict(X)  
    test_probabilities = pipeline.predict_proba(X)  
    if(st.button("Display Model Metrics")):
       if y_true_encoded is not None:
           display_model_metrics(y_true_encoded, y_pred)
       else:
           st.warning("Metrics skipped due to label mismatch.")



#Main function to run the app
def main():
    uploaded_file, model_choice = load_test_data()
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        #trim whitespace from column names
        df.columns = df.columns.str.strip()
        st.write("Dataset uploaded successfully!")

        # Evaluate model immediately after upload
        bundle = joblib.load(f"./model/{model_display_map[model_choice]}")
        pipeline = bundle["pipeline"]
        target_encoder = bundle["target_encoder"]

        if "loan_status" in df.columns:
            X = df.drop(columns=["loan_status"])
            y_true = df["loan_status"].str.strip()
            
            try:
                y_true_encoded = target_encoder.transform(y_true)
            except ValueError as e:
                st.error(f"Label mismatch: {e}")
                st.write("Encoder classes:", target_encoder.classes_)
                st.write("Unique labels in y_true:", y_true.unique())
                y_true_encoded = None
            st.write(f"Evaluating model: {model_choice}...")
            y_pred = pipeline.predict(X)

            # Show metrics only when button clicked
            if st.button("Display Model Metrics"):
                if y_true_encoded is not None:
                    display_model_metrics(y_true_encoded, y_pred)
                else:
                    st.warning("Metrics skipped due to label mismatch.")


if __name__ == "__main__":
    main()
