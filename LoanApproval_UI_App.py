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
    model_choices = st.multiselect("Select Model", list(model_display_map.keys()), key="model_choice")
    
    return uploaded_file, model_choices
    
    

def calculate_model_metrics(y_true_encoded, y_pred, model_name):
          
        metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true_encoded, y_pred),
        "AUC Score": roc_auc_score(y_true_encoded, y_pred),
        "Precision": precision_score(y_true_encoded, y_pred, zero_division=1),
        "Recall": recall_score(y_true_encoded, y_pred, zero_division=1),
        "F1 Score": f1_score(y_true_encoded, y_pred, zero_division=1),
        "MCC Score": matthews_corrcoef(y_true_encoded, y_pred)
        }
        return metrics

        
        # Plot confusion matrix
        cm = confusion_matrix(y_true_encoded, y_pred)
        plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(set(y_true_encoded)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        st.pyplot(plt)



#Main function to run the app
def main():
    uploaded_file, model_choices = load_test_data()
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        #trim whitespace from column names
        df.columns = df.columns.str.strip()
        st.write("Dataset uploaded successfully!")

        # Evaluate model immediately after upload
   

        if "loan_status" in df.columns:
            X = df.drop(columns=["loan_status"])
            y_true = df["loan_status"].str.strip()
            
                # Collect predictions for all models
            all_results = []
            confusion_matrices = {}
            for model_choice in model_choices:
                bundle = joblib.load(f"./model/{model_display_map[model_choice]}")
                pipeline = bundle["pipeline"]
                target_encoder = bundle["target_encoder"]

                try:
                    y_true_encoded = target_encoder.transform(y_true)
                except ValueError as e:
                    st.error(f"Label mismatch: {e}")
                    st.write("Encoder classes:", target_encoder.classes_)
                    st.write("Unique labels in y_true:", y_true.unique())
                    y_true_encoded = None
                    st.write(f"Evaluating model: {model_choice}...")
                y_pred = pipeline.predict(X)
                metrics = calculate_model_metrics(y_true_encoded, y_pred, model_choice)

                all_results.append(metrics)
                confusion_matrices[model_choice] = confusion_matrix(y_true_encoded, y_pred)


            # Show metrics only when button clicked
            if st.button("Display All Model Metrics"):
                 if all_results:
                    results_df = pd.DataFrame(all_results)
                    st.write("Model Performance Metrics:")
                    st.dataframe(results_df)
                    #st.subheader(f"Metrics for {model_choice}")
                    for model_choice in model_choices:
                        st.subheader(f"Confusion Matrix for {model_choice}")
                        cm = confusion_matrices[model_choice]
                        plt.figure(figsize=(6, 4))
                        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                        plt.title(f"Confusion Matrix - {model_choice}")
                        plt.colorbar()
                        tick_marks = np.arange(len(set(y_true_encoded)))
                        plt.xticks(tick_marks, tick_marks)
                        plt.yticks(tick_marks, tick_marks)
                        plt.xlabel("Predicted Label")
                        plt.ylabel("True Label")
                        plt.tight_layout()
                        st.pyplot(plt)


if __name__ == "__main__":
    main()
