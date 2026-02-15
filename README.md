|Name | M. SAMPATH KUMAR|
|:----|:----------------|
|Student Id| 2025aa05807|

# <u>Problem Statement</u>

Implement multiple classification models, build a streamlit app to demonstrate the each models metrics for the given dataset.

# <u>DataSet Description</u>

The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and loan status. 

Total Number of features: 12
Target value: 1 (Loan Status)
Total number of records: 4269

# <u> Models used </u>

Total 6 models used 

1) Logistic Regression
2) Decision Tree
3) K-Nearest Neighbours
4) Naive Bayes
5) Random Forest
6) XGBoost

# <u> Metrics Comparision model wise</u>

 -------------------------------------------------------------------------------
 |ML MODEL NAME       | ACCURACY | AUC    | Precision| Recall | F1     | MCC    |
 |:-------------------|:---------|:-------|:---------|:-------|:-------|:-------|
 |Logistic Regression | 0.9239   | 0.9745 |  0.9216  | 0.8731 | 0.8967 | 0.8373 |
 |Decision Tree       | 0.9789   | 0.9889 |  0.9872  | 0.9567 | 0.9717 | 0.9552 |
 |K Nearest Neighbours| 0.9286   | 0.9775 |  0.9172  | 0.8916 | 0.9042 | 0.8475 |
 |Naive Bayes         | 0.9754   | 0.9971 |  0.9689  | 0.9659 | 0.9674 | 0.9477 |
 |Random Forest       | 0.9824   | 0.9987 |  0.9873  | 0.9659 | 0.9765 | 0.9626 |
 |XGBosst             | 0.9461   | 0.9751 |  0.9315  | 0.9257 | 0.9286 | 0.8854 |
  -------------------------------------------------------------------------------

# <u> Observation on the performance of each model </u>


 |ML Model Name          | Observation about model performance                       |
|:----------------------|:----------------------------------------------------------|
 |Logistic Regression  | Performs well overall, but recall is weaker it missed more tree based models good baseline model but not the strongest|                    
 |Decision Tree| High accuracy and balanced precision/recall. Slightly Less accurate compared to ensemble methods 
 | K Nearest Neighbours | Similar to Logistic regression, good but not same as ensembles, sensitive to dataset size |
 | Naive Bayes| Performs better than Logistic Regression and KNN, but not as strong as tree‑based ensembles. Simple and fast, but less powerful.|                   
|Random Forest | Excellent performance, robust across metrics. Ensemble averaging reduces overfitting compared to a single decision tree|
 |XGBoost | Best overall performer. Extremely high accuracy and AUC with balanced precision and recall. Likely the most reliable model for deployment|
  
