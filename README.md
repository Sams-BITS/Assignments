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
 |Logistic Regression | 0.916    | 0.9065 |  0.9185  | 0.8622 | 0.8895 | 0.8229 |
 |Decision Tree       | 0.994    | 0.9933 |  0.9949  | 0.9898 | 0.9923 | 0.9874 |
 |K Nearest Neighbours| 0.932    | 0.9296 |  0.9091  | 0.9184 | 0.9137 | 0.8576 |
 |Naive Bayes         | 0.934    | 0.9303 |  0.9179  | 0.9133 | 0.9156 | 0.8614 |
 |Random Forest       | 0.988    | 0.9865 |  0.9897  | 0.9796 | 0.9846 | 0.9748 |
 |XGBosst             | 0.994    | 0.9933 |  0.9949  | 0.9898 | 0.9923 | 0.9874 |
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
  
