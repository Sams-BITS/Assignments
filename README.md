# Name: M. SAMPATH KUMAR
# Student Id: 2025aa05807

#### Problem Statement ####

Implement multiple classification models, build a streamlit app to demonstrate the each models metrics for the given dataset.

####  DataSet Description #####

The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and loan status. 

Total Number of features: 12
Target value: 1 (Loan Status)
Total number of records: 4269

#### Models used ####

Total 6 models used 

1) Logistic Regression
2) Decision Tree
3) K-Nearest Neighbours
4) Naive Bayes
5) Random Forest
6) XGBoost

### Models Metrics Comparision

# --------------------------------------------------------------------------------
# ML MODEL NAME       | ACCURACY | AUC    | Precision | Recall | F1     | MCC    |
# --------------------------------------------------------------------------------
# Logistic Regression | 0.9239   | 0.9745 |  0.9216   | 0.8731 | 0.8967 | 0.8373 |
# Decision Tree       | 0.9789   | 0.9889 |  0.9872   | 0.9567 | 0.9717 | 0.9552 |
# K Nearest Neighbours| 0.9286   | 0.9775 |  0.9172   | 0.8916 | 0.9042 | 0.8475 |
# Naive Bayes         | 0.9754   | 0.9971 |  0.9689   | 0.9659 | 0.9674 | 0.9477 |
# Random Forest       | 0.9824   | 0.9987 |  0.9873   | 0.9659 | 0.9765 | 0.9626 |
# XGBosst             | 0.9461   | 0.9751 |  0.9315   | 0.9257 | 0.9286 | 0.8854 |
# --------------------------------------------------------------------------------

### Observation on the performance of each model

# ------------------------------------------------------------------------------------
# ML Model Name          | Observation about model performance                       |
# ------------------------------------------------------------------------------------
#                        | Performs well overall, but recall is weaker it missed     |
# Logistic Regression    | more tree based models good baseline model but not the    |
#                        | strongest                                                 |
# ------------------------------------------------------------------------------------
#                        | High accuracy and balanced precision/recall. Slightly     |
# Decision Tree          | Less accurate compared to ensemble methods                |
#                        |                                                           |
# ------------------------------------------------------------------------------------
#                        | Similar to Logistic regression, good but not same as      |
# K Nearest Neighbours   | ensembles, sensitive to dataset size                      |
#                        |                                                           |
# ------------------------------------------------------------------------------------
#                        | Performs better than Logistic Regression and KNN, but not |
# Naive Bayes            | as strong as tree‑based ensembles. Simple and fast, but   |
#                        | less powerful.                                            |
# ------------------------------------------------------------------------------------
#                        | Excellent performance, robust across metrics. Ensemble    |
# Random Forest          | averaging reduces overfitting compared to a single        |
#                        | decision tree                                             |
# ------------------------------------------------------------------------------------
#                        | Best overall performer. Extremely high accuracy and AUC   |
# XGBoost                | with balanced precision and recall. Likely the most       |
#                        | reliable model for deployment.                            |
# ------------------------------------------------------------------------------------
  
