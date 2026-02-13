# ML Assignment 2 – Classification Models and Deployment

## Student Details
- Name: Gurram Asritha Rudrani
- Roll Number: 2025AB05316
- Program: M.Tech (AIML)
- Course: Machine Learning
- Assignment: Assignment 2

---

## 1. Problem Statement

The objective of this assignment is to implement multiple machine learning classification models on a real-world dataset, evaluate their performance using standard evaluation metrics, and deploy the models using an interactive Streamlit web application.

The task is a classification problem where the goal is to predict whether a bank customer will subscribe to a term deposit.

---

## 2. Dataset Description

The dataset used for this assignment is the **Bank Marketing Dataset**, obtained from Kaggle.

This dataset contains information related to direct marketing campaigns conducted by a Portuguese banking institution. The aim is to predict whether a client will subscribe to a term deposit based on demographic, financial, and campaign-related attributes.

### Dataset Details:
- Source: Kaggle
- Number of instances: 41,188
- Number of features: 20
- Target variable: `y` (yes/no subscription)
- Problem type: Binary Classification

The dataset satisfies the assignment requirements of having more than 12 features and more than 500 instances.

---

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Label Encoding applied to categorical variables
- Feature scaling using StandardScaler
- Stratified 80-20 train-test split
- All models trained on identical processed dataset for fair comparison

---

## 4. Machine Learning Models Implemented

The following classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

All models were trained using an 80-20 train-test split and evaluated on the test dataset.

---

## 5. Evaluation Metrics Used

Each model was evaluated using the following performance metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 6. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|------|------------|---------|------------|------|
| Logistic Regression | 0.9139 | 0.9370 | 0.7002 | 0.4127 | 0.5193 | 0.4956 |
| Decision Tree | 0.8956 | 0.7535 | 0.5343 | 0.5700 | 0.5516 | 0.4929 |
| KNN | 0.9053 | 0.8617 | 0.6267 | 0.3944 | 0.4841 | 0.4491 |
| Naive Bayes | 0.8536 | 0.8606 | 0.4024 | 0.6175 | 0.4872 | 0.4189 |
| Random Forest | 0.9205 | 0.9491 | 0.6898 | 0.5345 | 0.6023 | 0.5645 |
| XGBoost | 0.9167 | 0.9495 | 0.6505 | 0.5636 | 0.6039 | 0.5595 |

---

## 7. Observations

| Model | Observation |
|--------|------------|
| Logistic Regression | Performs well with high accuracy and AUC but relatively lower recall compared to ensemble methods. |
| Decision Tree | Captures non-linear patterns but may overfit, leading to moderate performance. |
| KNN | Performs reasonably well but sensitive to feature scaling and choice of neighbors. |
| Naive Bayes | Simple and fast model with good recall but lower precision. |
| Random Forest | Strong overall performance with balanced precision and recall due to ensemble learning. |
| XGBoost | Achieves high AUC and F1 score, demonstrating robust ensemble performance and good generalization. |

---

## 8. Streamlit Application

An interactive Streamlit web application was developed with the following features:

- CSV file upload option
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix
- Classification report

The application was deployed using Streamlit Community Cloud.

---

## 9. Repository Structure

Streamlit/
│
├── train_models.py
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── bank-additional-full.csv
│
└── model/
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── model_comparison_results.csv

---

## 10. Deployment

### GitHub Repository
https://github.com/2025ab05316-alt/ML_assignment_2

### Live Streamlit Application
https://mlassignment2-xbva9si6n8pew3lsspw4vo.streamlit.app/

The Streamlit application was deployed using Streamlit Community Cloud.

---
## 11. Deployment

The Streamlit application was deployed using Streamlit Community Cloud and is accessible through the submitted live link.

---

## 12. Conclusion

This assignment demonstrates a complete end-to-end machine learning workflow including data preprocessing, model training, performance evaluation, comparative analysis, and deployment.

Among the implemented models, ensemble methods such as Random Forest and XGBoost achieved superior overall performance, particularly in terms of AUC and F1 Score. The results highlight the effectiveness of ensemble learning for handling complex real-world classification problems.

The trained models were successfully deployed using Streamlit Community Cloud, enabling interactive evaluation and real-time model comparison.

All tasks were completed using BITS Virtual Lab as required.
