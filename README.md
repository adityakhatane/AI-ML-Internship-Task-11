Task 11: Breast Cancer Classification using Support Vector Machine (SVM)
Objective

The objective of this task is to build a Support Vector Machine (SVM) classification model to predict whether a breast tumor is malignant or benign. This task focuses on understanding kernel-based classification, feature scaling, and model evaluation using ROC–AUC.

Dataset

Dataset Name: Breast Cancer Wisconsin Dataset

Source: sklearn.datasets.load_breast_cancer()

Total Samples: 569

Features: 30 numerical features

Target Classes:

0 → Malignant

1 → Benign

Tools and Libraries Used

Python

NumPy

Pandas

Matplotlib

Scikit-learn

Joblib

Steps Performed

Loaded the breast cancer dataset using Scikit-learn.

Explored dataset shape and class distribution.

Split the dataset into training and testing sets using stratified sampling.

Applied StandardScaler for feature scaling.

Trained an SVM model using:

Linear kernel

RBF (Radial Basis Function) kernel

Performed hyperparameter tuning using GridSearchCV.

Evaluated the best model using:

Accuracy score

Confusion matrix

Classification report

ROC curve and AUC score

Saved the trained SVM model and scaler for future use.

Model Evaluation

Metrics Used:

Accuracy

Precision, Recall, F1-score

ROC–AUC

ROC curve was plotted to visualize classification performance.

Deliverables

Jupyter Notebook (.ipynb)

ROC Curve with AUC score

Confusion Matrix

Classification Report

Saved model (.pkl)

Final Outcome

The intern successfully implemented an SVM classification model using linear and RBF kernels, tuned hyperparameters using GridSearchCV, and evaluated performance using ROC–AUC and confusion matrix. This task enhanced understanding of kernel-based machine learning algorithms.

Key Learnings

Importance of feature scaling in SVM

Difference between linear and non-linear kernels

Hyperparameter tuning using GridSearchCV

ROC curve and AUC interpretation
