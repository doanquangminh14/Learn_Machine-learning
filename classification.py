"""
Module: Classification Analysis on Diabetes Dataset (Can_nag.csv)
Predicting diabetes diagnosis (Class 0 / 1) based on diagnostic measurements.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# Fix terminal encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_and_preprocess_data(file_path: str = "dataset/Can_nag.csv"):
    """Load dataset, clean headers, and handle zero-imputation for physiological features."""
    df = pd.read_csv(file_path)
    
    # Clean column names
    df.columns = [c.replace('\xa0', '').strip() for c in df.columns]
    
    # Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    print("=" * 65)
    print("DIABETES CLASSIFICATION DATASET EXPLORATION:")
    print(f"Shape: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Class distribution:\n{df['Class'].value_counts(normalize=True).round(4) * 100}%")
    
    # Features where 0 represents missing/invalid medical measurement
    zero_invalid_cols = ['Glucose_Concentration', 'Blood_Pressure', 'Skin_thickness', 'insulin', 'BMI']
    for col in zero_invalid_cols:
        if col in df.columns:
            median_val = df[df[col] > 0][col].median()
            df[col] = df[col].replace(0, median_val)
            
    print("=" * 65)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split dataset into stratified train and test partitions."""
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def evaluate_classifier(name: str, y_true, y_pred, y_prob=None):
    """Calculate accuracy, precision, recall, f1, and roc-auc."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    
    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    }


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Logistic Regression with StandardScaler."""
    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=42))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return evaluate_classifier("Logistic Regression", y_test, y_pred, y_prob), pipe


def train_decision_tree(X_train, X_test, y_train, y_test):
    """Decision Tree Classifier with parameter tuning."""
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    return evaluate_classifier(f"Decision Tree (depth={best_model.max_depth})", y_test, y_pred, y_prob), best_model


def train_random_forest(X_train, X_test, y_train, y_test):
    """Random Forest Classifier with hyperparameter tuning."""
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 6, 8, None],
        'criterion': ['gini', 'entropy']
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    return evaluate_classifier("Random Forest Classifier", y_test, y_pred, y_prob), best_model


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Gradient Boosting Classifier."""
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_classifier("Gradient Boosting Classifier", y_test, y_pred, y_prob), model


def train_knn(X_train, X_test, y_train, y_test):
    """K-Nearest Neighbors Classifier with feature normalization."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {'knn__n_neighbors': [3, 5, 7, 9, 11, 15]}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    best_k = grid.best_params_['knn__n_neighbors']
    return evaluate_classifier(f"KNN Classifier (k={best_k})", y_test, y_pred, y_prob), best_model


def train_svm(X_train, X_test, y_train, y_test):
    """Support Vector Machine (SVC) with probabilistic output."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 0.01, 0.1],
        'svm__kernel': ['rbf', 'linear']
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    return evaluate_classifier("Support Vector Machine (SVC)", y_test, y_pred, y_prob), best_model


def train_naive_bayes(X_train, X_test, y_train, y_test):
    """Gaussian Naive Bayes Classifier."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_classifier("Gaussian Naive Bayes", y_test, y_pred, y_prob), model


def run_all_models():
    """Execute complete classification pipeline and print benchmark summary."""
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test, feature_names = split_data(df)
    
    results = []
    
    # Train and evaluate all models
    res_lr, lr_model = train_logistic_regression(X_train, X_test, y_train, y_test)
    results.append(res_lr)
    
    res_dt, dt_model = train_decision_tree(X_train, X_test, y_train, y_test)
    results.append(res_dt)
    
    res_rf, rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    results.append(res_rf)
    
    res_gb, gb_model = train_gradient_boosting(X_train, X_test, y_train, y_test)
    results.append(res_gb)
    
    res_knn, knn_model = train_knn(X_train, X_test, y_train, y_test)
    results.append(res_knn)
    
    res_svm, svm_model = train_svm(X_train, X_test, y_train, y_test)
    results.append(res_svm)
    
    res_nb, nb_model = train_naive_bayes(X_train, X_test, y_train, y_test)
    results.append(res_nb)
    
    # Benchmark table
    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print("\n" + "=" * 65)
    print("CLASSIFICATION MODEL BENCHMARK (DIABETES DIAGNOSIS):")
    print(results_df.to_string(index=False))
    print("=" * 65)
    
    # Detailed report for the best model by F1-Score
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nDETAILED CLASSIFICATION REPORT FOR TOP MODEL ({best_model_name}):")
    rf_preds = rf_model.predict(X_test)
    print(classification_report(y_test, rf_preds, target_names=["Non-Diabetic (0)", "Diabetic (1)"]))
    
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_test, rf_preds))
    
    # Feature Importance from Random Forest
    print("\nFEATURE IMPORTANCE (Random Forest):")
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    for _, row in feature_imp.iterrows():
        print(f" - {row['Feature']:<24}: {row['Importance'] * 100:.2f}%")
        
    return results_df


if __name__ == "__main__":
    run_all_models()