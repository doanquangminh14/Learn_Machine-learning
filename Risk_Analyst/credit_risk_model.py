"""
Module: Credit Default Prediction & Risk Scoring Pipeline (credit_risk_model.py)
Training ML models to predict loan default probability based on customer credit profiles.
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)


def load_and_preprocess_credit_data(file_path: str = "dataset/Credit Risk Dataset.xlsx"):
    """Load dataset, filter invalid records, and handle missing values."""
    print("=" * 65)
    print("LOADING & PREPROCESSING CREDIT RISK DATASET...")
    df = pd.read_excel(file_path)
    
    # Filter biologically impossible anomalies
    initial_len = len(df)
    df = df[(df['person_age'] <= 100) & (df['person_emp_length'] <= 60)].copy()
    print(f"Filtered out {initial_len - len(df)} anomalous records. Current records: {len(df)}")
    
    # Impute missing loan interest rate by grade median
    if 'loan_int_rate' in df.columns:
        df['loan_int_rate'] = df.groupby('loan_grade')['loan_int_rate'].transform(
            lambda x: x.fillna(x.median())
        )
        
    # Impute missing employment length with median
    if 'person_emp_length' in df.columns:
        df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
        
    # Feature Engineering
    df['total_debt'] = df['loan_amnt'] + df['other_debt']
    df['monthly_income'] = (df['person_income'] / 12.0) + 1
    df['monthly_loan_installment'] = (df['loan_amnt'] * (1 + (df['loan_int_rate'] / 100))) / df['loan_term_months']
    df['installment_to_income_ratio'] = df['monthly_loan_installment'] / df['monthly_income']
    
    print("=" * 65)
    return df


def prepare_features(df: pd.DataFrame):
    """Separate target and feature pipelines (Numeric + Categorical)."""
    target_col = 'loan_status'
    
    drop_cols = ['client_ID', 'loan_status', 'city', 'state', 'country', 'city_latitude', 'city_longitude']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=existing_drop_cols)
    y = df[target_col]
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    return X, y, preprocessor, numeric_cols, categorical_cols


def evaluate_model(name: str, model, X_test, y_test):
    """Compute performance metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    }


def train_and_benchmark(file_path: str = "dataset/Credit Risk Dataset.xlsx"):
    """Train multiple classifiers for credit default prediction."""
    df = load_and_preprocess_credit_data(file_path)
    X, y, preprocessor, num_cols, cat_cols = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    print(f"Default rate: {y.mean() * 100:.2f}%\n")
    
    models = {
        "Weighted Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    benchmarks = []
    fitted_pipelines = {}
    
    for name, clf in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(name, pipe, X_test, y_test)
        benchmarks.append(metrics)
        fitted_pipelines[name] = pipe
        
    benchmark_df = pd.DataFrame(benchmarks).sort_values(by="ROC-AUC", ascending=False)
    print("=" * 65)
    print("CREDIT DEFAULT RISK MODEL BENCHMARK:")
    print(benchmark_df.to_string(index=False))
    print("=" * 65)
    
    # Detailed report for the top model
    best_name = benchmark_df.iloc[0]["Model"]
    best_pipe = fitted_pipelines[best_name]
    y_pred_best = best_pipe.predict(X_test)
    
    print(f"\nDETAILED CLASSIFICATION REPORT ({best_name}):")
    print(classification_report(y_test, y_pred_best, target_names=["Non-Default (0)", "Default (1)"]))
    
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred_best))
    
    return benchmark_df, best_pipe


if __name__ == "__main__":
    train_and_benchmark()
