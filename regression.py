"""
Module: Regression Analysis on Marketing Dataset
Predicting Profit based on R&D, Administration, and Marketing expenditures.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Fix terminal encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_and_explore_data(file_path: str = "dataset/Marketing.csv"):
    """Load dataset and print exploratory statistics."""
    df = pd.read_csv(file_path)
    print("=" * 65)
    print("MARKETING DATASET EXPLORATION:")
    print(f"Shape: {df.shape[0]} samples, {df.shape[1]} features")
    print("\nStatistical Summary:")
    print(df.describe().round(2))
    print("=" * 65)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split dataset into train and test partitions."""
    feature_cols = ['RnD', 'Adm', 'Marketing']
    target_col = 'Profit'
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(name: str, y_true, y_pred):
    """Compute regression performance metrics: R2, MAE, RMSE."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "Model": name,
        "R2 Score": round(r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    }


def train_linear_regression(X_train, X_test, y_train, y_test):
    """Linear Regression baseline model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model("Linear Regression", y_test, y_pred), model


def train_regularized_regression(X_train, X_test, y_train, y_test):
    """Ridge and Lasso Regularized Regression."""
    pipe_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
    pipe_ridge.fit(X_train, y_train)
    res_ridge = evaluate_model("Ridge Regression", y_test, pipe_ridge.predict(X_test))

    pipe_lasso = Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(alpha=0.5))])
    pipe_lasso.fit(X_train, y_train)
    res_lasso = evaluate_model("Lasso Regression", y_test, pipe_lasso.predict(X_test))

    return res_ridge, res_lasso


def train_decision_tree(X_train, X_test, y_train, y_test):
    """Decision Tree Regressor with hyperparameter tuning."""
    param_grid = {'max_depth': list(range(2, 10)), 'min_samples_split': [2, 5]}
    grid = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=3, scoring='r2')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    return evaluate_model(f"Decision Tree (depth={best_model.max_depth})", y_test, y_pred), best_model


def train_random_forest(X_train, X_test, y_train, y_test):
    """Random Forest Regressor with grid search."""
    param_grid = {
        'n_estimators': [10, 30, 50, 100],
        'max_depth': [3, 5, 8, None],
        'min_samples_split': [2, 4]
    }
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    return evaluate_model("Random Forest Regressor", y_test, y_pred), best_model


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Gradient Boosting Regressor."""
    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model("Gradient Boosting Regressor", y_test, y_pred), model


def train_knn(X_train, X_test, y_train, y_test):
    """K-Nearest Neighbors Regressor with scaling."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor())
    ])
    param_grid = {'knn__n_neighbors': list(range(2, 8))}
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='r2')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    best_k = grid.best_params_['knn__n_neighbors']
    return evaluate_model(f"KNN Regressor (k={best_k})", y_test, y_pred), grid.best_estimator_


def train_svr(X_train, X_test, y_train, y_test):
    """Support Vector Regressor with standard scaling."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    param_grid = {
        'svr__C': [0.1, 1, 10, 100, 1000],
        'svr__gamma': ['scale', 'auto', 0.01, 0.1],
        'svr__kernel': ['linear', 'rbf']
    }
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='r2')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    return evaluate_model("Support Vector Regressor (SVR)", y_test, y_pred), grid.best_estimator_


def run_all_models():
    """Run and benchmark all regression models."""
    df = load_and_explore_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    results = []
    
    # Train all models
    res_lr, lr_model = train_linear_regression(X_train, X_test, y_train, y_test)
    results.append(res_lr)
    
    res_ridge, res_lasso = train_regularized_regression(X_train, X_test, y_train, y_test)
    results.extend([res_ridge, res_lasso])
    
    res_dt, dt_model = train_decision_tree(X_train, X_test, y_train, y_test)
    results.append(res_dt)
    
    res_rf, rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    results.append(res_rf)
    
    res_gb, gb_model = train_gradient_boosting(X_train, X_test, y_train, y_test)
    results.append(res_gb)
    
    res_knn, knn_model = train_knn(X_train, X_test, y_train, y_test)
    results.append(res_knn)
    
    res_svr, svr_model = train_svr(X_train, X_test, y_train, y_test)
    results.append(res_svr)
    
    # Model comparison table
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    print("\n" + "=" * 65)
    print("REGRESSION MODEL BENCHMARK (MARKETING PROFIT PREDICTION):")
    print(results_df.to_string(index=False))
    print("=" * 65)
    
    # Feature weights and importance
    print("\nLINEAR REGRESSION COEFFICIENTS (Feature Weights):")
    for feat, coef in zip(['RnD', 'Adm', 'Marketing'], lr_model.coef_):
        print(f" - {feat:<12}: {coef:.4f}")
    print(f" - Intercept   : {lr_model.intercept_:.4f}")
    
    print("\nRANDOM FOREST FEATURE IMPORTANCE:")
    for feat, imp in zip(['RnD', 'Adm', 'Marketing'], rf_model.feature_importances_):
        print(f" - {feat:<12}: {imp * 100:.2f}%")
        
    return results_df


if __name__ == "__main__":
    run_all_models()