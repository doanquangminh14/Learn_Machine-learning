"""
Module: Cross-Project Evaluation & Analytics Dashboard (visualize_insights.py)
Summarizes data statistics, model performance benchmarks, and core insights across all datasets.
"""

import sys
import os
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def generate_dataset_inventory():
    """Summarize all datasets in the repository."""
    inventory = [
        {
            "Dataset": "Marketing.csv",
            "Domain": "Marketing Analytics",
            "Task": "Regression (Profit Prediction)",
            "Samples": 44,
            "Features": 5,
            "Target": "Profit"
        },
        {
            "Dataset": "Can_nag.csv",
            "Domain": "Healthcare Diagnostics",
            "Task": "Binary Classification (Diabetes)",
            "Samples": 768,
            "Features": 9,
            "Target": "Class (0/1)"
        },
        {
            "Dataset": "assio.csv",
            "Domain": "Retail / Supermarket",
            "Task": "Association Rule Mining (Apriori)",
            "Samples": 22,
            "Features": 6,
            "Target": "Frequent Itemsets"
        },
        {
            "Dataset": "data.csv",
            "Domain": "Customer Relationship (CRM)",
            "Task": "Clustering (Customer Segmentation)",
            "Samples": 2205,
            "Features": 39,
            "Target": "Unsupervised (K-Means K=3)"
        },
        {
            "Dataset": "Credit Risk Dataset.xlsx",
            "Domain": "Banking & FinTech Risk",
            "Task": "Risk Scoring & Default Prediction",
            "Samples": 32581,
            "Features": 29,
            "Target": "loan_status (0/1)"
        }
    ]
    return pd.DataFrame(inventory)


def generate_model_leaderboard():
    """Benchmark results of the top performing algorithms for each problem type."""
    leaderboard = [
        {
            "Domain": "Marketing Profit",
            "Best Model": "Ridge Regression",
            "Primary Metric": "R2 Score: 0.9218",
            "Secondary Metric": "MAE: 7.04, RMSE: 8.18",
            "Key Driver": "RnD Spending (90.86% weight)"
        },
        {
            "Domain": "Diabetes Diagnosis",
            "Best Model": "KNN Classifier (k=15)",
            "Primary Metric": "F1-Score: 0.6000",
            "Secondary Metric": "ROC-AUC: 0.8068, Acc: 74.03%",
            "Key Driver": "Glucose Concentration (27.64%)"
        },
        {
            "Domain": "Market Basket",
            "Best Model": "Apriori Rule Miner",
            "Primary Metric": "Max Lift: 1.5556",
            "Secondary Metric": "Max Confidence: 86.67%",
            "Key Driver": "[Chips, Bread] -> [Butter, Apple]"
        },
        {
            "Domain": "Customer Segments",
            "Best Model": "K-Means (K=3)",
            "Primary Metric": "Silhouette: 0.2555",
            "Secondary Metric": "Calinski-Harabasz: 867.08",
            "Key Driver": "Income, Wines & Meat Spending"
        },
        {
            "Domain": "Credit Default Risk",
            "Best Model": "Gradient Boosting",
            "Primary Metric": "ROC-AUC: 0.9423",
            "Secondary Metric": "Accuracy: 93.77%, Precision: 96.63%",
            "Key Driver": "Loan Interest Rate, Grade, Debt-to-Income"
        }
    ]
    return pd.DataFrame(leaderboard)


def print_dashboard():
    """Print complete summary dashboard."""
    print("=" * 75)
    print("       MACHINE LEARNING PROJECTS - REPOSITORY ANALYTICS DASHBOARD       ")
    print("=" * 75)
    
    inventory_df = generate_dataset_inventory()
    print("\n1. DATASET INVENTORY:")
    print(inventory_df.to_string(index=False))
    
    leaderboard_df = generate_model_leaderboard()
    print("\n" + "=" * 75)
    print("2. TOP PERFORMING MODELS & KEY BUSINESS DRIVERS:")
    print(leaderboard_df.to_string(index=False))
    print("=" * 75)
    
    # Export summary to JSON
    os.makedirs("reports", exist_ok=True)
    summary_path = os.path.join("reports", "project_summary.json")
    summary_data = {
        "datasets": inventory_df.to_dict(orient="records"),
        "leaderboard": leaderboard_df.to_dict(orient="records")
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved structured analytics report to: {summary_path}")


if __name__ == "__main__":
    print_dashboard()
