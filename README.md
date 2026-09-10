# Machine Learning Projects Portfolio

A comprehensive, production-grade Machine Learning repository covering end-to-end workflows across five major real-world datasets and domains: **Regression**, **Binary Classification**, **Market Basket Association Rules**, **Customer Segmentation (Clustering)**, and **Credit Default Risk Analysis**.

---

## 📌 Repository Structure

```text
Machine-learning-projects/
├── dataset/
│   ├── Marketing.csv                # 44 rows x 5 cols (Marketing spend vs Profit)
│   ├── Can_nag.csv                  # 768 rows x 9 cols (Pima Diabetes Diagnostic)
│   ├── assio.csv                    # 22 transactions (Retail basket items)
│   ├── data.csv                     # 2,205 rows x 39 cols (Customer Personality)
│   └── Credit Risk Dataset.xlsx     # 32,581 rows x 29 cols (Banking Credit Risk)
├── Risk_Analyst/
│   ├── clean_data.ipynb             # In-depth EDA, cleaning & feature engineering
│   └── credit_risk_model.py         # End-to-end credit default prediction pipeline
├── reports/
│   └── project_summary.json         # Structured benchmark metadata & metrics
├── regression.py                    # Marketing profit prediction models
├── classification.py                # Diabetes classification pipeline
├── association_rules.py             # Apriori frequent itemset & rule mining
├── clustering.py                    # Modular K-Means customer segmentation
├── kmean.ipynb                      # Jupyter notebook for customer clustering & PCA
├── visualize_insights.py            # Project-wide metrics dashboard & summaries
├── .gitignore                       # Clean environment rules
└── README.md                        # Documentation and usage guide
```

---

## 📊 Dataset Overviews & Problem Formulations

### 1. Marketing Profit Prediction (`Marketing.csv`)
- **Domain**: Marketing & Financial Analytics
- **Task**: Supervised Regression
- **Objective**: Predict company profit based on R&D spend (`RnD`), Administration cost (`Adm`), and Marketing spend (`Marketing`).
- **Top Performer**: **Ridge Regression** ($R^2 = 0.9218$, $MAE = 7.04$, $RMSE = 8.18$).
- **Key Insight**: R&D spending contributes over **90.86%** of the variance in corporate profitability.

### 2. Diabetes Medical Diagnosis (`Can_nag.csv`)
- **Domain**: Healthcare & Clinical Diagnostics
- **Task**: Supervised Binary Classification (0: Healthy, 1: Diabetic)
- **Objective**: Predict diabetes risk from physiological measurements (Glucose, Blood Pressure, BMI, Insulin, Age).
- **Top Performer**: **KNN Classifier ($k=15$)** ($F1\text{-Score} = 0.6000$, $\text{ROC-AUC} = 0.8068$, $\text{Accuracy} = 74.03\%$).
- **Key Insight**: Fasting Glucose concentration ($27.64\%$) and BMI ($17.90\%$) are the two strongest predictors.

### 3. Market Basket Analysis (`assio.csv`)
- **Domain**: Supermarket & E-commerce Retail
- **Task**: Unsupervised Association Rule Mining (Apriori Algorithm)
- **Objective**: Discover cross-selling product combinations from customer transactions.
- **Top Rule**: Purchasing `[Chips, Bread]` increases the probability of buying `[Butter, Apple]` by **1.55x** (Confidence: $77.8\%$).

### 4. Customer Personality Segmentation (`data.csv`)
- **Domain**: Customer Relationship Management (CRM)
- **Task**: Unsupervised Clustering (K-Means with $K=3$)
- **Objective**: Group 2,205 consumers by purchasing habits, income, web visits, and recency.
- **Customer Personas**:
  - **Cluster 0 (Value Shoppers)**: Balanced spending, moderate income, multichannel buyers.
  - **Cluster 1 (High-Net-Worth VIPs)**: Highest income, heavy spenders in Wines & Meat, low web traffic.
  - **Cluster 2 (Budget-Conscious Web Browsers)**: Lower disposable income, high web store visits, sensitive to deals.

### 5. Credit Default Risk Scoring (`Credit Risk Dataset.xlsx`)
- **Domain**: Banking & FinTech Risk Underwriting
- **Task**: Supervised Classification for Default Prediction (`loan_status`)
- **Objective**: Predict default probabilities on 31,600+ loans using income, interest rate, employment length, loan grade, and historical delinquencies.
- **Top Performer**: **Gradient Boosting Classifier** ($\text{ROC-AUC} = 0.9423$, $\text{Accuracy} = 93.77\%$, $\text{Precision} = 96.63\%$).

---

## 🚀 Model Benchmark Summary

| Module / Problem | Algorithms Evaluated | Primary Metric | Secondary Metric | Best Algorithm |
|:---|:---|:---:|:---:|:---|
| **Marketing Regression** | Linear, Ridge, Lasso, DT, RF, SVR, KNN, GBDT | $R^2: \mathbf{0.9218}$ | $MAE: 7.04$ | Ridge Regression |
| **Diabetes Classification** | LogReg, DT, RF, GBDT, KNN, SVM, Naive Bayes | $F1: \mathbf{0.6000}$ | $\text{ROC-AUC}: 0.8068$ | KNN Classifier ($k=15$) |
| **Market Basket Rules** | Apriori Mining (Support $\ge 0.3$, Conf $\ge 0.5$) | $\text{Lift}: \mathbf{1.5556}$ | $\text{Conf}: 86.67\%$ | Apriori Association Rules |
| **Customer Segmentation** | K-Means ($K=2$ to $7$), Silhouette, PCA | $\text{Silhouette}: \mathbf{0.2555}$ | $CH: 867.08$ | K-Means ($K=3$) |
| **Credit Risk Scoring** | Balanced LogReg, Random Forest, GBDT | $\text{ROC-AUC}: \mathbf{0.9423}$ | $\text{Accuracy}: 93.77\%$ | Gradient Boosting Classifier |

---

## 🛠️ Quick Start & Execution

### 1. Requirements
Ensure Python 3.9+ is installed along with standard scientific packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn apyori openpyxl
```

### 2. Running Individual Modules

```bash
# 1. Marketing Profit Regression
python regression.py

# 2. Diabetes Classification Pipeline
python classification.py

# 3. Market Basket Apriori Rules
python association_rules.py

# 4. Customer Segmentation K-Means
python clustering.py

# 5. Credit Default Risk Pipeline
python Risk_Analyst/credit_risk_model.py

# 6. Overall Analytics Dashboard
python visualize_insights.py
```