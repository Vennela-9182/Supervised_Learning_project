# Customer Churn Prediction – Supervised ML (Tree Methods)

Predicting whether a telecom customer will churn using tree-based ensemble classifiers. The goal is to identify at-risk customers early so the business can take retention action before losing them.

---

## Dataset

**Telco Customer Churn** — a real-world telecom dataset containing customer demographics, account info, and service usage.

| Detail | Value |
|---|---|
| File | `Telco-Customer-Churn.csv` |
| Rows | ~7,043 customers |
| Target Column | `Churn` (Yes / No) |
| Class Imbalance | ~26% Yes (churners) vs ~74% No |

---

## Problem Statement

Customer churn is costly — acquiring a new customer is far more expensive than retaining an existing one. This project builds a classification model to predict churn based on customer features like contract type, monthly charges, tenure, and add-on services.

Since the dataset is **imbalanced** (more non-churners than churners), **F1-score** is used as the primary evaluation metric rather than accuracy.

---

## Exploratory Data Analysis

Key findings from EDA:

- Customers on **month-to-month contracts** churn at significantly higher rates
- **Short-tenure customers (0–12 months)** have the highest churn rate (~62% at 1 month)
- Higher **monthly charges** correlate with increased churn
- Customers with **no online security or tech support** are more likely to churn
- **Paperless billing** and **electronic check payment** show positive correlation with churn

Tenure cohorts created for deeper analysis:
- `0–12 Months` | `12–24 Months` | `24–48 Months` | `Over 48 Months`

---

## Modeling Approach

**Train/Test Split:** 90% train, 10% test (`random_state=101`)

**Models Trained:**

| Model | Notes |
|---|---|
| Decision Tree | `max_depth=6` to prevent overfitting |
| Random Forest | `n_estimators=100` |
| AdaBoost | Default hyperparameters |
| Gradient Boosting | Available for further tuning |

All models trained on the **same train/test split** for fair comparison.

---

## Results

**Primary metric: F1-score on the Yes (Churn) class** — chosen because of class imbalance and business cost of missing churners.

| Model | Yes-Class F1 |
|---|---|
| Decision Tree | Baseline |
| Random Forest | +improvement over DT |
| **AdaBoost** | **Best — highest minority-class F1** |

> AdaBoost outperformed both Decision Tree and Random Forest on the churner class, making it the chosen model for this task.

**Why F1 over Accuracy?**
The dataset has ~74% non-churners. A model that predicts "No" for everyone would get 74% accuracy while being completely useless. F1-score penalizes this by balancing precision and recall for the minority class.

---

## Tech Stack

| Tool | Usage |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | EDA visualizations |
| scikit-learn | ML models, metrics, train/test split |

---

## Project Structure

```
├── Telco-Customer-Churn.csv          # Dataset
├── churn_prediction.ipynb            # Main notebook (EDA + Modeling)
└── README.md                         # This file
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/Vennela-9182/Supervised_Learning_project
cd Supervised_Learning_project

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Open the notebook
jupyter notebook churn_prediction.ipynb
```

---

## Key Takeaways

- Ensemble methods (AdaBoost, Random Forest) consistently outperform a single Decision Tree on imbalanced data
- Contract type and tenure are the strongest predictors of churn
- For churn problems, optimizing for the **minority class F1** matters more than overall accuracy
- Future improvement: apply **SMOTE oversampling** or **class_weight='balanced'** to further boost recall on churners
