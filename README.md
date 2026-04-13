# Credit Card Transaction Fraud Detection

## Overview
This project applies supervised machine learning algorithms to detect fraudulent credit card transactions using a dataset of 8.5 million transactions.

## Dataset
- **Source:** [Kaggle - Credit Card Transaction Dataset](https://www.kaggle.com/datasets/orogunadebola/credit-card-transaction-dataset-fraud-detection)
- **Total Records:** 8,580,255 transactions
- **Features:** 44 columns
- **Class Distribution:** 98.9% legitimate, 1.1% fraudulent

## Algorithms Used
| Algorithm | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 86.56% | 0.8949 |
| Random Forest | 91.74% | 0.9671 |

## Methodology
- Sampled dataset to 294,806 rows (all fraud + 200k legitimate)
- Dropped irrelevant/PII columns
- Encoded categorical features
- Applied StandardScaler for feature scaling
- Applied SMOTE to handle class imbalance
- Trained and compared two models

## Results
Random Forest outperformed Logistic Regression across all metrics, achieving 91.74% accuracy and AUC-ROC of 0.9671.

## Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn

## How to Run
1. Clone the repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn`
3. Place `data.csv` in the project folder
4. Open `fraud_detection.ipynb` in Jupyter Notebook
Then run:
bashgit add README.md
git commit -m "Add README with project description and results"
git push origin main
