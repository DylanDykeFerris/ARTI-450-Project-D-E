# Loan Default Prediction Using MLP and Traditional Machine Learning

**ARTI-450: Deep Learning | Ferris State University**\
**Team:** Dylan Dyke and Eddie Henry

## Project Overview

This project aims to predict whether a customer is at risk of defaulting on a loan. We developed a multilayer perceptron (MLP) to learn patterns in customer and predict a binary outcome: Yes (default) or No (no default). We also compared the MLP with Logistic Regression, Random Forest, and a linear Support Vector Machine to see how well each model classifies customers who default.

## Dataset Description and Source

- **Dataset:** Loan Default Prediction Dataset by NIKHIL on Kaggle.
- **Source:** [Kaggle dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)
- **CSV filename used by the notebook:** `Loan_default.csv`
- **Size:** 255,347 records and 18 columns.
- **Target:** `Default`, where `0` means no default and `1` means default.
- **Class distribution:** 225,694 no-default records (88.39%) and 29,653 default records (11.61%).

The dataset includes customer information, financial information, and loan details. Checks found no missing values or duplicate rows. No outliers were flagged in the nine numerical predictors.

### Original Input Features

| Numerical features | Categorical features |
| ------------------ | -------------------- |
| Age                | Education            |
| Income             | EmploymentType       |
| LoanAmount         | MaritalStatus        |
| CreditScore        | HasMortgage          |
| MonthsEmployed     | HasDependents        |
| NumCreditLines     | LoanPurpose          |
| InterestRate       | HasCoSigner          |
| LoanTerm           |                      |
| DTIRatio           |                      |

`LoanID` is excluded because it is a unique ID. `Default` is separated as the target. The 16 original input features become 28 input columns after encoding.

| Dataset                   | Records | Encoded input columns |
| ------------------------- | ------- | --------------------- |
| Original training         | 178,742 | 28                    |
| SMOTENC-balanced training | 315,970 | 28                    |
| Validation                | 38,302  | 28                    |
| Test                      | 38,303  | 28                    |

## Models Implemented

### Multilayer Perceptrons

Two MLPs use the same architecture, with different numbers of input columns:

- **Selected-feature MLP:** 15 inputs.
- **Full-feature MLP:** 28 inputs.
- **Hidden layers:** 64 neurons and 32 neurons, both using ReLU.
- **Output:** One neuron using sigmoid activation.
- **Loss:** Binary cross-entropy.
- **Optimizer:** Adam with a learning rate of 0.001.
- **Batch size:** 256.
- **Maximum epochs:** 50.
- **Early stopping:** Monitor validation loss, wait five epochs without improvement, and restore the best weights.
- **Actual epochs:** 14 for the selected-feature MLP and 30 for the full-feature MLP.
- **Classification threshold:** 0.5.

### Traditional Models

- **Logistic Regression:** L2 regularization, `C=0.1`, and the liblinear solver.
- **Random Forest:** 500 trees, maximum depth 10, minimum split size 10, and minimum leaf size 4.
- **Linear SVM:** `LinearSVC` with `C=0.1` and squared-hinge loss.

These three models use all 28 input columns and the SMOTENC-balanced training data. Their class-weight setting is `balanced`.

## Running the Code

### 1. Prepare a Python Environment

Use a Python environment compatible with TensorFlow. Install the packages used by the notebook:

### 2. Place the Files Together

Put the project's original `.ipynb` notebook and `Loan_default.csv` in the same folder. The CSV filename must match the notebook exactly, including capitalization.

### 3. Open and Run the Notebook

In your Python environment, open the project's `.ipynb` file, select the first cell containing the installed packages, and run from top to bottom.&#x20;



The notebook loads the CSV, performs EDA and preprocessing, trains the models, displays the labeled results, and produces comparison tables. Random Forest training and the 150 cross-validation fits used in tuning may take time. A GPU is not required; the submitted run completed despite a TensorFlow GPU warning.

## Summary of Results

All results below use the same held-out test set of 38,303 original records. Precision, recall, and F1 refer to the default class (`Default=1`).

| Model                     | Input columns | Accuracy | Precision | Recall | F1-score |
| ------------------------- | ------------- | -------- | --------- | ------ | -------- |
| Selected-feature MLP      | 15            | 65.01%   | 20.47%    | 69.76% | 31.65%   |
| Full-feature MLP          | 28            | 67.67%   | 21.59%    | 67.83% | 32.76%   |
| Logistic Regression       | 28            | 69.12%   | 22.74%    | 69.22% | 34.24%   |
| Random Forest             | 28            | 72.05%   | 23.63%    | 63.02% | 34.37%   |
| Linear SVM                | 28            | 68.88%   | 22.61%    | 69.36% | 34.11%   |
| Tuned Logistic Regression | 28            | 68.08%   | 22.34%    | 70.62% | 33.94%   |

### Key Findings

- Random Forest had the highest accuracy, precision, and F1-score among the evaluated models. Its F1 advantage over Logistic Regression was small.
- Tuned Logistic Regression had the highest recall, identifying 3,141 of the 4,448 actual defaults. It also incorrectly flagged 10,919 no-default records.
- The full-feature MLP performed better than the selected-feature MLP on accuracy, precision, and F1. The smaller MLP had higher recall.
- Neither MLP outperformed the traditional models on F1-score.
- Recall-focused Logistic Regression tuning increased recall from 69.22% to 70.62%, while accuracy, precision, and F1 decreased.

## Team Members and Contributions

**Dylan Dyke**
-Model Development
-Model Comparison
-Review of Data Preprocessing and Exploratory Data Analysis
**Eddie Henry**
-Data Preprocessing
-Exploratory Data Analysis
-Review of Model Development and Comparison
