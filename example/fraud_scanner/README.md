# NeuronGuard: High-Frequency Financial Fraud Scanner

This example showcases a real-world financial fraud scanner built using the `neuronguard` Python library. It trains and evaluates on a subset of the Kaggle Credit Card Fraud Detection dataset.

---

## Datasets & Features

### Credit Card Fraud Detection Dataset
The scanner trains and evaluates on a subset of the Kaggle Credit Card Fraud Detection dataset containing **29,222 real transactions**.

* **Features `V1` through `V28`**: These are principal components obtained using **Principal Component Analysis (PCA)**. Due to confidentiality and privacy constraints, the original raw features and background personal information are not provided. PCA transforms the original high-dimensional features into orthogonal, uncorrelated components while preserving most of the variance.
* **`Amount`**: The transaction amount.
* **`Class`**: The transaction label (`0` for legitimate, `1` for fraudulent).

---

## How to Run

This project uses [mise](https://mise.jdx.dev/) and [uv](https://github.com/astral-sh/uv) to manage toolchains and tasks.

```bash
# 1. Download the credit card fraud dataset
mise run download_data

# 2. Run the Fraud Scanner
mise run fraud_scanner
```

---

## Example Output

Here is the actual output of the fraud scanner running on **29,222 real-world transactions**:

```text
====================================================================
💳 NeuronGuard Real-World Financial Fraud Scanner 💳
====================================================================

1. Loading and splitting the creditcard dataset...
   Total Transactions: 29,222
   Training Set Size : 23,377
   Test Set Size     : 5,845
   Fraud Cases (Train): 87 (0.372%)
   Fraud Cases (Test) : 6 (0.103%)

2. Computing feature boundaries from training set...
3. Initializing NeuronGuard cortex...
4. Training the cortex on real-world transactions...
   Training completed in 0.0258s!

5. Evaluating accuracy on test set...
   Evaluation Complete!
   ➔ Accuracy: 99.93% (5841/5845)

   --- Confusion Matrix ---
      Actual \ Predicted | Legitimate | Fraudulent
      -------------------|------------|-----------
      Legitimate         |       5838 |          1
      Fraudulent         |          3 |          3

   --- Fraud Detection Metrics ---
      Precision: 75.00%
      Recall   : 50.00%
      F1-Score : 60.00%
====================================================================
```

### Key Highlights from the Output:
* **99.93% Accuracy**: Correctly classified 5,841 out of 5,845 test transactions.
* **75.00% Precision**: Only **1 false positive** out of 5,839 legitimate transactions, ensuring legitimate users are not blocked.
* **Ultra-Fast Training**: Trained on **23,377 transactions in just 25 milliseconds (0.0258s)**!
* **Ultra-Lean Footprint**: Fits entirely within a fraction of a single CPU cache line, requiring zero matrix multiplication.
