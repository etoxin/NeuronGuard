# NeuronGuard: High-Frequency Financial Fraud Scanner

This example showcases a real-world financial fraud scanner built using the `neuronguard` Python library. It trains and evaluates on the complete Kaggle Credit Card Fraud Detection dataset.

---

## Datasets & Features

### Credit Card Fraud Detection Dataset
The scanner trains and evaluates on the complete Kaggle Credit Card Fraud Detection dataset containing **284,807 real transactions**.

* **Features `V1` through `V28`**: These are principal components obtained using **Principal Component Analysis (PCA)**. Due to confidentiality and privacy constraints, the original raw features and background personal information are not provided. PCA transforms the original high-dimensional features into orthogonal, uncorrelated components while preserving most of the variance.
* **`Amount`**: The transaction amount.
* **`Class`**: The transaction label (`0` for legitimate, `1` for fraudulent).

---

## How to Run

This project uses [mise](https://mise.jdx.dev/) and [uv](https://github.com/astral-sh/uv) to manage toolchains and tasks.

```bash
# Run these commands from this example directory.

# 1. Download the credit card fraud dataset
mise run examples:fraud_scanner:download_data

# 2. Run the Fraud Scanner
mise run examples:fraud_scanner:run

# Optional: train each fold for 10 passes instead of one
mise run examples:fraud_scanner:run -- --epochs 10

# Optional: average the original prediction with 10 slightly altered inputs
mise run examples:fraud_scanner:run -- --jitter-samples 10 --jitter-fraction 0.05
```

Increasing the epoch count repeats training over each fold. It is exposed as an
experimental control rather than enabled by default because repeated updates can
saturate weights and do not add new information.

The jitter mode perturbs every feature by up to the selected fraction of its
bucket width, runs each altered transaction through the same model, and averages
the raw class scores with the original prediction. It is available for
experimentation but is not enabled by default.

## Evaluation methodology

The example uses deterministic five-fold stratified cross-validation. Fraud and
legitimate records are shuffled with a fixed seed and distributed independently,
so every fold contains 98 or 99 fraud cases. For each run, three folds train the
model, one fold tunes its binary F1 decision threshold, and one fold tests it.
Every transaction is evaluated once by a model that neither trained nor tuned on
that transaction. All 29 features use 16 exact quantile buckets calculated from
the training folds only.

In addition to accuracy, precision, recall, and F1, the example reports mean
precision-recall area under the curve (PR-AUC) across the five folds. PR-AUC is
particularly useful here because only 0.173% of the transactions are fraudulent
and overall accuracy is therefore dominated by the legitimate class.

---

## Example Output

Here is the actual output of the fraud scanner running on **284,807 real-world transactions**:

```text
====================================================================
💳 NeuronGuard Real-World Financial Fraud Scanner 💳
====================================================================

1. Loading the creditcard dataset...
   Total Transactions: 284,807
   Fraud Cases       : 492 (0.173%)

2. Building 5 deterministic stratified folds...
   Fraud cases per fold: [99, 99, 98, 98, 98]

3. Running stratified cross-validation (1 training epoch)...
   Fold 1: trained on 170,883, tuned on 56,962, tested on 56,962, threshold -823581.0, detected 59/99 frauds, PR-AUC 57.66%, trained in 2.1751s
   Fold 2: trained on 170,884, tuned on 56,961, tested on 56,962, threshold -830169.0, detected 63/99 frauds, PR-AUC 51.13%, trained in 2.1550s
   Fold 3: trained on 170,885, tuned on 56,961, tested on 56,961, threshold -797180.0, detected 53/98 frauds, PR-AUC 57.69%, trained in 2.1641s
   Fold 4: trained on 170,885, tuned on 56,961, tested on 56,961, threshold -827212.0, detected 69/98 frauds, PR-AUC 62.59%, trained in 2.1311s
   Fold 5: trained on 170,884, tuned on 56,962, tested on 56,961, threshold -826266.0, detected 63/98 frauds, PR-AUC 65.11%, trained in 2.1421s
   Mean training time: 2.1535s (10.7673s total)

4. Aggregate out-of-fold evaluation...
   --- Normal prediction ---
   ➔ Accuracy: 99.91% (284538/284807)

   --- Confusion Matrix ---
      Actual \ Predicted | Legitimate | Fraudulent
      -------------------|------------|-----------
      Legitimate         |     284231 |         84
      Fraudulent         |        185 |        307

   --- Fraud Detection Metrics ---
      Precision: 78.52%
      Recall   : 62.40%
      F1-Score : 69.54%
      Mean PR-AUC: 58.84%
====================================================================
```

### Key Highlights from the Output:
* **All Available Inputs**: Each transaction activates tokens for V1 through V28 plus Amount, using 464 base sensory neurons.
* **Full-Dataset Evaluation**: Every transaction is tested out of fold, and each fold contains 98 or 99 fraud cases.
* **Validation-Tuned Decisions**: Quantile buckets and held-out threshold tuning raised recall to 62.40%, detecting 307 of 492 frauds.
* **Fast Training**: Each model trained on approximately 170,884 transactions and tuned on roughly 56,962 more in about 2.15 seconds on the measured development machine.

The all-feature model improves recall over the earlier five-feature baseline but
has lower precision-recall ranking quality. Feature selection or weighting
remains an important next step.

Exact timings vary by CPU, operating system, toolchain, and background load.
