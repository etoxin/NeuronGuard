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
   Fold 1: trained on 170,883, tuned on 56,962, tested on 56,962, threshold 4445.0, detected 72/99 frauds, PR-AUC 66.78%, trained in 2.7083s
   Fold 2: trained on 170,884, tuned on 56,961, tested on 56,962, threshold 3682.0, detected 76/99 frauds, PR-AUC 54.61%, trained in 2.6769s
   Fold 3: trained on 170,885, tuned on 56,961, tested on 56,961, threshold 5226.0, detected 69/98 frauds, PR-AUC 64.98%, trained in 2.6844s
   Fold 4: trained on 170,885, tuned on 56,961, tested on 56,961, threshold 5345.0, detected 76/98 frauds, PR-AUC 66.91%, trained in 2.7192s
   Fold 5: trained on 170,884, tuned on 56,962, tested on 56,961, threshold 4488.0, detected 76/98 frauds, PR-AUC 68.51%, trained in 2.6954s
   Mean training time: 2.6968s (13.4842s total)

4. Aggregate out-of-fold evaluation...
   --- Normal prediction ---
   ➔ Accuracy: 99.92% (284581/284807)

   --- Confusion Matrix ---
      Actual \ Predicted | Legitimate | Fraudulent
      -------------------|------------|-----------
      Legitimate         |     284212 |        103
      Fraudulent         |        123 |        369

   --- Fraud Detection Metrics ---
      Precision: 78.18%
      Recall   : 75.00%
      F1-Score : 76.56%
      Mean PR-AUC: 64.36%
====================================================================
```

### Key Highlights from the Output:
* **All Available Inputs**: Each transaction activates tokens for V1 through V28 plus Amount, using 464 base sensory neurons.
* **Full-Dataset Evaluation**: Every transaction is tested out of fold, and each fold contains 98 or 99 fraud cases.
* **Validation-Tuned Decisions**: Quantile buckets and held-out threshold tuning reached 75.00% recall, detecting 369 of 492 frauds.
* **Normalized Learning**: Smoothed class likelihoods avoid saturating bucket weights and raised F1 from the previous 69.54% to 76.56%.
* **Fast Training**: Each model trained on approximately 170,884 transactions and tuned on roughly 56,962 more in about 2.70 seconds on the measured development machine.

The all-feature model now improves recall and F1 over the earlier five-feature
Hebbian baseline while retaining low false-positive volume. More expressive
models still achieve higher PR-AUC in the separate benchmark.

Exact timings vary by CPU, operating system, toolchain, and background load.
