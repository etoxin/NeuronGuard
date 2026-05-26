# NeuronGuard: AG News Classifier

This example demonstrates how to train and evaluate a `neuronguard` spiking neural network (SNN) on the **AG News dataset** (120,000 training samples, 7,600 test samples, 4 classes).

---

## How to Run

```bash
# 1. Download the AG News dataset
mise run download_data

# 2. Run the AG News Classifier
mise run ag_news
```

---

## Performance & Accuracy

```text
====================================================================
📰 AG News 120,000 Dataset Classification PoC (Python) 📰
====================================================================

--- Step 1: Building Vocabulary from 120,000 Training Samples ---
Vocabulary built successfully!
  Top 1,000 most frequent words selected.
  Total Neuron Field Size: 1004 neurons (64 bytes each)

--- Step 2: Training on 120,000 Samples (Trainer Mode) ---
Applying the Guard feedback loop over the entire dataset...
  Processed 30000/120,000 samples...
  Processed 60000/120,000 samples...
  Processed 90000/120,000 samples...
  Processed 120000/120,000 samples...
Training completed in 1.28s!

--- Step 3: Evaluating on 7,600 Test Samples (Run Mode) ---
Evaluation Complete!
  Accuracy: 82.14% (6243/7600)

--- Confusion Matrix ---
  Actual  Predicted | World | Sports | Business | Sci/Tech
  -------------------|-------|--------|----------|---------
  World News         |  1638 |    101 |      124 |       37
  Sports             |    72 |   1743 |       51 |       34
  Business           |   137 |     75 |     1496 |      192
  Sci/Tech           |   126 |    141 |      267 |     1366
====================================================================
```
