# NeuronGuard: DBpedia Ontology Classifier & Router

This example demonstrates how to train and evaluate a `neuronguard` spiking neural network (SNN) on the massive **DBpedia Ontology dataset** (560,000 training samples, 70,000 test samples, 14 classes).

---

## How to Run

```bash
# 1. Download the DBpedia dataset
mise run download_data

# 2. Run the DBpedia Classifier & Router
mise run dbpedia
```

---

## Performance & Accuracy

```text
====================================================================
📚 DBpedia Ontology 560,000 Classifier & Router (Python) 📚
====================================================================

Pre-trained model not found. Starting training on 560,000 samples...
(This will take about 10 seconds and will save the weights for instant future startups)

--- Step 1: Building Vocabulary from 560,000 Training Samples ---

--- Step 2: Training on 560,000 Samples (Trainer Mode) ---
  Loaded 559,969 training records.
  Processed 100000/560,000 samples...
  Processed 200000/560,000 samples...
  Processed 300000/560,000 samples...
  Processed 400000/560,000 samples...
  Processed 500000/560,000 samples...
Training completed in 0.64s!
Saving model weights to disk for instant future startups...
Model saved successfully!

--- Step 3: Evaluating on 70,000 Test Samples (Run Mode) ---
Evaluation Complete!
  ➔ Overall Accuracy: 86.26% (60380/70000)

   --- Class-wise Performance Metrics ---
   Category                  | Precision  | Recall     | F1-Score  
   -------------------------------------------------------------
   Company                   |    94.52%  |    70.66%  |    80.87%
   Educational Institution   |    82.91%  |    97.24%  |    89.51%
   Artist                    |    93.23%  |    56.22%  |    70.14%
   Athlete                   |    93.43%  |    92.08%  |    92.75%
   Office Holder             |    88.14%  |    87.40%  |    87.77%
   Mean of Transportation    |    83.11%  |    92.32%  |    87.47%
   Building                  |    89.54%  |    88.72%  |    89.13%
   Natural Place             |    95.71%  |    88.28%  |    91.84%
   Village                   |    70.50%  |    98.76%  |    82.27%
   Animal                    |    92.99%  |    68.14%  |    78.65%
   Plant                     |    76.07%  |    93.86%  |    84.04%
   Album                     |    82.69%  |    98.14%  |    89.76%
   Film                      |    95.13%  |    91.78%  |    93.42%
   Written Work              |    87.83%  |    84.00%  |    85.87%
```
