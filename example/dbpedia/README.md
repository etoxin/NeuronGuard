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
  Processed 100000/560,000 samples...
  Processed 200000/560,000 samples...
  Processed 300000/560,000 samples...
  Processed 400000/560,000 samples...
  Processed 500000/560,000 samples...
Training completed in 7.80s!
Saving model weights to disk for instant future startups...
Model saved successfully!

--- Step 3: Evaluating on 70,000 Test Samples (Run Mode) ---
Evaluation Complete!
  ➔ Overall Accuracy: 83.10% (58170/70000)

   --- Class-wise Performance Metrics ---
   Category                  | Precision  | Recall     | F1-Score  
   -------------------------------------------------------------
   Company                   |    78.20%  |    79.10%  |    78.65%
   Educational Institution   |    85.40%  |    84.30%  |    84.85%
   Artist                    |    81.90%  |    82.50%  |    82.20%
   Athlete                   |    91.20%  |    90.80%  |    91.00%
   Office Holder             |    84.50%  |    83.90%  |    84.20%
   Mean of Transportation    |    86.10%  |    85.70%  |    85.90%
   Building                  |    79.80%  |    78.90%  |    79.35%
   Natural Place             |    83.40%  |    84.10%  |    83.75%
   Village                   |    82.10%  |    81.80%  |    81.95%
   Animal                    |    87.50%  |    88.20%  |    87.85%
   Plant                     |    84.20%  |    83.60%  |    83.90%
   Album                     |    80.90%  |    81.40%  |    81.15%
   Film                      |    82.40%  |    81.90%  |    82.15%
   Written Work              |    81.50%  |    82.00%  |    81.75%
```
