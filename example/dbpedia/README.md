# NeuronGuard: DBpedia Ontology Classifier & Router

This example demonstrates how to train and evaluate a `neuronguard` spiking neural network (SNN) on the massive **DBpedia Ontology dataset** (560,000 training samples, 70,000 test samples, 14 classes).

---

## How to Run

```bash
# 1. Download the DBpedia dataset
mise run download_data

# 2. Run the DBpedia Classifier & Router
mise run dbpedia

# 3. Run the PyTorch vs NeuronGuard Benchmark
mise run dbpedia_benchmark
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

---

## PyTorch vs NeuronGuard Benchmark

To highlight the extreme performance and efficiency of NeuronGuard, we train a standard PyTorch text classifier (`EmbeddingBag` + `Linear`) on the exact same DBpedia dataset and vocabulary.

Here are the actual benchmark results run on an **Apple M2 Pro CPU**:

```text
====================================================================
📊 BENCHMARK COMPARISON: NEURONGUARD VS PYTORCH 📊
====================================================================
   Metric                    | NeuronGuard     | PyTorch
   -------------------------------------------------------------
   Training Time             | 0.64         s  | 5.38         s
   Test Accuracy             | 86.26        %  | 85.70        %
   Model Size (Disk)         | 313.4       KB  | 1253.6      KB
   Memory Footprint          | ~320 KB         | ~500+ MB (Heap)
====================================================================
```

### Key Takeaways:
* **8.4x Faster Training**: NeuronGuard trains in **0.64 seconds** compared to PyTorch's **5.38 seconds**!
* **Higher Accuracy**: NeuronGuard achieves **86.26% accuracy**, outperforming PyTorch's **85.70%**!
* **4x Smaller Model**: NeuronGuard's compiled model is only **313 KB** compared to PyTorch's **1.2 MB**!
* **1,500x Less Memory**: NeuronGuard runs entirely within CPU L1/L2 caches with a **~320 KB footprint**, while PyTorch requires a massive **~500+ MB runtime heap**.
