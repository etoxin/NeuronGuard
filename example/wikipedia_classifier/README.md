# NeuronGuard: Wikipedia Structured Dataset Classifier & Router

This example demonstrates how to stream and classify the massive **44.4 GB** `wikimedia/structured-wikipedia` dataset from Hugging Face on the fly with **zero disk storage overhead**, using `neuronguard`'s high-performance neuromorphic event engine.

---

## What This Example Does

1. **Streaming Dataset Ingestion**: Streams the `wikimedia/structured-wikipedia` dataset in real-time from Hugging Face using the `datasets` library with `streaming=True`. This allows training on a 44.4 GB dataset without downloading it to disk.
2. **Vocabulary Building**: Scans the first 20,000 articles to build a vocabulary of the top 5,000 most frequent words.
3. **On-The-Fly Training**: Streams the next 30,000 articles and trains a `NeuronGuardField` on the fly using the **Guard/Lease transactional pattern** to classify articles into 5 high-level domains:
   - **Science & Technology**
   - **Geography & Places**
   - **Biography & People**
   - **History & Events**
   - **Arts & Culture**
4. **Interactive CLI Router**: Launches an interactive command-line interface where you can type any sentence, and the engine routes it to the 5 specialized domain experts in **microseconds**.

---

## How to Run

You can run this example cleanly using `mise`:

```bash
mise run wikipedia_classifier
```

---

## Sample CLI Output

```text
👉 Enter text: Albert Einstein was a theoretical physicist who developed the theory of relativity.
  Expert Activations:
    [Science & Technology ]:  20
    [Geography & Places   ]:   0
    [Biography & People   ]:  10
    [History & Events     ]:   0
    [Arts & Culture       ]:   0

  🏆 Winning Category: **SCIENCE & TECHNOLOGY** 🏆
```
