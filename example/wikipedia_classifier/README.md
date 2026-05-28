# NeuronGuard: Wikipedia Structured Dataset Classifier & Router

This example demonstrates how to stream and classify the massive **44.4 GB** `wikimedia/structured-wikipedia` dataset from Hugging Face on the fly with **zero disk storage overhead**, using `neuronguard`'s high-performance neuromorphic event engine.

---

## What This Example Does

1. **Streaming Dataset Ingestion**: Streams the `wikimedia/structured-wikipedia` dataset in real-time from Hugging Face using the `datasets` library with `streaming=True`. This allows training on a 44.4 GB dataset without downloading it to disk.
2. **Vocabulary Building**: Scans the first 20,000 articles to build a vocabulary of the top 5,000 most frequent words.
3. **On-The-Fly Training**: Streams the next 100,000 articles and trains a `NeuronGuardField` on the fly using the **Guard/Lease transactional pattern** to classify articles into 5 high-level domains:
   - **Science & Technology**
   - **Geography & Places**
   - **Biography & People**
   - **History & Events**
   - **Arts & Culture**
4. **Batch Evaluation**: Evaluates the trained model on the next 10,000 streamed articles to calculate and print the **overall classification accuracy** and evaluation speed (samples/sec).
5. **Automated Live Routing Demonstration**: Evaluates the trained model on 5 pre-defined, high-quality Wikipedia-style sentences (one for each category), printing the routing results and exiting cleanly.

---

## How to Run

You can run this example cleanly using `mise`:

```bash
# Run with default settings (100,000 training samples, 10,000 test samples, 20,000 vocab samples)
mise run wikipedia_classifier
```

### Customizing Training Scale
Since the dataset is streamed, you can easily scale training up or down using command-line arguments:

```bash
# Train on 500,000 articles, evaluate on 50,000, with a 10,000 word vocabulary
uv run --no-project python example/wikipedia_classifier/main.py --train-samples 500000 --test-samples 50000 --vocab-size 10000

# Force retraining even if weights already exist
uv run --no-project python example/wikipedia_classifier/main.py --force-retrain
```

---

## Sample Output

```text
--- Step 4: Evaluating Accuracy on Test Split ---
Streaming the next 10,000 articles for evaluation...
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 86/86 [00:00<00:00, 20055.05it/s]
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 86/86 [00:00<00:00, 20397.54it/s]
Skipping the first 1,000,000 training/vocab articles to reach the test split...
  Evaluated 2,000/10,000 test samples...
  Evaluated 4,000/10,000 test samples...
  Evaluated 6,000/10,000 test samples...
  Evaluated 8,000/10,000 test samples...
Evaluation Complete!
  ➔ Overall Accuracy: 90.13% (9,013/10,000)
  ➔ Evaluation Time : 12.46s (802.85 samples/sec)

--- Step 5: Live Routing Examples ---
  Input   : "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles."
  ➔ Winner: BIOGRAPHY & PEOPLE (Expected: SCIENCE & TECHNOLOGY)

  Input   : "The Amazon River in South America is the largest river by discharge volume of water in the world, flowing through Peru, Colombia, and Brazil."
  ➔ Winner: GEOGRAPHY & PLACES (Expected: GEOGRAPHY & PLACES)

  Input   : "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity."
  ➔ Winner: BIOGRAPHY & PEOPLE (Expected: BIOGRAPHY & PEOPLE)

  Input   : "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789."
  ➔ Winner: BIOGRAPHY & PEOPLE (Expected: HISTORY & EVENTS)

  Input   : "The Starry Night is an oil-on-canvas painting by the Dutch Post-Impressionist painter Vincent van Gogh, painted in June 1889."
  ➔ Winner: BIOGRAPHY & PEOPLE (Expected: ARTS & CULTURE)

====================================================================
🎉 Wikipedia Classifier & Router completed successfully! 🎉
====================================================================
```
