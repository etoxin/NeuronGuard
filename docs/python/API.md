<a id="neuronguard"></a>

# neuronguard

NeuronGuard: Cache-Aligned Neuromorphic Event Engine

This package provides both the raw Rust bindings and a high-level Python SDK.

Raw bindings (backward compatible)::

    import neuronguard as ng
    field = ng.NeuronGuardField(sensory_count=1000, motor_count=4)

High-level SDK::

    from neuronguard import TextClassifier, TabularClassifier

<a id="neuronguard.vocab"></a>

# neuronguard.vocab

Discriminative vocabulary builder for NeuronGuard text classification.

Replaces the raw-frequency vocabulary selection used in previous examples
with a scoring function that prioritises words concentrated in fewer categories.

<a id="neuronguard.vocab.discriminative_score"></a>

#### discriminative\_score

```python
def discriminative_score(counts: List[int]) -> float
```

Score a word by frequency × category concentration.

A word appearing 1000 times with 80% in one category scores 800.
A word appearing 2000 times uniformly across 4 categories scores 500.
The discriminative word wins the vocabulary slot.

**Arguments**:

- `counts` _List[int]_ - List of per-category occurrence counts.
  

**Returns**:

- `float` - A float score (higher = more discriminative).
  

**Examples**:

  >>> discriminative_score([800, 100, 100])
  800.0

<a id="neuronguard.vocab.frequency_score"></a>

#### frequency\_score

```python
def frequency_score(counts: List[int]) -> float
```

Score a word by raw total frequency (legacy behaviour).

**Arguments**:

- `counts` _List[int]_ - List of per-category occurrence counts.
  

**Returns**:

- `float` - Total frequency as a float.

<a id="neuronguard.vocab.build_vocab"></a>

#### build\_vocab

```python
def build_vocab(
    records: Iterable[Tuple[int, str]],
    num_classes: int,
    vocab_size: int,
    stop_words: Optional[Set[str]] = None,
    scoring: str = "discriminative",
    apply_stemming: bool = True
) -> Tuple[Dict[str, int], List[Tuple[str, List[int], int]], Dict[str, Any]]
```

Build a vocabulary from training records using discriminative scoring.

**Arguments**:

- `records` _Iterable[Tuple[int, str]]_ - Iterable of (label, text) tuples where label is an int
  (0-indexed class) and text is the raw input string.
- `num_classes` _int_ - Number of output classes.
- `vocab_size` _int_ - Maximum vocabulary size.
- `stop_words` _Optional[Set[str]], optional_ - Optional set of stop words (defaults to tokenizer's built-in set). Defaults to None.
- `scoring` _str, optional_ - Scoring strategy — "discriminative" or "frequency". Defaults to "discriminative".
- `apply_stemming` _bool, optional_ - Whether to apply lightweight stemming during tokenization. Defaults to True.
  

**Returns**:

  Tuple[Dict[str, int], List[Tuple[str, List[int], int]], Dict[str, Any]]: A tuple of (vocab_map, vocab_list, vocab_stats) where:
  - vocab_map: dict mapping word → index (0-indexed)
  - vocab_list: list of (word, per_class_counts, total_count) tuples
  - vocab_stats: dict of summary statistics
  

**Examples**:

  >>> build_vocab([(0, "hello world")], 2, 1000)
- `({'hello'` - 0, 'world': 1}, [('hello', [1, 0], 1), ('world', [1, 0], 1)], {'total_records': 1, 'unique_words_seen': 2, 'vocab_size': 2, 'scoring': 'discriminative'})

<a id="neuronguard.tabular"></a>

# neuronguard.tabular

TabularClassifier: High-level tabular/numerical classification API for NeuronGuard.

Wraps the bare-metal NeuronGuardField with automatic feature bucketing, class
weighting for imbalanced datasets, and atomic prediction.

Replaces the manual bucket-computation and oversampling boilerplate in the
fraud scanner example.

<a id="neuronguard.tabular.TabularClassifier"></a>

## TabularClassifier Objects

```python
class TabularClassifier()
```

High-level tabular data classifier backed by a NeuronGuardField.

Automatically buckets continuous features into discrete sensory neuron
indices and handles class imbalance through configurable oversampling.

Example::

classifier = TabularClassifier(num_classes=2, num_features=5)
classifier.fit(
records=train_data,
feature_indices=[0, 1, 2, 3, 4],
label_index=5,
class_weights={1: 100},
)
label = classifier.predict([v10, v12, v14, v17, amount])

<a id="neuronguard.tabular.TabularClassifier.__init__"></a>

#### \_\_init\_\_

```python
def __init__(num_classes: int,
             num_features: int,
             buckets_per_feature: int = 10,
             amplify_delta: int = 15,
             suppress_delta: int = 5,
             baseline_delta: int = 10,
             use_feature_interactions: bool = False,
             interaction_vocab_size: int = 1000000) -> None
```

Initialise a TabularClassifier.

**Arguments**:

- `num_classes` _int_ - Number of output categories.
- `num_features` _int_ - Number of input features.
- `buckets_per_feature` _int, optional_ - Number of discrete buckets per feature. Defaults to 10.
- `amplify_delta` _int, optional_ - Weight increment for the correct class during training. Defaults to 15.
- `suppress_delta` _int, optional_ - Weight decrement for incorrect classes during training. Defaults to 5.
- `baseline_delta` _int, optional_ - Weight used for initial baseline seeding. Defaults to 10.
- `use_feature_interactions` _bool, optional_ - If True, hashes pairs of features to capture 2D non-linear patterns. Defaults to False.
- `interaction_vocab_size` _int, optional_ - Size of the hash space for interactions to prevent collisions. Defaults to 1000000.

<a id="neuronguard.tabular.TabularClassifier.fit"></a>

#### fit

```python
def fit(records: Iterable[Union[List[float], Tuple[float, ...]]],
        feature_indices: List[int],
        label_index: int,
        epochs: int = 1,
        shuffle: bool = True,
        class_weights: Optional[Dict[int, int]] = None,
        default_class: int = 0) -> None
```

Train the classifier on tabular records.

**Arguments**:

- `records` _Iterable[Union[List[float], Tuple[float, ...]]]_ - List of records (lists/tuples of values).
- `feature_indices` _List[int]_ - List of column indices for input features.
- `label_index` _int_ - Column index for the integer class label.
- `epochs` _int, optional_ - Number of training epochs. Defaults to 1.
- `shuffle` _bool, optional_ - Whether to shuffle records before each epoch. Defaults to True.
- `class_weights` _Optional[Dict[int, int]], optional_ - Optional dict mapping class_label → oversample_multiplier.
  For example, {1: 100} trains fraud cases 100 times per epoch. Defaults to None.
- `default_class` _int, optional_ - The class to seed all neurons to initially (e.g., 0 for "legitimate"). Defaults to 0.

<a id="neuronguard.tabular.TabularClassifier.fit_from_csv"></a>

#### fit\_from\_csv

```python
def fit_from_csv(file_path: str,
                 feature_indices: List[int],
                 label_index: int,
                 epochs: int = 1,
                 class_weights: Optional[Dict[int, int]] = None,
                 default_class: int = 0,
                 delimiter: str = ",",
                 skip_header: bool = False) -> None
```

Train the classifier by streaming directly from a CSV file.

This uses O(1) memory and is designed for massive datasets (10M+ rows)
that cannot fit in RAM. It makes multiple passes over the file.

**Arguments**:

- `file_path` _str_ - Path to the CSV file.
- `feature_indices` _List[int]_ - List of column indices for input features.
- `label_index` _int_ - Column index for the integer class label.
- `epochs` _int, optional_ - Number of training epochs. Defaults to 1.
- `class_weights` _Optional[Dict[int, int]], optional_ - Optional dict mapping class_label -> oversample_multiplier. Defaults to None.
- `default_class` _int, optional_ - The class to seed all neurons to initially. Defaults to 0.
- `delimiter` _str, optional_ - CSV delimiter. Defaults to ",".
- `skip_header` _bool, optional_ - Whether to skip the first row. Defaults to False.

<a id="neuronguard.tabular.TabularClassifier.update"></a>

#### update

```python
def update(X: Iterable[Union[List[float], Tuple[float, ...]]],
           label_index: int,
           class_weights: Optional[Dict[int, int]] = None) -> None
```

Continually learn from new records on the fly.

This enables zero-overhead online/continuous learning. The model weights are
updated instantly.

**Arguments**:

- `X` _Iterable[Union[List[float], Tuple[float, ...]]]_ - Iterable of lists of floats (features) with the label appended.
- `label_index` _int_ - The index of the label in each record.
- `class_weights` _Optional[Dict[int, int]], optional_ - Optional dict mapping class_label -> oversample_multiplier. Defaults to None.

<a id="neuronguard.tabular.TabularClassifier.unlearn"></a>

#### unlearn

```python
def unlearn(X: Iterable[Union[List[float], Tuple[float, ...]]],
            label_index: int) -> None
```

Surgically unlearn records by applying negative Hebbian deltas.

**Arguments**:

- `X` _Iterable[Union[List[float], Tuple[float, ...]]]_ - Iterable of lists of floats (features) with the label appended.
- `label_index` _int_ - The index of the label in each record.

<a id="neuronguard.tabular.TabularClassifier.predict"></a>

#### predict

```python
def predict(features: List[float]) -> int
```

Classify a feature vector and return the predicted class index.

Atomically handles reset → tokenize → process → argmax.

**Arguments**:

- `features` _List[float]_ - List of numerical feature values (same order as training features).
  

**Returns**:

- `int` - The predicted class index (0-indexed).

<a id="neuronguard.tabular.TabularClassifier.predict_scores"></a>

#### predict\_scores

```python
def predict_scores(features: List[float]) -> List[int]
```

Classify a feature vector and return raw potentials for all classes.

**Arguments**:

- `features` _List[float]_ - List of numerical feature values.
  

**Returns**:

- `List[int]` - A list of integer potentials, one per class.

<a id="neuronguard.tabular.TabularClassifier.evaluate"></a>

#### evaluate

```python
def evaluate(records: Iterable[Union[List[float], Tuple[float, ...]]],
             feature_indices: List[int],
             label_index: int) -> Tuple[float, str]
```

Evaluate accuracy on test records.

**Arguments**:

- `records` _Iterable[Union[List[float], Tuple[float, ...]]]_ - List of test records.
- `feature_indices` _List[int]_ - List of column indices for input features.
- `label_index` _int_ - Column index for the class label.
  

**Returns**:

  Tuple[float, str]: A tuple of (accuracy_pct, report_str) with per-class metrics.

<a id="neuronguard.tabular.TabularClassifier.evaluate_from_csv"></a>

#### evaluate\_from\_csv

```python
def evaluate_from_csv(file_path: str,
                      feature_indices: List[int],
                      label_index: int,
                      delimiter: str = ",",
                      skip_header: bool = False) -> Tuple[float, str]
```

Evaluate accuracy by streaming directly from a CSV file.

**Arguments**:

- `file_path` _str_ - Path to the test CSV file.
- `feature_indices` _List[int]_ - List of column indices for input features.
- `label_index` _int_ - Column index for the class label.
- `delimiter` _str, optional_ - CSV delimiter. Defaults to ",".
- `skip_header` _bool, optional_ - Whether to skip the first row. Defaults to False.
  

**Returns**:

  Tuple[float, str]: A tuple of (accuracy_pct, report_str) with per-class metrics.

<a id="neuronguard.tabular.TabularClassifier.save"></a>

#### save

```python
def save(path: str) -> None
```

Save the trained model to a directory.

**Arguments**:

- `path` _str_ - Directory path to save the model to.

<a id="neuronguard.tabular.TabularClassifier.load"></a>

#### load

```python
@classmethod
def load(cls, path: str) -> "TabularClassifier"
```

Load a trained model from a directory.

**Arguments**:

- `path` _str_ - Directory path containing weights.bin and config.json.
  

**Returns**:

- `TabularClassifier` - A fitted TabularClassifier instance.

<a id="neuronguard.tabular.TabularClassifier.exists"></a>

#### exists

```python
@staticmethod
def exists(path: str) -> bool
```

Check whether a saved model exists at the given path.

**Arguments**:

- `path` _str_ - Directory path to check.
  

**Returns**:

- `bool` - True if weights and config exist.

<a id="neuronguard.tokenizer"></a>

# neuronguard.tokenizer

Unified tokenizer for NeuronGuard text classification.

Provides a single, consistent tokenization pipeline used by all classifiers.
Replaces the 3 different ad-hoc tokenizers previously scattered across examples.

<a id="neuronguard.tokenizer.tokenize"></a>

#### tokenize

```python
def tokenize(text: str,
             stop_words: Optional[Set[str]] = None,
             apply_stemming: bool = True,
             min_length: int = MIN_TOKEN_LENGTH) -> List[str]
```

Tokenize a string into a list of words.

**Arguments**:

- `text` _str_ - The input text to tokenize.
- `stop_words` _Optional[Set[str]], optional_ - A set of stop words to exclude. Defaults to None, which uses DEFAULT_STOP_WORDS.
- `apply_stemming` _bool, optional_ - Whether to apply stemming to the tokens. Defaults to True.
- `min_length` _int, optional_ - Minimum length for a token to be kept. Defaults to MIN_TOKEN_LENGTH.
  

**Returns**:

- `List[str]` - A list of processed token strings.
  

**Examples**:

  >>> tokenize("The quick brown foxes!", apply_stemming=True)
  ['quick', 'brown', 'fox']

<a id="neuronguard.text"></a>

# neuronguard.text

TextClassifier: High-level text classification API for NeuronGuard.

Wraps the bare-metal NeuronGuardField Rust core with unified tokenization,
discriminative vocabulary building, correct default hyperparameters, and
a predict() method that handles reset → process → argmax atomically.

Replaces the 100-200 lines of boilerplate previously duplicated in every
text classification example.

<a id="neuronguard.text.TextClassifier"></a>

## TextClassifier Objects

```python
class TextClassifier()
```

High-level text classifier backed by a NeuronGuardField.

Handles vocabulary building, weight seeding, multi-epoch shuffled training,
and atomic prediction in a single clean API. All accuracy-critical defaults
(discriminative scoring, 3:1 amplify/suppress ratio, proportional seeding)
are baked in.

Example::

classifier = TextClassifier(num_classes=4, vocab_size=1000)
classifier.fit("train.csv", text_col=[1, 2], label_col=0, epochs=3)
accuracy, report = classifier.evaluate("test.csv", text_col=[1, 2], label_col=0)
label = classifier.predict("some text to classify")

<a id="neuronguard.text.TextClassifier.__init__"></a>

#### \_\_init\_\_

```python
def __init__(num_classes: int,
             vocab_size: int = 5000,
             amplify_delta: int = 15,
             suppress_delta: int = 5,
             seed_max_weight: int = 30,
             stop_words: Optional[Iterable[str]] = None,
             apply_stemming: bool = True,
             vocab_scoring: str = "discriminative",
             class_names: Optional[List[str]] = None,
             use_hashed_bigrams: bool = False) -> None
```

Initialise a TextClassifier.

**Arguments**:

- `num_classes` _int_ - Number of output categories.
- `vocab_size` _int, optional_ - Maximum vocabulary size (number of sensory neurons). Defaults to 5000.
- `amplify_delta` _int, optional_ - Weight increment for the correct class during training. Defaults to 15.
- `suppress_delta` _int, optional_ - Weight decrement for incorrect classes during training. Defaults to 5.
- `seed_max_weight` _int, optional_ - Maximum weight used when pre-seeding category distributions. Defaults to 30.
- `stop_words` _Optional[Iterable[str]], optional_ - Optional custom stop words set. Defaults to built-in comprehensive list.
- `apply_stemming` _bool, optional_ - Whether to apply lightweight suffix stripping. Defaults to True.
- `vocab_scoring` _str, optional_ - Vocabulary scoring strategy — "discriminative" or "frequency". Defaults to "discriminative".
- `class_names` _Optional[List[str]], optional_ - Optional list of human-readable class names. Defaults to None.
- `use_hashed_bigrams` _bool, optional_ - Whether to use hashed bigrams. Defaults to False.

<a id="neuronguard.text.TextClassifier.fit"></a>

#### fit

```python
def fit(train_file: str,
        text_col: Union[int, List[int]],
        label_col: int,
        epochs: int = 3,
        shuffle: bool = True,
        label_offset: int = -1) -> None
```

Train the classifier from a CSV file.

**Arguments**:

- `train_file` _str_ - Path to the training CSV file.
- `text_col` _Union[int, List[int]]_ - Column index (int) or list of column indices to concatenate as text.
- `label_col` _int_ - Column index containing the integer class label.
- `epochs` _int, optional_ - Number of training epochs with per-epoch shuffling. Defaults to 3.
- `shuffle` _bool, optional_ - Whether to shuffle records before each epoch. Defaults to True.
- `label_offset` _int, optional_ - Value subtracted from the raw label to get a 0-indexed class.
  Use -1 for 1-indexed CSV labels (default), 0 if labels are already 0-indexed. Defaults to -1.

<a id="neuronguard.text.TextClassifier.fit_records"></a>

#### fit\_records

```python
def fit_records(records: Iterable[Tuple[int, str]],
                epochs: int = 3,
                shuffle: bool = True) -> None
```

Train the classifier from pre-parsed records.

**Arguments**:

- `records` _Iterable[Tuple[int, str]]_ - Iterable of (label, text) tuples where label is a
  0-indexed int and text is the raw input string.
- `epochs` _int, optional_ - Number of training epochs. Defaults to 3.
- `shuffle` _bool, optional_ - Whether to shuffle records before each epoch. Defaults to True.

<a id="neuronguard.text.TextClassifier.update_records"></a>

#### update\_records

```python
def update_records(records: Iterable[Tuple[int, str]]) -> None
```

Continually learn from new records on the fly without rebuilding the vocabulary.

This enables zero-overhead online/continuous learning. The model weights are
updated instantly. Words not in the original vocabulary are ignored.

**Arguments**:

- `records` _Iterable[Tuple[int, str]]_ - Iterable of (label, text) tuples.

<a id="neuronguard.text.TextClassifier.unlearn_records"></a>

#### unlearn\_records

```python
def unlearn_records(records: Iterable[Tuple[int, str]]) -> None
```

Instantly 'unlearn' records to comply with data privacy or correct errors.

Because NeuronGuard uses reversible Hebbian plasticity rather than entangled
gradient descent, you can cleanly subtract the exact synaptic weight modifications
caused by a specific record. This solves the 'Machine Unlearning' problem instantly.

**Arguments**:

- `records` _Iterable[Tuple[int, str]]_ - Iterable of (label, text) tuples to unlearn.

<a id="neuronguard.text.TextClassifier.predict"></a>

#### predict

```python
def predict(text: str) -> int
```

Classify text and return the predicted class index.

Atomically handles reset → tokenize → process → argmax.
Forgetting to reset potentials between predictions was a common
silent accuracy bug in the raw API — this method makes it impossible.

**Arguments**:

- `text` _str_ - The input text string to classify.
  

**Returns**:

- `int` - The predicted class index (0-indexed).

<a id="neuronguard.text.TextClassifier.predict_scores"></a>

#### predict\_scores

```python
def predict_scores(text: str) -> List[int]
```

Classify text and return raw motor neuron potentials for all classes.

**Arguments**:

- `text` _str_ - The input text string to classify.
  

**Returns**:

- `List[int]` - A list of integer potentials, one per class.

<a id="neuronguard.text.TextClassifier.predict_name"></a>

#### predict\_name

```python
def predict_name(text: str) -> str
```

Classify text and return the human-readable class name.

Requires class_names to be set during construction.

**Arguments**:

- `text` _str_ - The input text string to classify.
  

**Returns**:

- `str` - The predicted class name string.

<a id="neuronguard.text.TextClassifier.explain"></a>

#### explain

```python
def explain(text: str) -> Dict[str, Any]
```

Provide a transparent, token-by-token explanation for a prediction.

Because NeuronGuard is a direct associative memory rather than a black-box
neural network, we can perfectly trace exactly which words contributed
to the final prediction, and by exactly how much weight.

**Arguments**:

- `text` _str_ - The input text string to classify.
  

**Returns**:

  Dict[str, Any]: A dictionary containing:
  - 'prediction': The predicted class index.
  - 'prediction_name': The predicted class name.
  - 'total_scores': Raw potentials for each class.
  - 'word_contributions': A list of dicts detailing each word's exact weight contribution.

<a id="neuronguard.text.TextClassifier.get_class_features"></a>

#### get\_class\_features

```python
def get_class_features(class_idx: int,
                       top_k: int = 10) -> List[Tuple[str, int]]
```

Introspect the memory to find the most strongly associated words for a class.

**Arguments**:

- `class_idx` _int_ - The class index to inspect.
- `top_k` _int, optional_ - Number of top words to return. Defaults to 10.
  

**Returns**:

  List[Tuple[str, int]]: A list of (word, weight) tuples.

<a id="neuronguard.text.TextClassifier.print_explanation"></a>

#### print\_explanation

```python
def print_explanation(text: str) -> None
```

Out-of-the-box diagnostic print for prediction explanations.

**Arguments**:

- `text` _str_ - The input text string to classify.

<a id="neuronguard.text.TextClassifier.print_class_features"></a>

#### print\_class\_features

```python
def print_class_features(class_idx: int, top_k: int = 10) -> None
```

Out-of-the-box diagnostic print for class features.

**Arguments**:

- `class_idx` _int_ - The class index to inspect.
- `top_k` _int, optional_ - Number of top words to return. Defaults to 10.

<a id="neuronguard.text.TextClassifier.evaluate"></a>

#### evaluate

```python
def evaluate(test_file: str,
             text_col: Union[int, List[int]],
             label_col: int,
             label_offset: int = -1) -> Tuple[float, str]
```

Evaluate accuracy on a test CSV file.

**Arguments**:

- `test_file` _str_ - Path to the test CSV file.
- `text_col` _Union[int, List[int]]_ - Column index or list of indices for text.
- `label_col` _int_ - Column index for the integer class label.
- `label_offset` _int, optional_ - Value subtracted from raw label. Defaults to -1 for 1-indexed.
  

**Returns**:

  Tuple[float, str]: A tuple of (accuracy_pct, report_str) where report_str is a
  formatted table with per-class Precision, Recall, and F1.

<a id="neuronguard.text.TextClassifier.evaluate_records"></a>

#### evaluate\_records

```python
def evaluate_records(records: Iterable[Tuple[int, str]]) -> Tuple[float, str]
```

Evaluate accuracy on pre-parsed records.

**Arguments**:

- `records` _Iterable[Tuple[int, str]]_ - Iterable of (label, text) tuples.
  

**Returns**:

  Tuple[float, str]: A tuple of (accuracy_pct, report_str).

<a id="neuronguard.text.TextClassifier.save"></a>

#### save

```python
def save(path: str) -> None
```

Save the trained model to a directory.

Creates a self-contained directory with weights, vocabulary, and config.
Unlike the raw API (which saves weights and vocab separately), this
bundles everything so they can never get out of sync.

**Arguments**:

- `path` _str_ - Directory path to save the model to.

<a id="neuronguard.text.TextClassifier.load"></a>

#### load

```python
@classmethod
def load(cls, path: str) -> "TextClassifier"
```

Load a trained model from a directory.

**Arguments**:

- `path` _str_ - Directory path containing weights.bin, vocab.txt, and config.json.
  

**Returns**:

- `TextClassifier` - A fitted TextClassifier instance.

<a id="neuronguard.text.TextClassifier.exists"></a>

#### exists

```python
@staticmethod
def exists(path: str) -> bool
```

Check whether a saved model exists at the given path.

**Arguments**:

- `path` _str_ - Directory path to check.
  

**Returns**:

- `bool` - True if all required model files exist.

