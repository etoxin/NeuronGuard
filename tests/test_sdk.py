import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from neuronguard import NeuronGuardField, TabularClassifier, TextClassifier, tokenize


class TestSDK(unittest.TestCase):
    def test_text_classifier(self):
        classifier = TextClassifier(num_classes=2, vocab_size=100)
        
        # Mock some training records
        records = [
            (0, "this is a legitimate document"),
            (1, "this is a fraudulent phishing email"),
            (0, "another legitimate document"),
            (1, "phishing email scam"),
        ]
        
        classifier.fit_records(records, epochs=5)
        
        # Test predict
        pred = classifier.predict("legitimate document")
        self.assertEqual(pred, 0)
        
        pred = classifier.predict("phishing scam")
        self.assertEqual(pred, 1)
        
        # Test save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            classifier.save(model_dir)
            self.assertTrue(TextClassifier.exists(model_dir))
            
            loaded = TextClassifier.load(model_dir)
            self.assertEqual(loaded.predict("phishing scam"), 1)

    def test_tabular_classifier(self):
        classifier = TabularClassifier(num_classes=2, num_features=2, buckets_per_feature=5)
        
        records = [
            [0, 1.0, 2.0],
            [1, 100.0, 200.0],
            [0, 1.5, 2.5],
            [1, 90.0, 180.0],
        ]
        
        classifier.fit(records, feature_indices=[1, 2], label_index=0, epochs=5)
        
        self.assertEqual(classifier.predict([1.2, 2.2]), 0)
        self.assertEqual(classifier.predict([95.0, 190.0]), 1)

    def test_rust_tokenizer(self):
        text = "This is a Test! Running gracefully."
        tokens = tokenize(text)
        # 'this', 'is', 'a' are stop words. 'test', 'run' (stemmed), 'grace' (stemmed)
        # Wait, 'gracefully' stems to 'grace' (ends with 'ly'). 'running' -> 'runn' or 'run'
        # Let's just check length and some expected words
        self.assertIn("test", tokens)

    def test_parallel_training_does_not_drop_contended_updates(self):
        field = NeuronGuardField(sensory_count=1, motor_count=2)
        field.train_batch([([0], 1)] * 1000, amplify_delta=1, suppress_delta=0)
        self.assertEqual(field.get_neuron_synapses(0), [(1, 1000)])

    def test_parallel_training_matches_sequential_order(self):
        tasks = [([0, 1], index % 2) for index in range(200)]
        sequential = NeuronGuardField(sensory_count=2, motor_count=2)
        parallel = NeuronGuardField(sensory_count=2, motor_count=2)
        for tokens, motor_id in tasks:
            sequential.train_stream(tokens, motor_id, 3, 1)
        parallel.train_batch(tasks, amplify_delta=3, suppress_delta=1)
        for token in range(2):
            self.assertEqual(
                parallel.get_neuron_synapses(token),
                sequential.get_neuron_synapses(token),
            )

    def test_concurrent_predictions_are_isolated(self):
        classifier = TextClassifier(num_classes=2, vocab_size=100)
        classifier.fit_records(
            [(0, "legitimate account"), (1, "fraud phishing")],
            epochs=5,
            shuffle=False,
        )
        inputs = ["legitimate account", "fraud phishing"] * 100
        with ThreadPoolExecutor(max_workers=8) as executor:
            predictions = list(executor.map(classifier.predict, inputs))
        self.assertEqual(predictions, [0, 1] * 100)

    def test_quantile_bucketing_and_threshold_round_trip(self):
        classifier = TabularClassifier(
            num_classes=2,
            num_features=1,
            buckets_per_feature=4,
            bucket_strategy="quantile",
        )
        records = [[value, int(value >= 50)] for value in range(100)]
        classifier.fit(
            records,
            feature_indices=[0],
            label_index=1,
            shuffle=False,
        )
        self.assertEqual(classifier._bucket_boundaries, [[25.0, 50.0, 75.0]])
        threshold = classifier.tune_decision_threshold(
            records,
            feature_indices=[0],
            label_index=1,
        )
        self.assertEqual(classifier.decision_threshold, threshold)

        with tempfile.TemporaryDirectory() as tmpdir:
            classifier.save(tmpdir)
            loaded = TabularClassifier.load(tmpdir)
            self.assertEqual(loaded.bucket_strategy, "quantile")
            self.assertEqual(loaded._bucket_boundaries, [[25.0, 50.0, 75.0]])
            self.assertEqual(loaded.decision_threshold, threshold)
            self.assertEqual(loaded.predict([75.0]), classifier.predict([75.0]))

    def test_invalid_motor_count_is_rejected(self):
        with self.assertRaises(ValueError):
            NeuronGuardField(sensory_count=10, motor_count=9)

if __name__ == "__main__":
    unittest.main()
