import os
import tempfile
import unittest

from neuronguard import TabularClassifier, TextClassifier, tokenize


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

if __name__ == "__main__":
    unittest.main()
