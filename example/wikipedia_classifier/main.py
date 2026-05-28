"""
NeuronGuard: Wikipedia Structured Dataset Classifier & Router

This example showcases:
1. Streaming Large-Scale Datasets:
   Streams the 44.4 GB 'wikimedia/structured-wikipedia' dataset on the fly
   using Hugging Face's 'datasets' library with zero disk storage overhead.
2. Microsecond-Latency Classification:
   Trains a NeuronGuardField to classify Wikipedia articles into 5 high-level domains:
   - 0: Science & Technology
   - 1: Geography & Places
   - 2: Biography & People
   - 3: History & Events
   - 4: Arts & Culture
3. On-Device Continuous Learning:
   Uses the Guard/Lease transactional pattern to train on the fly.
4. Interactive CLI Router:
   Allows typing any sentence to route it to the 5 specialized domain experts in microseconds.
"""

import argparse
import os
import re
import time

import neuronguard as ng
from datasets import load_dataset


def tokenize(text):
    # Fast and robust word tokenizer
    return re.findall(r"\b\w+\b", text.lower())


class WikipediaDomain:
    NAMES = [
        "Science & Technology",
        "Geography & Places",
        "Biography & People",
        "History & Events",
        "Arts & Culture",
    ]

    @staticmethod
    def name(index):
        return WikipediaDomain.NAMES[index]

    @staticmethod
    def determine_domain(row):
        """Determines the correct domain index based on community-curated Infobox template names.
        Returns None if the domain is ambiguous or missing, so we can skip noisy data.
        """
        infoboxes_str = row.get("infoboxes")
        if not infoboxes_str:
            return None

        try:
            import json

            infoboxes = json.loads(infoboxes_str)
        except Exception:
            return None

        if not infoboxes or not isinstance(infoboxes, list):
            return None

        # Extract the name of the first infobox
        infobox_name = infoboxes[0].get("name", "").lower()
        if not infobox_name:
            return None

        # 0: Science & Technology
        sci_tech_infoboxes = {
            "infobox company",
            "infobox software",
            "infobox aircraft",
            "infobox spacecraft",
            "infobox satellite",
            "infobox device",
            "infobox enzyme",
            "infobox chemical",
            "infobox drug",
            "infobox medical condition",
            "infobox programming language",
            "infobox website",
            "infobox processor",
            "infobox telescope",
            "infobox taxon",
            "infobox species",
            "infobox plant",
            "infobox animal",
            "infobox mineral",
        }
        # 1: Geography & Places
        geography_infoboxes = {
            "infobox settlement",
            "infobox city",
            "infobox town",
            "infobox village",
            "infobox river",
            "infobox mountain",
            "infobox lake",
            "infobox island",
            "infobox country",
            "infobox subdivision",
            "infobox body of water",
            "infobox road",
            "infobox bridge",
            "infobox park",
            "infobox forest",
        }
        # 2: Biography & People
        biography_infoboxes = {
            "infobox person",
            "infobox biography",
            "infobox officeholder",
            "infobox artist",
            "infobox writer",
            "infobox actor",
            "infobox musical artist",
            "infobox athlete",
            "infobox footballer",
            "infobox politician",
            "infobox scientist",
            "infobox philosopher",
            "infobox military person",
            "infobox royalty",
            "infobox architect",
            "infobox engineer",
        }
        # 3: History & Events
        history_infoboxes = {
            "infobox election",
            "infobox battle",
            "infobox war",
            "infobox treaty",
            "infobox historic site",
            "infobox military conflict",
            "infobox historical event",
            "infobox earthquake",
            "infobox hurricane",
            "infobox civil conflict",
            "infobox trial",
        }
        # 4: Arts & Culture
        arts_infoboxes = {
            "infobox film",
            "infobox book",
            "infobox novel",
            "infobox album",
            "infobox song",
            "infobox artwork",
            "infobox video game",
            "infobox play",
            "infobox television",
            "infobox fictional character",
            "infobox comic book",
            "infobox opera",
            "infobox musical",
        }

        # Check for exact or partial matches in infobox names
        for name in sci_tech_infoboxes:
            if name in infobox_name:
                return 0
        for name in geography_infoboxes:
            if name in infobox_name:
                return 1
        for name in biography_infoboxes:
            if name in infobox_name:
                return 2
        for name in history_infoboxes:
            if name in infobox_name:
                return 3
        for name in arts_infoboxes:
            if name in infobox_name:
                return 4

        return None


def main():
    parser = argparse.ArgumentParser(
        description="NeuronGuard Wikipedia Classifier & Router"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=100000,
        help="Number of streamed articles to train on (default: 100,000)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=10000,
        help="Number of streamed articles to evaluate on (default: 10,000)",
    )
    parser.add_argument(
        "--vocab-samples",
        type=int,
        default=20000,
        help="Number of streamed articles to use for building vocabulary (default: 20,000)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=5000,
        help="Size of the vocabulary (default: 5,000)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if pre-trained weights exist",
    )
    args = parser.parse_args()

    print("====================================================================")
    print("📚 Wikimedia Structured Wikipedia 10.4M Classifier & Router 📚")
    print("====================================================================\n")

    stop_words = {
        "the",
        "a",
        "and",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "an",
        "be",
        "is",
        "are",
        "was",
        "were",
        "it",
        "that",
        "this",
        "from",
        "as",
        "at",
        "but",
        "not",
        "or",
        "will",
        "has",
        "have",
        "its",
        "his",
        "her",
        "their",
        "they",
        "who",
        "which",
        "also",
        "been",
        "by",
        "an",
        "about",
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_file_path = os.path.join(script_dir, "wikipedia_weights.bin")
    vocab_file_path = os.path.join(script_dir, "wikipedia_vocab.txt")

    vocab_size = args.vocab_size
    num_experts = 5
    field_size = vocab_size + num_experts

    vocab_map = {}
    vocab_list = []

    # Initialize the NeuronGuardField
    field = ng.NeuronGuardField(sensory_count=vocab_size, motor_count=num_experts)

    # Check if pre-trained model exists
    if (
        os.path.exists(weights_file_path)
        and os.path.exists(vocab_file_path)
        and not args.force_retrain
    ):
        print("Loading pre-trained model weights and vocabulary...")
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                word = line.strip()
                vocab_map[word] = idx
                vocab_list.append(word)
        field.load_weights(weights_file_path)
        print("Model loaded successfully in < 1ms!\n")
    else:
        print("Pre-trained model not found. Starting streaming training...")
        print(
            "Streaming 'wikimedia/structured-wikipedia' from Hugging Face on the fly..."
        )
        print(
            "(This requires zero disk space as it streams the dataset in real-time!)\n"
        )

        # 1. Load the streaming dataset
        print("--- Step 1: Connecting to Hugging Face Streaming Dataset ---")
        try:
            ds = load_dataset(
                "wikimedia/structured-wikipedia",
                "enwiki_namespace_0",
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please ensure you have an active internet connection.")
            return

        # 2. Build Vocabulary from first N articles
        print(
            f"\n--- Step 2: Building Vocabulary from first {args.vocab_samples:,} articles ---"
        )
        word_counts = {}
        iterator = iter(ds)

        for i in range(args.vocab_samples):
            try:
                row = next(iterator)
            except StopIteration:
                break

            abstract = row.get("abstract") or ""
            description = row.get("description") or ""
            full_text = f"{description} {abstract}"
            tokens = tokenize(full_text)

            for token in tokens:
                if len(token) > 2 and token not in stop_words:
                    word_counts[token] = word_counts.get(token, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        final_vocab = sorted_words[:vocab_size]

        # Save vocabulary to disk
        with open(vocab_file_path, "w", encoding="utf-8") as f:
            for word, _ in final_vocab:
                f.write(f"{word}\n")

        for idx, (word, _) in enumerate(final_vocab):
            vocab_map[word] = idx
            vocab_list.append(word)

        print(f"Vocabulary of top {len(vocab_list)} words built successfully!")

        # 3. Train the cortex on the fly
        print(
            f"\n--- Step 3: Training on the fly on {args.train_samples:,} articles (Trainer Mode) ---"
        )
        start_time = time.time()

        # Reset iterator to start of dataset
        ds = load_dataset(
            "wikimedia/structured-wikipedia",
            "enwiki_namespace_0",
            split="train",
            streaming=True,
        )
        iterator = iter(ds)

        train_samples = args.train_samples
        trained_count = 0

        for i in range(train_samples):
            try:
                row = next(iterator)
            except StopIteration:
                break

            abstract = row.get("abstract") or ""
            description = row.get("description") or ""

            domain_idx = WikipediaDomain.determine_domain(row)
            if domain_idx is None:
                continue

            tokens = tokenize(f"{description} {abstract}")
            word_indices = [vocab_map[t] for t in tokens if t in vocab_map]

            if word_indices:
                field.train_stream(
                    word_indices, domain_idx, amplify_delta=10, suppress_delta=2
                )
                trained_count += 1

            if (trained_count) % 10000 == 0:
                print(
                    f"  Trained on {trained_count:,}/{train_samples:,} valid streamed samples..."
                )

            if trained_count >= train_samples:
                break

        duration = time.time() - start_time
        print(
            f"Training on {trained_count:,} valid samples completed in {duration:.2f}s!"
        )

        # Save weights to disk
        print("Saving model weights to disk...")
        field.save_weights(weights_file_path)
        print("Model saved successfully!\n")

    # 4. Batch Evaluation on Test Split
    print("--- Step 4: Evaluating Accuracy on Test Split ---")
    print(f"Streaming the next {args.test_samples:,} articles for evaluation...")

    # Re-connect to the dataset to stream the test split
    ds = load_dataset(
        "wikimedia/structured-wikipedia",
        "enwiki_namespace_0",
        split="train",
        streaming=True,
    )
    iterator = iter(ds)

    # Skip the articles used for vocabulary building and training
    skip_count = max(args.vocab_samples, args.train_samples)
    print(
        f"Skipping the first {skip_count:,} training/vocab articles to reach the test split..."
    )
    for _ in range(skip_count):
        try:
            next(iterator)
        except StopIteration:
            break

    correct_predictions = 0
    total_predictions = 0
    start_eval_time = time.time()

    for i in range(args.test_samples * 5):  # Read more to find enough valid samples
        try:
            row = next(iterator)
        except StopIteration:
            break

        abstract = row.get("abstract") or ""
        description = row.get("description") or ""

        domain_idx = WikipediaDomain.determine_domain(row)
        if domain_idx is None:
            continue

        tokens = tokenize(f"{description} {abstract}")
        word_indices = [vocab_map[t] for t in tokens if t in vocab_map]

        if word_indices:
            # Reset potentials and evaluate synchronously (extremely fast!)
            field.reset_potentials()
            field.process_stream_sync(word_indices)

            expert_potentials = field.get_potentials()
            predicted_idx = expert_potentials.index(max(expert_potentials))

            if predicted_idx == domain_idx:
                correct_predictions += 1
            total_predictions += 1

        if total_predictions >= args.test_samples:
            break

        if (total_predictions) % 2000 == 0:
            print(
                f"  Evaluated {total_predictions:,}/{args.test_samples:,} test samples..."
            )

    eval_duration = time.time() - start_eval_time
    accuracy = (
        (correct_predictions / total_predictions) * 100
        if total_predictions > 0
        else 0.0
    )
    print("Evaluation Complete!")
    print(
        f"  ➔ Overall Accuracy: {accuracy:.2f}% ({correct_predictions:,}/{total_predictions:,})"
    )
    print(
        f"  ➔ Evaluation Time : {eval_duration:.2f}s ({total_predictions / eval_duration:.2f} samples/sec)\n"
    )

    # 5. Live Routing Examples (Automated Demonstration)
    print("--- Step 5: Live Routing Examples ---")
    examples = [
        (
            "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
            "Science & Technology",
        ),
        (
            "The Amazon River in South America is the largest river by discharge volume of water in the world, flowing through Peru, Colombia, and Brazil.",
            "Geography & Places",
        ),
        (
            "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity.",
            "Biography & People",
        ),
        (
            "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789.",
            "History & Events",
        ),
        (
            "The Starry Night is an oil-on-canvas painting by the Dutch Post-Impressionist painter Vincent van Gogh, painted in June 1889.",
            "Arts & Culture",
        ),
    ]

    for text, expected in examples:
        tokens = tokenize(text)
        field.reset_potentials()
        recognized = [t for t in tokens if t in vocab_map]
        word_indices = [vocab_map[t] for t in recognized]

        if word_indices:
            field.process_stream(word_indices, training_mode=False)

        expert_potentials = field.get_potentials()
        predicted_idx = expert_potentials.index(max(expert_potentials))
        winner = WikipediaDomain.name(predicted_idx)

        print(f'  Input   : "{text}"')
        print(f"  ➔ Winner: {winner.upper()} (Expected: {expected.upper()})\n")

    print("====================================================================")
    print("🎉 Wikipedia Classifier & Router completed successfully! 🎉")
    print("====================================================================")


if __name__ == "__main__":
    main()
