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
    def determine_domain(description, abstract):
        """Determines the correct domain index based on keywords in the description or abstract."""
        text = f"{description or ''} {abstract or ''}".lower()

        # 0: Science & Technology
        sci_tech_keywords = {
            "science",
            "technology",
            "physics",
            "chemistry",
            "software",
            "computer",
            "engine",
            "space",
            "aircraft",
            "satellite",
            "species",
            "animal",
            "plant",
            "genus",
            "biology",
            "mathematics",
            "astronomy",
            "device",
            "invention",
            "system",
            "discovery",
            "enzyme",
        }
        # 1: Geography & Places
        geography_keywords = {
            "city",
            "town",
            "village",
            "river",
            "mountain",
            "lake",
            "island",
            "country",
            "province",
            "capital",
            "located",
            "municipality",
            "region",
            "state",
            "county",
            "geography",
            "ocean",
        }
        # 2: Biography & People
        biography_keywords = {
            "born",
            "died",
            "politician",
            "actor",
            "singer",
            "writer",
            "player",
            "athlete",
            "officer",
            "king",
            "queen",
            "emperor",
            "president",
            "minister",
            "activist",
            "founder",
            "director",
        }
        # 3: History & Events
        history_keywords = {
            "war",
            "battle",
            "election",
            "treaty",
            "dynasty",
            "empire",
            "revolution",
            "founded",
            "established",
            "century",
            "year",
            "ancient",
            "historical",
            "reign",
            "parliamentary",
        }
        # 4: Arts & Culture
        arts_keywords = {
            "film",
            "movie",
            "book",
            "novel",
            "album",
            "song",
            "music",
            "painting",
            "sculpture",
            "theatre",
            "game",
            "play",
            "artist",
            "composer",
            "band",
            "museum",
            "exhibition",
        }

        words = set(tokenize(text))

        scores = [
            len(words.intersection(sci_tech_keywords)),
            len(words.intersection(geography_keywords)),
            len(words.intersection(biography_keywords)),
            len(words.intersection(history_keywords)),
            len(words.intersection(arts_keywords)),
        ]

        max_score = max(scores)
        if max_score > 0:
            return scores.index(max_score)

        # Default fallback
        return 0


def main():
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

    vocab_size = 5000
    num_experts = 5
    field_size = vocab_size + num_experts

    vocab_map = {}
    vocab_list = []

    # Initialize the NeuronGuardField
    field = ng.NeuronGuardField(sensory_count=vocab_size, motor_count=num_experts)

    # Check if pre-trained model exists
    if os.path.exists(weights_file_path) and os.path.exists(vocab_file_path):
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

        # 2. Build Vocabulary from first 20,000 articles
        print("\n--- Step 2: Building Vocabulary from first 20,000 articles ---")
        word_counts = {}
        iterator = iter(ds)

        for i in range(20000):
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
        print("\n--- Step 3: Training on the fly (Trainer Mode) ---")
        start_time = time.time()

        # Reset iterator to start of dataset
        ds = load_dataset(
            "wikimedia/structured-wikipedia",
            "enwiki_namespace_0",
            split="train",
            streaming=True,
        )
        iterator = iter(ds)

        train_samples = 30000
        trained_count = 0

        for i in range(train_samples):
            try:
                row = next(iterator)
            except StopIteration:
                break

            abstract = row.get("abstract") or ""
            description = row.get("description") or ""

            domain_idx = WikipediaDomain.determine_domain(description, abstract)
            tokens = tokenize(f"{description} {abstract}")
            word_indices = [vocab_map[t] for t in tokens if t in vocab_map]

            if word_indices:
                field.train_stream(
                    word_indices, domain_idx, amplify_delta=10, suppress_delta=2
                )
                trained_count += 1

            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1}/{train_samples} streamed samples...")

        duration = time.time() - start_time
        print(
            f"Training on {trained_count} valid samples completed in {duration:.2f}s!"
        )

        # Save weights to disk
        print("Saving model weights to disk...")
        field.save_weights(weights_file_path)
        print("Model saved successfully!\n")

    # 4. Interactive CLI Loop
    print("--------------------------------------------------------------------")
    print("Type any sentence or description below to classify it.")
    print(
        "The engine will route the context to the 5 specialized domain experts in real-time."
    )
    print("Type 'exit' or 'quit' to close the tool.")
    print("--------------------------------------------------------------------\n")

    while True:
        try:
            user_input = input("👉 Enter text: ")
        except (EOFError, KeyboardInterrupt):
            break

        trimmed = user_input.strip()
        if trimmed == "exit" or trimmed == "quit":
            break

        if not trimmed:
            continue

        tokens = tokenize(trimmed)
        field.reset_potentials()

        recognized_words = [token for token in tokens if token in vocab_map]
        if not recognized_words:
            print("  ⚠️  No valid tokens found.")
            continue

        print("  Expert Activations:")

        word_indices = [vocab_map[token] for token in recognized_words]
        field.process_stream(word_indices, training_mode=False)

        expert_potentials = field.get_potentials()
        predicted_idx = expert_potentials.index(max(expert_potentials))

        for idx in range(5):
            p = expert_potentials[idx]
            cat_name = WikipediaDomain.name(idx)
            print(f"    [{cat_name:22}]: {p:3}")

        winner = WikipediaDomain.name(predicted_idx)
        print(f"\n  🏆 Winning Category: **{winner.upper()}** 🏆\n")
        print("--------------------------------------------------------------------")

    print("\nThank you for using the Wikipedia Classifier CLI Tool! Goodbye!")


if __name__ == "__main__":
    main()
