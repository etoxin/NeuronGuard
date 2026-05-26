import csv
import os
import re
import time

import neuronguard as ng


def tokenize(text):
    # Simple text tokenizer and cleaner
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    return text.split()


def main():
    print("====================================================================")
    print("📰 AG News 120,000 Dataset Classification PoC (Python) 📰")
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
    }

    print("--- Step 1: Building Vocabulary from 120,000 Training Samples ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file_path = os.path.join(script_dir, "data", "train.csv")

    # Count word frequencies per category to build a discriminative vocabulary
    word_counts = {}

    with open(train_file_path, mode="r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for record in rdr:
            class_index = int(record[0])
            cat_idx = class_index - 1
            title = record[1]
            description = record[2]

            full_text = f"{title} {description}"
            tokens = tokenize(full_text)

            for token in tokens:
                if len(token) > 2 and token not in stop_words:
                    if token not in word_counts:
                        word_counts[token] = [0, 0, 0, 0]
                    word_counts[token][cat_idx] += 1

    # Sort words by total frequency and keep the top 1,000 most frequent words
    word_list = []
    for word, counts in word_counts.items():
        total_count = sum(counts)
        word_list.append((word, counts, total_count))

    word_list.sort(key=lambda x: x[2], reverse=True)
    vocab_size = 1000
    final_vocab = word_list[:vocab_size]

    # Map words to their index in the vocabulary
    vocab_map = {word: idx for idx, (word, _, _) in enumerate(final_vocab)}

    num_words = len(final_vocab)
    num_experts = 4
    field_size = num_words + num_experts

    print("Vocabulary built successfully!")
    print("  Top 1,000 most frequent words selected.")
    print(f"  Total Neuron Field Size: {field_size} neurons (64 bytes each)\n")

    # Initialize the NeuronGuardField
    field = ng.NeuronGuardField(sensory_count=num_words, motor_count=num_experts)

    # Configure word neurons to target their respective experts (0..4)
    # Each word targets the expert (category) in which it occurs most frequently!
    for i in range(num_words):
        counts = final_vocab[i][1]
        max_idx = 0
        max_val = 0
        for idx, val in enumerate(counts):
            if val > max_val:
                max_val = val
                max_idx = idx

        field.train_stream([i], max_idx, 15, 0)

    print("--- Step 2: Training on 120,000 Samples (Trainer Mode) ---")
    print("Applying the Guard feedback loop over the entire dataset...")

    start_time = time.time()
    sample_count = 0

    with open(train_file_path, mode="r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for record in rdr:
            class_index = int(record[0])
            cat_idx = class_index - 1
            title = record[1]
            description = record[2]

            full_text = f"{title} {description}"
            tokens = tokenize(full_text)

            word_indices = [vocab_map[token] for token in tokens if token in vocab_map]
            if word_indices:
                field.train_stream(word_indices, cat_idx, 5, 15)

            sample_count += 1
            if sample_count % 30000 == 0:
                print(f"  Processed {sample_count}/120,000 samples...")

    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f}s!")

    print("\n--- Step 3: Evaluating on 7,600 Test Samples (Run Mode) ---")
    test_file_path = os.path.join(script_dir, "data", "test.csv")

    correct_predictions = 0
    total_predictions = 0
    confusion_matrix = [[0] * 4 for _ in range(4)]

    with open(test_file_path, mode="r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for record in rdr:
            class_index = int(record[0])
            actual_idx = class_index - 1
            title = record[1]
            description = record[2]

            full_text = f"{title} {description}"
            tokens = tokenize(full_text)

            # Reset expert potentials
            field.reset_potentials()

            word_indices = [vocab_map[token] for token in tokens if token in vocab_map]
            if word_indices:
                field.process_stream_sync(word_indices)

            # Determine which expert has the highest potential
            expert_potentials = field.get_potentials()
            predicted_idx = expert_potentials.index(max(expert_potentials))

            confusion_matrix[actual_idx][predicted_idx] += 1

            if predicted_idx == actual_idx:
                correct_predictions += 1
            total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print("Evaluation Complete!")
    print(f"  Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

    print("\n--- Confusion Matrix ---")
    print("  Actual \\ Predicted | World | Sports | Business | Sci/Tech")
    print("  -------------------|-------|--------|----------|---------")
    cat_names = [
        "  World News        ",
        "  Sports            ",
        "  Business          ",
        "  Sci/Tech          ",
    ]
    for i in range(4):
        print(
            f"{cat_names[i]} | {confusion_matrix[i][0]:5} | {confusion_matrix[i][1]:6} | {confusion_matrix[i][2]:8} | {confusion_matrix[i][3]:8}"
        )
    print("====================================================================")


if __name__ == "__main__":
    main()
