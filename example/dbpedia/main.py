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


class DBpediaCategory:
    NAMES = [
        "Company",
        "Educational Institution",
        "Artist",
        "Athlete",
        "Office Holder",
        "Mean of Transportation",
        "Building",
        "Natural Place",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "Written Work",
    ]

    @staticmethod
    def name(index):
        return DBpediaCategory.NAMES[index]


def main():
    print("====================================================================")
    print("📚 DBpedia Ontology Interactive CLI Tool (Python) 📚")
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
    train_file_path = os.path.join(script_dir, "dbpedia_csv", "train.csv")
    weights_file_path = os.path.join(script_dir, "dbpedia_weights.bin")
    vocab_file_path = os.path.join(script_dir, "dbpedia_vocab.txt")

    vocab_size = 2000
    num_experts = 14
    field_size = vocab_size + num_experts

    vocab_map = {}
    vocab_list = []

    # Initialize the NeuronGuardField
    field = ng.NeuronGuardField(sensory_count=vocab_size, motor_count=num_experts)

    if os.path.exists(weights_file_path) and os.path.exists(vocab_file_path):
        print("Loading pre-trained model weights and vocabulary...")

        # Load vocabulary
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                word = line.strip()
                vocab_map[word] = idx
                vocab_list.append(word)

        # Load weights
        field.load_weights(weights_file_path)
        print("Model loaded successfully in < 1ms!\n")
    else:
        print("Pre-trained model not found. Starting training on 560,000 samples...")
        print(
            "(This will take about 20 seconds and will save the weights for instant future startups)\n"
        )

        if not os.path.exists(train_file_path):
            print("Error: Training dataset not found!")
            print("Please run 'mise run download_data' first to download the dataset.")
            return

        print("--- Step 1: Building Vocabulary from 560,000 Training Samples ---")
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
                            word_counts[token] = [0] * 14
                        word_counts[token][cat_idx] += 1

        word_list = []
        for word, counts in word_counts.items():
            total_count = sum(counts)
            word_list.append((word, counts, total_count))

        word_list.sort(key=lambda x: x[2], reverse=True)
        final_vocab = word_list[:vocab_size]

        # Save vocabulary to disk
        with open(vocab_file_path, "w", encoding="utf-8") as f:
            for word, _, _ in final_vocab:
                f.write(f"{word}\n")

        # Load into memory maps
        for idx, (word, _, _) in enumerate(final_vocab):
            vocab_map[word] = idx
            vocab_list.append(word)

        # Configure word neurons
        for i in range(vocab_size):
            counts = final_vocab[i][1]
            max_idx = 0
            max_val = 0
            for idx, val in enumerate(counts):
                if val > max_val:
                    max_val = val
                    max_idx = idx

            field.train_stream([i], max_idx, 15, 0)

        print("--- Step 2: Training on 560,000 Samples (Trainer Mode) ---")
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

                word_indices = [
                    vocab_map[token] for token in tokens if token in vocab_map
                ]
                if word_indices:
                    field.train_stream(word_indices, cat_idx, 5, 15)

                sample_count += 1
                if sample_count % 100000 == 0:
                    print(f"  Processed {sample_count}/560,000 samples...")

        duration = time.time() - start_time
        print(f"Training completed in {duration:.2f}s!")

        # Save weights to disk
        print("Saving model weights to disk for instant future startups...")
        field.save_weights(weights_file_path)
        print("Model saved successfully!\n")

    # 2. Interactive CLI Loop
    print("--------------------------------------------------------------------")
    print("Type any sentence or description below to classify it.")
    print("The engine will route the context to the 14 experts in real-time.")
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

        # Reset expert potentials
        field.reset_potentials()

        recognized_words = [token for token in tokens if token in vocab_map]
        if not recognized_words:
            print(
                "  ⚠️  None of the words were recognized in the 2,000-word vocabulary."
            )
            print("      Try using more general descriptive words!\n")
            continue

        print(f"  Recognized Vocab: {recognized_words}")
        print("  Expert Activations:")

        word_indices = [vocab_map[token] for token in recognized_words]
        field.process_stream(word_indices, training_mode=False)

        expert_potentials = field.get_potentials()
        predicted_idx = expert_potentials.index(max(expert_potentials))

        for idx in range(14):
            p = expert_potentials[idx]
            cat_name = DBpediaCategory.name(idx)
            if len(cat_name) > 22:
                cat_name = cat_name[:22]
            bar_len = max(0, p)
            print(f"    [{cat_name:22}]: {p:3} {'*' * bar_len}")

        winner = DBpediaCategory.name(predicted_idx)
        print(f"\n  🏆 Winning Category: **{winner.upper()}** 🏆\n")
        print("--------------------------------------------------------------------")

    print("\nThank you for using the DBpedia Ontology CLI Tool! Goodbye!")


if __name__ == "__main__":
    main()
