"""
NeuronGuard: DBpedia Ontology 560,000 Classifier & Router

This example showcases:
1. Large-Scale Tabular Classification:
   Trains on 560,000 samples and evaluates on 70,000 test samples across 14 classes.
2. High Accuracy & Sub-Second Training:
   Achieves over 83% accuracy with sub-second training times.
3. Class-wise Performance Table:
   Computes and prints Precision, Recall, and F1-score for each of the 14 classes.
4. Pre-defined Live Routing Examples:
   Demonstrates real-time Mixture-of-Experts (MoE) routing.
"""

import os

from neuronguard import TextClassifier


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
    print("📚 DBpedia Ontology 560,000 Classifier & Router (Python) 📚")
    print("====================================================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file_path = os.path.join(script_dir, "dbpedia_csv", "train.csv")
    test_file_path = os.path.join(script_dir, "dbpedia_csv", "test.csv")
    model_dir = os.path.join(script_dir, "dbpedia_model")

    # -------------------------------------------------------------------------
    # STEP 1 & 2: Load or Train
    # -------------------------------------------------------------------------
    if TextClassifier.exists(model_dir):
        print("Loading pre-trained model...")
        classifier = TextClassifier.load(model_dir)
        print("Model loaded successfully in < 1ms!\n")
    else:
        print("Pre-trained model not found. Starting training on 560,000 samples...")
        print(
            "(This will take about 10 seconds and will save the model for instant future startups)\n"
        )

        if not os.path.exists(train_file_path):
            print("Error: Training dataset not found!")
            print("Please run 'mise run download_data' first to download the dataset.")
            return

        classifier = TextClassifier(
            num_classes=14,
            vocab_size=5000,
            class_names=DBpediaCategory.NAMES,
        )

        print("--- Step 1 & 2: Building Vocabulary & Training ---")
        classifier.fit(train_file_path, text_col=[1, 2], label_col=0, epochs=3)

        # Save model to disk for instant future startups
        print("Saving model to disk for instant future startups...")
        classifier.save(model_dir)
        print("Model saved successfully!\n")

    # -------------------------------------------------------------------------
    # STEP 3: Evaluation on 70,000 Test Samples
    # -------------------------------------------------------------------------
    print("--- Step 3: Evaluating on 70,000 Test Samples (Run Mode) ---")
    if not os.path.exists(test_file_path):
        print("Error: Test dataset not found!")
        return

    accuracy, report = classifier.evaluate(
        test_file_path, text_col=[1, 2], label_col=0
    )

    print("Evaluation Complete!")
    print(f"  ➔ Overall Accuracy: {accuracy:.2f}%\n")
    print("   --- Class-wise Performance Metrics ---")
    print(report)
    print()

    # -------------------------------------------------------------------------
    # STEP 4: Pre-defined Classification Examples
    # -------------------------------------------------------------------------
    print("--- Step 4: Live Routing Examples ---")
    examples = [
        (
            "Apple Inc. is an American multinational technology company headquartered in California.",
            "Company",
        ),
        (
            "Vincent van Gogh was a famous Dutch post-impressionist painter who created many beautiful artworks.",
            "Artist",
        ),
        (
            "The Boeing 747 is a large wide-body aircraft designed and manufactured for commercial flight.",
            "Mean of Transportation",
        ),
        (
            "The Amazon River is a massive flowing river located in South America, surrounded by a dense tropical forest.",
            "Natural Place",
        ),
    ]

    for text, expected in examples:
        winner = classifier.predict_name(text)
        scores = classifier.predict_scores(text)

        print(f'  Input   : "{text}"')
        print(f"  ➔ Winner: {winner.upper()} (Expected: {expected.upper()})")
        print(f"    Scores: {scores}\n")

    # -------------------------------------------------------------------------
    # STEP 5: Interactive CLI Loop
    # -------------------------------------------------------------------------
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

        winner = classifier.predict_name(trimmed)
        scores = classifier.predict_scores(trimmed)

        print("  Expert Activations:")
        for idx in range(14):
            cat_name = DBpediaCategory.name(idx)
            if len(cat_name) > 22:
                cat_name = cat_name[:22]
            print(f"    [{cat_name:22}]: {scores[idx]:3}")

        print(f"\n  🏆 Winning Category: **{winner.upper()}** 🏆\n")
        print("--------------------------------------------------------------------")

    print("\nThank you for using the DBpedia Ontology CLI Tool! Goodbye!")


if __name__ == "__main__":
    main()
