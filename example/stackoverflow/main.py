import os
import time
from neuronguard import TextClassifier

def main():
    print("====================================================================")
    print("📦 StackOverflow Tag Prediction PoC (Python) 📦")
    print("====================================================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(script_dir, "data", "train.csv")
    test_csv = os.path.join(script_dir, "data", "test.csv")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("Dataset not found! Run `mise run examples:stackoverflow:download_data` first.")
        return

    class_names = ["csharp", "java", "javascript", "python"]

    # We use a vocab size of 5000 and the standard discriminative vocab builder.
    # StackOverflow questions heavily rely on specific keywords (e.g. "pandas", "println", "console.log")
    # which NeuronGuard's Hebbian learning will naturally isolate!
    classifier = TextClassifier(
        num_classes=4, 
        vocab_size=5000, 
        class_names=class_names,
        amplify_delta=15,
        suppress_delta=5,
        use_hashed_bigrams=False
    )

    print("--- Training on 8,000 StackOverflow Questions (3 epochs) ---")
    start_time = time.time()
    
    # label_offset=0 because our script mapped them from 0 to 3.
    classifier.fit(
        train_file=train_csv, 
        text_col=1, 
        label_col=0, 
        epochs=10, 
        label_offset=0
    )
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s!\n")

    print("--- Evaluating on 8,000 Test Questions ---")
    start_time = time.time()
    
    accuracy, report = classifier.evaluate(
        test_file=test_csv, 
        text_col=1, 
        label_col=0,
        label_offset=0
    )
    
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f}s!\n")

    print(report)
    print("====================================================================\n")
    
    print("Let's look at what the model learned about 'java' vs 'python':\n")
    classifier.print_class_features(class_names.index("java"), top_k=5)
    classifier.print_class_features(class_names.index("python"), top_k=5)
    
    print("\nLet's test it on a live snippet:")
    snippet = "How do I reverse a list using list comprehension or [::-1]?"
    classifier.print_explanation(snippet)

if __name__ == "__main__":
    main()
