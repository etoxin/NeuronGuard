"""
NeuronGuard: Diagnostics & Explainability Example

This example demonstrates how NeuronGuard's white-box associative memory
architecture allows for perfect transparency. Unlike black-box neural networks,
we can introspect exactly which words contributed to a prediction and by how much,
or extract the highest-weighted features for any class directly from memory!
"""

import os
from neuronguard import TextClassifier

def main():
    print("====================================================================")
    print("  NeuronGuard Diagnostics & Explainability Tool")
    print("====================================================================\n")

    # Toy dataset demonstrating clear class boundaries
    records = [
        (0, 'urgent bank account suspended verify immediately'),
        (0, 'claim your free prize money today winner'),
        (0, 'reset password security alert unusual login'),
        (1, 'meeting scheduled for tomorrow afternoon'),
        (1, 'project update quarter earnings report attached'),
        (1, 'team lunch on friday please RSVP'),
    ]

    classifier = TextClassifier(
        num_classes=2, 
        vocab_size=100, 
        class_names=['Phishing / Spam', 'Legitimate Work']
    )
    
    print("Training the TextClassifier...")
    classifier.fit_records(records, epochs=5)
    print("Training complete.\n")

    # --- Feature Extraction ---
    print("--- 1. Class Feature Extraction ---")
    print("We can directly ask the model to print its top memory triggers.")
    
    for class_idx in range(len(classifier.class_names)):
        classifier.print_class_features(class_idx, top_k=3)
    
    print("\n--- 2. Transparent Prediction Explanation ---")
    # A tricky sentence containing words from both classes
    tricky_text = "urgent meeting tomorrow to reset password security"
    
    print("We can ask the model to explain exactly how it arrived at a decision:")
    classifier.print_explanation(tricky_text)

    print("\n====================================================================")

if __name__ == "__main__":
    main()
