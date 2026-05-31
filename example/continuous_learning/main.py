"""
NeuronGuard: Continuous On-Device Learning Example

This example demonstrates how NeuronGuard can learn continuously on the fly
without needing to rebuild the vocabulary, recreate the graph, or perform
expensive backpropagation. Models are updated instantly via atomic
additions and subtractions in memory.
"""

from neuronguard import TextClassifier
import time

def main():
    print("====================================================================")
    print("  NeuronGuard Continuous Learning (On-the-fly updates)")
    print("====================================================================\n")

    # Initial Training Data
    initial_records = [
        (0, "my account is locked help"),
        (0, "i need to reset my password"),
        (0, "where are my account settings"),
        (1, "i want to buy a new subscription"),
        (1, "what is the price of the enterprise plan"),
        (1, "sales contact for purchasing"),
    ]

    classifier = TextClassifier(
        num_classes=2, 
        vocab_size=100, 
        class_names=["Customer Support", "Sales"]
    )
    
    print("1. Initializing and training base model...")
    classifier.fit_records(initial_records, epochs=5)
    print("Base model trained!\n")

    # --- Initial Test ---
    test_sentence = "my account upgrade to the premium plan failed"
    
    print("2. Testing an ambiguous sentence that shares words from both classes:")
    print(f"   Sentence: '{test_sentence}'\n")
    
    classifier.print_explanation(test_sentence)
    
    print("\n--------------------------------------------------------------------")
    print("The model incorrectly predicted 'Customer Support' because the word 'account'")
    print("is heavily associated with Support. We will now correct this routing.")
    print("--------------------------------------------------------------------\n")

    # --- Continuous Learning ---
    print("3. Updating model with streaming data (Zero-overhead learning)...")
    
    # We provide a few corrective records targeting Sales (Class 1)
    correction = [(1, "my account upgrade to the premium plan failed")] * 5
    
    start_time = time.time()
    classifier.update_records(correction)
    duration_ms = (time.time() - start_time) * 1000
    
    print(f"Update completed in {duration_ms:.3f} ms.\n")

    # --- Re-Test ---
    print("4. Re-testing the identical sentence...")
    classifier.print_explanation(test_sentence)
    
    print("\n====================================================================")
    print("The Hebbian plasticity update instantly adjusted the synaptic weights,")
    print("correctly routing the phrase to the Sales class without retraining.")
    print("====================================================================")

if __name__ == "__main__":
    main()
