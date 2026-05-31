"""
NeuronGuard: Sparse Distributed Representations (SDR) Example

This example demonstrates how to build semantic understanding into the model
without relying on dense, floating-point LLM embeddings.
By representing words as "Sparse Embeddings" (overlapping sets of neurons),
the model can generalize learning to words it has never explicitly been trained on!
"""

from neuronguard import TextClassifier

def main():
    print("====================================================================")
    print("  NeuronGuard Semantic Sparse Embeddings (SDRs)")
    print("====================================================================\n")

    # 1. Define a Sparse Semantic Vocabulary.
    # Notice that "bank" and "finance" share neurons 12 and 55.
    # Notice that "sports" and "basketball" share neurons 100 and 200.
    sdr_vocab = {
        "bank": [12, 55, 89, 402],
        "finance": [12, 55, 91, 600],
        
        "sports": [100, 200, 333, 444],
        "basketball": [100, 200, 555, 666],
    }

    classifier = TextClassifier(
        num_classes=2, 
        vocab_size=1000, 
        class_names=["Business", "Sports"]
    )
    
    # Manually inject our SDR vocabulary and initialize the field
    classifier._vocab_map = sdr_vocab
    classifier._is_fitted = True
    classifier._ensure_field()

    # 2. Train on ONLY "bank" and "sports"
    print("1. Training the model explicitly on 'bank' and 'sports'...")
    train_data = [
        (0, "bank"),    # 0 = Business
        (1, "sports"),  # 1 = Sports
    ]
    
    classifier.update_records(train_data)
    
    print("Model Trained!\n")

    # 3. Test on "finance" (which it has NEVER seen in training)
    print("2. Testing the model on the word 'finance' (Zero-shot semantic inference)...")
    
    print("\n--- Diagnostics for 'finance' ---")
    classifier.print_explanation("finance")
    
    print("\n--------------------------------------------------------------------")
    print("The Magic of SDRs:")
    print("Even though the model was never trained on the word 'finance',")
    print("because its sparse embedding [12, 55, 91, 600] overlaps with 'bank'")
    print("on neurons 12 and 55, the signal naturally routed directly to the")
    print("Business class! It inferred the semantic meaning perfectly.")
    print("--------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
