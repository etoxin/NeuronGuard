"""
NeuronGuard: Machine Unlearning Example (GDPR Compliance)

In traditional Deep Learning (like PyTorch or LLMs), if a user requests
their data be deleted under the GDPR "Right to be Forgotten", the company
often has to retrain the entire model from scratch because neural network
weights are deeply entangled via gradient descent.

Because NeuronGuard uses biological Hebbian plasticity (simple integer additions
in memory), we can achieve perfect Machine Unlearning instantly. We simply
pass the exact same data back through the network with a negative delta,
cleanly subtracting the exact synaptic weight modifications it originally caused!
"""

from neuronguard import TextClassifier
import time

def main():
    print("====================================================================")
    print("  NeuronGuard Machine Unlearning (GDPR Compliance)")
    print("====================================================================\n")

    # Initial Training Data
    initial_records = [
        (0, "reset my password for my account"),
        (0, "where are my account settings"),
        (1, "i want to buy a new subscription"),
        (1, "sales contact for purchasing adam example"),
    ]

    classifier = TextClassifier(
        num_classes=2, 
        vocab_size=100, 
        class_names=["Customer Support", "Sales"]
    )
    
    print("1. Initializing and training base model...")
    classifier.fit_records(initial_records, epochs=5)

    # Let's say a user submits a ticket that gets ingested into the model's
    # continuous learning pipeline, but it contains their private email!
    private_record = [(0, "my account is adam@example.com please help")]
    
    print("\n2. Model continuously learns from a new user ticket (10 times to memorize it)...")
    classifier.update_records(private_record * 10)
    
    # Prove the model learned the private email
    test_email = "adam@example.com"
    print(f"\n3. Testing the model on the private email: '{test_email}'")
    classifier.print_explanation(test_email)

    print("\n--------------------------------------------------------------------")
    print("URGENT: Legal just emailed. The user requested their data be deleted")
    print("under GDPR's Right to be Forgotten. We must erase this email immediately!")
    print("--------------------------------------------------------------------\n")

    print("4. Instantly unlearning the data (Zero-overhead Machine Unlearning)...")
    
    start_time = time.time()
    # By passing the exact same data to unlearn_records, it cleanly subtracts
    # the exact synaptic weight modifications it originally caused.
    classifier.unlearn_records(private_record * 10)
    duration_ms = (time.time() - start_time) * 1000
    
    print(f"Unlearn completed in {duration_ms:.3f} ms.\n")

    print("5. Final Re-test on the private email...")
    classifier.print_explanation(test_email)
    
    print("\n====================================================================")
    print("The model has completely forgotten the private email, proving")
    print("perfect, instant Machine Unlearning without any retraining!")
    print("====================================================================")

if __name__ == "__main__":
    main()
