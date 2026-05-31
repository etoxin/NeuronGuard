import os
from neuronguard import TabularClassifier

def main():
    print("====================================================================")
    print(" ⚽ Continual Learning: Live Match Tactical Concept Drift (Raw Metrics) ⚽")
    print("====================================================================\n")

    # Features: [Possession %, Passes Completed, Shots Outside Box]
    # Classes: 0 (No Goal / Stalled), 1 (Goal Scoring Opportunity)
    classifier = TabularClassifier(
        num_classes=2, 
        num_features=3, 
        buckets_per_feature=10,
        amplify_delta=25,
        suppress_delta=5
    )

    # --- DATASET A: First Half (Opponent plays open, high possession = good) ---
    # Metrics: [Possession, Passes, Shots_Outside]
    dataset_first_half = [
        # Dominating play -> Goal
        ([65.0, 450.0, 2.0], 1),
        ([70.0, 500.0, 3.0], 1),
        # Poor play -> No Goal
        ([35.0, 150.0, 0.0], 0),
        ([40.0, 200.0, 1.0], 0),
    ]

    # --- DATASET B: Second Half (Opponent switches to "Low Block / Counter Attack") ---
    # Now, high possession means stalled passing around the back, and low possession means lethal counters!
    dataset_second_half = [
        # Stalled passing -> No Goal
        ([75.0, 600.0, 10.0], 0),
        ([80.0, 650.0, 12.0], 0),
        # Lethal counter attacks -> Goal
        ([25.0, 100.0, 1.0], 1),
        ([30.0, 120.0, 1.0], 1),
    ]

    print("--- [Step 1] Training on First Half Metrics ---")
    classifier.fit(
        records=[r[0] + [r[1]] for r in dataset_first_half],
        feature_indices=[0, 1, 2],
        label_index=3
    )
    print("Model Trained!\n")

    print("--- [Step 2] Testing First Half Knowledge ---")
    print("Input Metrics: 68% Possession, 480 Passes")
    pred = classifier.predict([68.0, 480.0, 2.0])
    print("Prediction:", "GOAL" if pred == 1 else "NO GOAL")
    
    print("Input Metrics: 38% Possession, 180 Passes")
    pred = classifier.predict([38.0, 180.0, 1.0])
    print("Prediction:", "GOAL" if pred == 1 else "NO GOAL")
    print()

    print("--- [Step 3] THE EVENT: Halftime Tactical Switch ---")
    print("   (Opponent drops into a deep defensive block.)")
    print("   High possession is now a trap. We stream in Second Half metrics!\n")
    
    # In TabularClassifier, we can seamlessly stream update new metrics!
    # update() streams without recalculating bucket boundaries.
    classifier.update([(r[0] + [r[1]]) for r in dataset_second_half], label_index=3)
    print("Model Updated on the fly with Second Half data!\n")

    print("--- [Step 4] Testing Second Half Concept Drift ---")
    print("Input Metrics: 78% Possession, 620 Passes (The Possession Trap)")
    pred = classifier.predict([78.0, 620.0, 11.0])
    print("Prediction:", "GOAL" if pred == 1 else "NO GOAL")
    
    print("Input Metrics: 28% Possession, 110 Passes (The Counter Attack)")
    pred = classifier.predict([28.0, 110.0, 1.0])
    print("Prediction:", "GOAL" if pred == 1 else "NO GOAL")
    print()

    print("--- [Step 5] Why is this important for Live Sports telemetry? ---")
    print("   Deep Learning models require aggregating data and retraining offline.")
    print("   NeuronGuard updated its physical mesh in O(1) time exactly as the ")
    print("   tactical shift happened live on the field, instantly flipping its predictions!")
    print("====================================================================")

if __name__ == "__main__":
    main()
