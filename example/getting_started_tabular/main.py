"""
NeuronGuard: Getting Started - Tabular Classification

This example introduces the TabularClassifier high-level API for continuous
numerical data, demonstrating automatic bucketing and class-imbalance weighting.
"""

from neuronguard import TabularClassifier

def main():
    print("====================================================================")
    print("  Getting Started 2: Tabular Classification")
    print("====================================================================\n")

    print("The TabularClassifier automatically buckets continuous features and handles")
    print("class imbalances via weighting.\n")

    tab_clf = TabularClassifier(num_classes=2, num_features=2, buckets_per_feature=5)
    train_data = [
        (1.0, 2.0, 0),
        (1.1, 2.1, 0),
        (8.0, 9.0, 1),
        (8.1, 9.1, 1),
    ]
    
    # Fit with features at indices 0,1 and label at index 2
    print("Training TabularClassifier on 4 toy records...")
    tab_clf.fit(train_data, feature_indices=[0, 1], label_index=2, epochs=5)

    test_features = [1.0, 2.0]
    pred = tab_clf.predict(test_features)
    print(f"Input features: {test_features}")
    print(f"Prediction: Class {pred}")
    print(f"Raw Scores: {tab_clf.predict_scores(test_features)}\n")

if __name__ == "__main__":
    main()
