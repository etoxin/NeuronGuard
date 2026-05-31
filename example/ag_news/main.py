# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

from neuronguard import TextClassifier


def main():
    print("====================================================================")
    print("📰 AG News 120,000 Dataset Classification PoC (Python) 📰")
    print("====================================================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "data", "train.csv")
    test_path = os.path.join(script_dir, "data", "test.csv")

    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    classifier = TextClassifier(
        num_classes=4,
        vocab_size=1000,
        class_names=class_names,
    )

    # --- Training --------------------------------------------------------
    print("--- Training on 120,000 Samples (3 epochs) ---")

    start_time = time.time()
    classifier.fit(train_path, text_col=[1, 2], label_col=0, epochs=3)
    duration = time.time() - start_time

    print(f"Training completed in {duration:.2f}s!\n")

    # --- Evaluation ------------------------------------------------------
    print("--- Evaluating on 7,600 Test Samples ---")

    accuracy, report = classifier.evaluate(test_path, text_col=[1, 2], label_col=0)

    print(f"  Accuracy: {accuracy:.2f}%\n")
    print(report)
    print("====================================================================")


if __name__ == "__main__":
    main()
