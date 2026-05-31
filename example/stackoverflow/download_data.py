import os
import urllib.request
import tarfile
import csv
import time

def process_directory(base_dir, out_csv):
    classes = ["csharp", "java", "javascript", "python"]
    records = []
    
    for label, cls_name in enumerate(classes):
        cls_dir = os.path.join(base_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
            
        for fname in os.listdir(cls_dir):
            if fname.endswith(".txt"):
                filepath = os.path.join(cls_dir, fname)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().replace("\n", " ").replace("\r", "")
                    records.append([label, text])
                    
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(records)
    
    return len(records)

def main():
    print("====================================================================")
    print(" Downloading StackOverflow Questions Dataset (16,000 Samples)")
    print("====================================================================\n")

    url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    tar_path = os.path.join(data_dir, "stack_overflow_16k.tar.gz")
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print("Dataset already exists!")
        return

    print(f"Downloading from {url}...")
    start_time = time.time()
    urllib.request.urlretrieve(url, tar_path)
    print(f"Download complete in {time.time() - start_time:.1f} seconds!\n")

    print("Extracting dataset...")
    start_time = time.time()
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print(f"Extraction complete in {time.time() - start_time:.1f} seconds!\n")
    
    print("Converting text files to CSV...")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    train_count = process_directory(train_dir, train_csv)
    test_count = process_directory(test_dir, test_csv)
    
    print(f"Parsed {train_count} training samples and {test_count} test samples.")
    
    os.remove(tar_path)
    print("\nDataset is ready at:")
    print(f"  - {train_csv}")
    print(f"  - {test_csv}")
    print("\nRun `mise run examples:stackoverflow:run` to train the model!")

if __name__ == "__main__":
    main()
