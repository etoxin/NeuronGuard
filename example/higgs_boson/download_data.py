import os
import subprocess
import gzip
import shutil
import time

def main():
    print("====================================================================")
    print(" Downloading Higgs Boson Dataset (11 Million Samples, 28 Features)")
    print("====================================================================\n")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    gz_path = os.path.join(data_dir, "HIGGS.csv.gz")
    csv_path = os.path.join(data_dir, "HIGGS.csv")

    if os.path.exists(csv_path):
        print(f"Dataset already exists at {csv_path}!")
        return

    print(f"Downloading from {url}...")
    start_time = time.time()
    try:
        subprocess.run(["curl", "-L", "-o", gz_path, url], check=True)
    except subprocess.CalledProcessError:
        print("Download failed. Please check your internet connection or try again later.")
        return
    print(f"Download complete in {time.time() - start_time:.1f} seconds!\n")

    print("Extracting dataset (~2.6 GB)...")
    start_time = time.time()
    with gzip.open(gz_path, 'rb') as f_in:
        with open(csv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extraction complete in {time.time() - start_time:.1f} seconds!\n")
    
    os.remove(gz_path)
    
    # We will split it into train/test to show real evaluation
    print("Splitting into 10M Train / 1M Test...")
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    
    with open(csv_path, "r", encoding="utf-8") as fin:
        with open(train_csv, "w", encoding="utf-8") as ftrain, open(test_csv, "w", encoding="utf-8") as ftest:
            for i, line in enumerate(fin):
                if i < 10000000:
                    ftrain.write(line)
                else:
                    ftest.write(line)
                    
    os.remove(csv_path)

    print("\nDataset is ready at:")
    print(f"  - {train_csv}")
    print(f"  - {test_csv}")
    print("\nRun `mise run examples:higgs_boson:run` to train the model!")

if __name__ == "__main__":
    main()
