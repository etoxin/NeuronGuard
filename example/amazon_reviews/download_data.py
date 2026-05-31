import os
import urllib.request
import tarfile
import time

def download_and_extract():
    url = "https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    tar_path = os.path.join(data_dir, "amazon_review_polarity_csv.tgz")
    extract_path = os.path.join(data_dir)

    print("====================================================================")
    print(" Downloading Amazon Reviews Polarity Dataset (3.6 Million Samples)")
    print("====================================================================\n")

    if not os.path.exists(tar_path):
        print(f"Downloading from {url}...")
        print("This is a 688MB file, please wait...")
        start = time.time()
        urllib.request.urlretrieve(url, tar_path)
        print(f"Download complete in {time.time() - start:.1f} seconds!")
    else:
        print("Dataset tar.gz already exists.")

    csv_dir = os.path.join(extract_path, "amazon_review_polarity_csv")
    if not os.path.exists(csv_dir):
        print("\nExtracting dataset...")
        start = time.time()
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Extraction complete in {time.time() - start:.1f} seconds!")
    else:
        print("Dataset is already extracted.")
        
    print("\nDataset is ready at:")
    print(f"  - {os.path.join(csv_dir, 'train.csv')}")
    print(f"  - {os.path.join(csv_dir, 'test.csv')}")
    print("\nRun `mise run examples:amazon_reviews:run` to train the model!")

if __name__ == "__main__":
    download_and_extract()
