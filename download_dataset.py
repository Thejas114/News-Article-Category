"""Downloads AG News (train.csv, test.csv) into the 'data' folder."""
import os
import urllib.request

BASE_URL = ("https://raw.githubusercontent.com/mhjabreel/CharCNN_Keras/"
            "master/data/ag_news_csv/")
FILES = ["train.csv", "test.csv"]
DATA_DIR = "data"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            print(f"[skip] {path} already exists")
            continue
        print(f"[download] {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, path)
        print(f"[done] saved to {path}")
    print("\nDataset is ready. Next step: python train_model.py")


if __name__ == "__main__":
    main()