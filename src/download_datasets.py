import os
import time
from datasets import load_dataset, DatasetDict

def save_split(dataset_name, split_name, split_data):
    """Save a dataset split to the appropriate folder."""
    output_dir = f"data/raw/{dataset_name}/{split_name}"
    os.makedirs(output_dir, exist_ok=True)
    split_data.save_to_disk(output_dir)
    print(f"Saved {dataset_name} [{split_name}] to {output_dir} ({len(split_data)} samples)")

def download_dataset(name, hf_path, splits, config=None):
    print(f"\nDownloading {name}...")
    try:
        if config:
            dataset = load_dataset(hf_path, name=config)
        else:
            dataset = load_dataset(hf_path)

        for split in splits:
            if split in dataset:
                save_split(name, split, dataset[split])
            else:
                print(f"Split '{split}' not available for {name}, skipping...")

    except Exception as e:
        print(f"Error downloading {name}: {e}")

def download_all():
    start = time.time()
    print("Starting dataset download...")

    datasets = [
        {
            "name": "conll2003",
            "hf_path": "conll2003",
            "splits": ["train", "validation", "test"]
        },
        {
            "name": "wnut17",
            "hf_path": "wnut_17",
            "splits": ["train", "validation", "test"]
        },
        {
            "name": "bc5cdr",
            "hf_path": "bigbio/bc5cdr",
            "splits": ["train", "validation", "test"],
            "config": "bc5cdr_bigbio_kb" 
        },
        {
            "name": "jnlpba",
            "hf_path": "bigbio/jnlpba",
            "splits": ["train", "validation", "test"]
        }
    ]

    for ds in datasets:
        download_dataset(
            name=ds["name"],
            hf_path=ds["hf_path"],
            splits=ds["splits"],
            config=ds.get("config") 
        )

    end = time.time()
    duration = end - start
    print(f"\nAll downloads complete in {duration:.2f} seconds.")

if __name__ == "__main__":
    download_all()
