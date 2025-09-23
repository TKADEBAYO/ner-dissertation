import os
import json
from datasets import load_from_disk
from datasets import DatasetDict
import spacy

nlp = spacy.load("en_core_web_sm")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def convert_and_save(dataset, dataset_name, split_name):
    output_path = f"data/processed/{dataset_name}"
    ensure_dir(output_path)
    out_file = os.path.join(output_path, f"{split_name}.jsonl")
    num_skipped = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for example in dataset:
            try:
                
                if "tokens" in example and "ner_tags" in example:
                    json_obj = {
                        "tokens": example["tokens"],
                        "ner_tags": example["ner_tags"]
                    }

                
                elif "passages" in example and "entities" in example:
                   

                    text = " ".join(p["text"][0] for p in example["passages"])
                    doc = nlp(text)

                    tokens = [token.text for token in doc]
                    token_start = [token.idx for token in doc]
                    token_end = [token.idx + len(token.text) for token in doc]
                    labels = ["O"] * len(tokens)

                    for entity in example["entities"]:
                        start = entity["offsets"][0][0]
                        end = entity["offsets"][0][1]
                        label = entity["type"]

                        inside = False
                        for i, (s, e) in enumerate(zip(token_start, token_end)):
                            if s >= start and e <= end:
                                if not inside:
                                    labels[i] = "B-" + label
                                    inside = True
                                else:
                                    labels[i] = "I-" + label

                    
                    if any(tag != "O" for tag in labels):
                        json_obj = {"tokens": tokens, "ner_tags": labels}
                    else:
                        num_skipped += 1
                        continue

                else:
                    num_skipped += 1
                    continue

                f.write(json.dumps(json_obj) + "\n")

            except Exception:
                num_skipped += 1
                continue

    print(f"Saved {split_name} for {dataset_name} to {out_file} ({len(dataset) - num_skipped} samples, skipped: {num_skipped})")


def process_dataset(dataset_name, splits):
    print(f"\n🔄 Processing {dataset_name}...")
    for split in splits:
        try:
            dataset = load_from_disk(f"data/raw/{dataset_name}/{split}")
            convert_and_save(dataset, dataset_name, split)
        except FileNotFoundError:
            print(f"⚠️  {split} split not found for {dataset_name}, skipping...")

def process_jnlpba_custom():
    print("\n🔄 Processing JNLPBA with custom validation/test split...")

    
    val_path = "data/raw/jnlpba/validation"
    val_ds = load_from_disk(val_path)
    split_ds = val_ds.train_test_split(test_size=0.5, seed=42)
    validation_set = split_ds["train"]
    test_set = split_ds["test"]

   
    train_set = load_from_disk("data/raw/jnlpba/train")

    
    convert_and_save(train_set, "jnlpba", "train")
    convert_and_save(validation_set, "jnlpba", "validation")
    convert_and_save(test_set, "jnlpba", "test")

def main():
    datasets = {
        "conll2003": ["train", "validation", "test"],
        "wnut17": ["train", "validation", "test"],
        "bc5cdr": ["train", "validation", "test"],
    }

    for name, splits in datasets.items():
        process_dataset(name, splits)

    process_jnlpba_custom()
    print("\nAll datasets processed and saved.")

if __name__ == "__main__":
    main()
