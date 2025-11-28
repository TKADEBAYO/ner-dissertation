import os
import json
from datasets import load_dataset
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
                # CASE 1 — Standard CoNLL-style datasets (tokens + ner_tags)
                if "tokens" in example and "ner_tags" in example:
                    json_obj = {
                        "tokens": example["tokens"],
                        "ner_tags": example["ner_tags"],
                    }
                    f.write(json.dumps(json_obj) + "\n")
                    continue

                # CASE 2 — TNER biomedical datasets (tokens + tags)
                if "tokens" in example and "tags" in example:
                    json_obj = {
                        "tokens": example["tokens"],
                        "ner_tags": example["tags"],
                    }
                    f.write(json.dumps(json_obj) + "\n")
                    continue

                # CASE 3 — Legacy BC5CDR-style datasets with passages/entities
                if "passages" in example and "entities" in example:
                    text = " ".join(p["text"][0] for p in example["passages"])
                    doc = nlp(text)

                    tokens = [t.text for t in doc]
                    starts = [t.idx for t in doc]
                    ends = [t.idx + len(t.text) for t in doc]
                    tags = ["O"] * len(tokens)

                    for ent in example["entities"]:
                        start, end = ent["offsets"][0]
                        label = ent["type"]
                        inside = False

                        for i, (s, e) in enumerate(zip(starts, ends)):
                            if s >= start and e <= end:
                                tags[i] = ("B-" if not inside else "I-") + label
                                inside = True

                    json_obj = {"tokens": tokens, "ner_tags": tags}
                    f.write(json.dumps(json_obj) + "\n")
                    continue

                # If nothing matches, skip
                num_skipped += 1

            except Exception:
                num_skipped += 1
                continue

    print(f"Saved {split_name} for {dataset_name} ({len(dataset) - num_skipped} samples, skipped {num_skipped})")

def preprocess_all():

    print("\n🔄 Loading datasets from HuggingFace...")

    datasets = {
        "conll2003": load_dataset("conll2003"),
        "wnut17": load_dataset("wnut_17"),
        "bc5cdr": load_dataset("tner/bc5cdr"),
        "jnlpba": load_dataset("jnlpba"),
    }

    for name, ds in datasets.items():
        print(f"\n🔄 Processing {name}...")
        for split in ds:
            convert_and_save(ds[split], name, split)

    print("\n🎉 All datasets processed successfully.")

if __name__ == "__main__":
    preprocess_all()
