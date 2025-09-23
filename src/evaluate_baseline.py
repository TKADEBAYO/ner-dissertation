import os
import json
import spacy
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


LABEL_MAPS = {
    "conll2003": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"],
    "wnut17": ["O", "B-corporation", "I-corporation", "B-creative-work", "I-creative-work",
               "B-group", "I-group", "B-location", "I-location", "B-person", "I-person",
               "B-product", "I-product"],
    "bc5cdr": ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"],
    "jnlpba": ["O", "B-DNA", "I-DNA", "B-protein", "I-protein", "B-cell_line", "I-cell_line",
               "B-cell_type", "I-cell_type", "B-RNA", "I-RNA"]
}

def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def evaluate(true_labels, pred_labels):
    flat_true = [l for seq in true_labels for l in seq]
    flat_pred = [l for seq in pred_labels for l in seq]
    p, r, f1, _ = precision_recall_fscore_support(flat_true, flat_pred, average="weighted", zero_division=0)
    return round(p, 3), round(r, 3), round(f1, 3), classification_report(flat_true, flat_pred, output_dict=True, zero_division=0)

def evaluate_spacy(dataset, dataset_name):
    print("🔍 Evaluating with spaCy...")
    nlp = spacy.load("en_core_web_sm")
    all_preds, all_trues = [], []

    for item in tqdm(dataset):
        original_tokens = item["tokens"]
        gold_ids = item["ner_tags"]
        gold_labels = [LABEL_MAPS[dataset_name][i] if isinstance(i, int) else i for i in gold_ids]

        text = " ".join(original_tokens)
        doc = nlp(text)

        spacy_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        pred_tags = ["O"] * len(original_tokens)
        char_idx = 0
        for i, orig_token in enumerate(original_tokens):
            while char_idx < len(text) and text[char_idx].isspace():
                char_idx += 1

            token_start = char_idx
            token_end = token_start + len(orig_token)

            for ent_start, ent_end, ent_label in spacy_entities:
                if token_start >= ent_start and token_end <= ent_end:
                    prefix = "B-" if pred_tags[i] == "O" else "I-"
                    pred_tags[i] = prefix + ent_label
                    break

            char_idx = token_end

        if len(gold_labels) != len(pred_tags):
            continue

        all_preds.append(pred_tags)
        all_trues.append(gold_labels)

    return evaluate(all_trues, all_preds)

def evaluate_hf(dataset, model_name, dataset_name):
    print(f"🤗 Evaluating with HuggingFace model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    nlp_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1  # CPU (-1) or GPU (0)
    )

    all_preds, all_trues = [], []

    for item in tqdm(dataset):
        tokens = item["tokens"]
        text = " ".join(tokens)

        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        
        results = nlp_pipe(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))

        pred_tags = ["O"] * len(tokens)
        for r in results:
            word = r['word'].lower()
            label = r['entity_group']
            for i, token in enumerate(tokens):
                if word in token.lower() or token.lower() in word:
                    prefix = "B-" if pred_tags[i] == "O" else "I-"
                    pred_tags[i] = prefix + label

        gold_ids = item["ner_tags"]
        gold_labels = [LABEL_MAPS[dataset_name][i] if isinstance(i, int) else i for i in gold_ids]

        all_preds.append(pred_tags)
        all_trues.append(gold_labels)

    return evaluate(all_trues, all_preds)



def run_evaluation():
    datasets = {
        "conll2003": "dslim/bert-base-NER",
        "wnut17": "Jean-Baptiste/roberta-large-ner-english",
        "bc5cdr": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "jnlpba": "kamalkraj/bioelectra-base-discriminator-pubmed",
    }

    spacy_results = []
    hf_results = []
    detailed_rows = []

    os.makedirs("outputs/metrics", exist_ok=True)

    for dataset, model_name in datasets.items():
        print(f"\n📊 Evaluating dataset: {dataset}")
        test_path = f"data/processed/{dataset}/test.jsonl"

        if not os.path.exists(test_path):
            print(f"❌ Test set not found for {dataset}, skipping...")
            continue

        data = read_jsonl(test_path)

        # spaCy
        spacy_p, spacy_r, spacy_f1, spacy_report = evaluate_spacy(data, dataset)
        spacy_results.append({
            "Dataset": dataset, "Tool": "spaCy",
            "Precision": spacy_p, "Recall": spacy_r, "F1": spacy_f1
        })
        with open(f"outputs/metrics/{dataset}_spacy.txt", "w") as f:
            f.write(f"Precision: {spacy_p}, Recall: {spacy_r}, F1: {spacy_f1}\n")

        for label in spacy_report:
            if label.startswith("B-") or label.startswith("I-"):
                detailed_rows.append({
                    "Dataset": dataset, "Tool": "spaCy", "Entity": label,
                    "Precision": round(spacy_report[label]["precision"], 3),
                    "Recall": round(spacy_report[label]["recall"], 3),
                    "F1": round(spacy_report[label]["f1-score"], 3)
                })

    
        hf_p, hf_r, hf_f1, hf_report = evaluate_hf(data, model_name, dataset)
        hf_results.append({
            "Dataset": dataset, "Tool": "HuggingFace",
            "Precision": hf_p, "Recall": hf_r, "F1": hf_f1
        })
        with open(f"outputs/metrics/{dataset}_hf.txt", "w") as f:
            f.write(f"Precision: {hf_p}, Recall: {hf_r}, F1: {hf_f1}\n")

        for label in hf_report:
            if label.startswith("B-") or label.startswith("I-"):
                detailed_rows.append({
                    "Dataset": dataset, "Tool": "HuggingFace", "Entity": label,
                    "Precision": round(hf_report[label]["precision"], 3),
                    "Recall": round(hf_report[label]["recall"], 3),
                    "F1": round(hf_report[label]["f1-score"], 3)
                })

    pd.DataFrame(spacy_results + hf_results).to_csv("outputs/metrics/model_comparison.csv", index=False)
    pd.DataFrame(detailed_rows).to_csv("outputs/metrics/model_comparison_detailed.csv", index=False)

    print("\nAll evaluations complete.")
    print("Results saved in: outputs/metrics/")

if __name__ == "__main__":
    run_evaluation()
