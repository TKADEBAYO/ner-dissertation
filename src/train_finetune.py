import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import classification_report
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


DATA_DIR = "data/processed"
MODEL_DIR = "models"
METRIC_DIR = "outputs/metrics"


MODEL_MAP = {
    "conll2003": "bert-base-cased",
    "wnut17": "bert-base-uncased",
    "bc5cdr": "bert-base-uncased",
    "jnlpba": "bert-base-uncased"
}

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_dataset_from_jsonl(dataset_name):
    dataset_path = os.path.join(DATA_DIR, dataset_name)
    dataset = DatasetDict({
        split: Dataset.from_list(load_jsonl(os.path.join(dataset_path, f"{split}.jsonl")))
        for split in ["train", "validation", "test"]
    })
    return dataset

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[label_seq[word_id]])
            else:
                label_ids.append(label2id[label_seq[word_id]])
            prev_word_id = word_id
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        p, l = [], []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                p.append(pred_id)
                l.append(label_id)
        true_predictions.append(p)
        true_labels.append(l)

    precision, recall, f1, _ = precision_recall_fscore_support(
        [item for sublist in true_labels for item in sublist],
        [item for sublist in true_predictions for item in sublist],
        average="weighted", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}

def main(dataset_name):
    print(f"\n🔧 Fine-tuning on {dataset_name}...")

    model_name = MODEL_MAP[dataset_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset_from_jsonl(dataset_name)
    all_labels = sorted({label for d in dataset["train"] for label in d["ner_tags"]})
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
    output_dir=f"{MODEL_DIR}/{dataset_name}",
    logging_dir=f"{MODEL_DIR}/{dataset_name}/logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to=[]  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(f"{MODEL_DIR}/{dataset_name}")
    tokenizer.save_pretrained(f"{MODEL_DIR}/{dataset_name}")

   
    print(f"\nEvaluating fine-tuned model on test set for {dataset_name}...")
    metrics = trainer.evaluate(tokenized_dataset["test"])

    output_file = os.path.join(METRIC_DIR, f"{dataset_name}_finetune.txt")
    os.makedirs(METRIC_DIR, exist_ok=True)
    with open(output_file, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k.capitalize()}: {v:.3f}\n")
    print(f"Metrics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: conll2003, wnut17, bc5cdr, jnlpba")
    args = parser.parse_args()
    main(args.dataset)
