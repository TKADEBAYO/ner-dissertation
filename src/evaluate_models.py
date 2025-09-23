import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

METRICS_DIR = "outputs/metrics"
OUTPUT_CSV = os.path.join(METRICS_DIR, "model_comparison.csv")
OUTPUT_DIR_GRAPHS = "outputs/graphs"
os.makedirs(OUTPUT_DIR_GRAPHS, exist_ok=True)

def parse_metrics_file(filepath):
    """Parse a metrics file and return precision, recall, and F1."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        metrics = {}

        for line in lines:
            if "," in line and "Precision" in line:
               
                parts = line.strip().split(",")
                for part in parts:
                    if ":" in part:
                        key, value = part.strip().split(":", 1)
                        metrics[key.strip().lower()] = float(value.strip())
            elif ":" in line:
                key, value = line.strip().split(":", 1)
                try:
                    metrics[key.strip().lower()] = float(value.strip())
                except:
                    pass

        return (
            metrics.get("eval_precision") or metrics.get("precision"),
            metrics.get("eval_recall") or metrics.get("recall"),
            metrics.get("eval_f1") or metrics.get("f1")
        )
    except Exception as e:
        print(f"⚠️ Could not parse {filepath}: {e}")
        return None, None, None

def evaluate_all_models():
    rows = []
    for filename in os.listdir(METRICS_DIR):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(METRICS_DIR, filename)
        name_parts = filename.replace(".txt", "").split("_")
        dataset = name_parts[0]
        model_type = name_parts[1] if len(name_parts) > 1 else "unknown"

        if model_type in ["finetune", "finetuned"]:
            model_label = "Fine-tuned BERT"
        elif model_type == "hf":
            model_label = "HuggingFace"
        elif model_type == "spacy":
            model_label = "spaCy"
        else:
            model_label = model_type.capitalize()

        precision, recall, f1 = parse_metrics_file(filepath)
        rows.append({
            "dataset": dataset,
            "model": model_label,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    df = pd.DataFrame(rows)
    df.sort_values(by=["dataset", "model"], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved comparison table to {OUTPUT_CSV}")

    
    datasets = sorted(df["dataset"].unique())
    models = df["model"].unique()

   
    plt.figure(figsize=(10, 6))
    for model in models:
        subset = df[df["model"] == model]
        plt.plot(subset["dataset"], subset["f1"], marker="o", label=model)
    plt.xlabel("Dataset")
    plt.ylabel("F1 Score")
    plt.title("Model F1 Score Comparison Across Datasets")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_GRAPHS, "model_comparison_f1.png"))
    plt.close()
    print(f"Saved F1 comparison plot to {os.path.join(OUTPUT_DIR_GRAPHS, 'model_comparison_f1.png')}")

    
    plt.figure(figsize=(10, 6))
    for model in models:
        subset = df[df["model"] == model]
        plt.plot(subset["dataset"], subset["precision"], marker="o", label=model)
    plt.xlabel("Dataset")
    plt.ylabel("Precision")
    plt.title("Model Precision Comparison Across Datasets")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_GRAPHS, "model_comparison_precision.png"))
    plt.close()
    print(f"📊 Saved Precision comparison plot to {os.path.join(OUTPUT_DIR_GRAPHS, 'model_comparison_precision.png')}")

   
    plt.figure(figsize=(10, 6))
    for model in models:
        subset = df[df["model"] == model]
        plt.plot(subset["dataset"], subset["recall"], marker="o", label=model)
    plt.xlabel("Dataset")
    plt.ylabel("Recall")
    plt.title("Model Recall Comparison Across Datasets")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_GRAPHS, "model_comparison_recall.png"))
    plt.close()
    print(f"📊 Saved Recall comparison plot to {os.path.join(OUTPUT_DIR_GRAPHS, 'model_comparison_recall.png')}")

    
    plt.figure(figsize=(12, 7))
    width = 0.2
    x = np.arange(len(datasets))

    for i, model in enumerate(models):
        subset = df[df["model"] == model]
        
        vals = [subset[subset["dataset"] == ds]["f1"].values[0] if ds in subset["dataset"].values else 0 for ds in datasets]
        plt.bar(x + i*width, vals, width=width, label=model)

    plt.xticks(x + width, datasets)
    plt.xlabel("Dataset")
    plt.ylabel("F1 Score")
    plt.title("Model F1 Score per Dataset (Grouped Bar Chart)")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_GRAPHS, "model_comparison_f1_bars.png"))
    plt.close()
    print(f"📊 Saved F1 grouped bar chart to {os.path.join(OUTPUT_DIR_GRAPHS, 'model_comparison_f1_bars.png')}")

  
    fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)
    metrics = ["precision", "recall", "f1"]
    titles = ["Precision", "Recall", "F1 Score"]

    for ax, metric, title in zip(axes, metrics, titles):
        for model in models:
            subset = df[df["model"] == model]
            ax.plot(subset["dataset"], subset[metric], marker="o", label=model)
        ax.set_ylabel(title)
        ax.set_title(f"Model {title} Comparison Across Datasets")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Dataset")
    plt.tight_layout()
    combined_path = os.path.join(OUTPUT_DIR_GRAPHS, "model_comparison_all_metrics.png")
    plt.savefig(combined_path)
    plt.close()
    print(f"📊 Saved combined metrics comparison plot to {combined_path}")

if __name__ == "__main__":
    evaluate_all_models()
