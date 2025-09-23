import os
import matplotlib.pyplot as plt
import numpy as np

METRICS_DIR = "outputs/metrics"
OUTPUT_PLOT = "outputs/graphs/error_estimates_from_metrics_horizontal.png"

def parse_metrics_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if "," in content and "Precision" in content:
            metrics = {}
            parts = content.strip().split(",")
            for part in parts:
                if ":" in part:
                    key, val = part.strip().split(":", 1)
                    metrics[key.strip().lower()] = float(val.strip())
            return metrics.get("precision"), metrics.get("recall")
        else:
            return None, None
    except Exception as e:
        print(f"⚠️ Failed to parse {filepath}: {e}")
        return None, None

def run_error_plot_from_metrics():
    datasets = []
    fps = [] 
    fns = []  

    for file in os.listdir(METRICS_DIR):
        if not file.endswith(".txt") or "finetune" in file:
            continue

        dataset = file.split("_")[0]
        model_type = file.split("_")[1].replace(".txt", "")

        if model_type not in ["hf", "spacy"]:
            continue

        filepath = os.path.join(METRICS_DIR, file)
        precision, recall = parse_metrics_file(filepath)

        if precision is not None and recall is not None:
            datasets.append(f"{dataset}_{model_type}")
            fps.append(1 - precision)
            fns.append(1 - recall)

    
    y = np.arange(len(datasets))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.barh(y - width/2, fps, height=width, label='Estimated False Positives (1 - Precision)', color='salmon')
    plt.barh(y + width/2, fns, height=width, label='Estimated False Negatives (1 - Recall)', color='skyblue')
    
    plt.xlabel("Estimated Error Rate")
    plt.ylabel("Dataset_Model")
    plt.yticks(y, datasets)
    plt.title("Estimated False Positive/Negative Rates from Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Horizontal error estimate plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    run_error_plot_from_metrics()
