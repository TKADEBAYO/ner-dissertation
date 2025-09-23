import os
import re
import pandas as pd
import matplotlib.pyplot as plt

METRICS_FILE = "outputs/metrics/model_comparison_detailed.csv"
OUTPUT_DIR = "outputs/graphs/entity_types"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ORDER = ["Fine-tuned BERT", "HuggingFace", "spaCy"]

def _normalize_entity_name(x: str) -> str:
    if pd.isna(x):
        return x
    x = re.sub(r"^(B-|I-)", "", str(x))
    return x.strip().upper()

def _clean_tool_name(x: str) -> str:
    x = str(x).strip()
    if x.lower().startswith("fine"):
        return "Fine-tuned BERT"
    if x.lower().startswith("hug"):
        return "HuggingFace"
    if x.lower().startswith("spa"):
        return "spaCy"
    return x

def load_and_prepare(dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(METRICS_FILE)
    df = df[df["Dataset"] == dataset_name].copy()
    if df.empty:
        return df

    df["Tool"] = df["Tool"].apply(_clean_tool_name)
    df["Entity"] = df["Entity"].apply(_normalize_entity_name)

    for col in ["Precision", "Recall"]:
        if col not in df.columns:
            df[col] = pd.NA

    grouped = (
        df.groupby(["Entity", "Tool"], as_index=False)[["F1", "Precision", "Recall"]]
          .mean(numeric_only=True)
    )
    grouped["Tool"] = pd.Categorical(grouped["Tool"], MODEL_ORDER, ordered=True)
    grouped = grouped.sort_values(["Entity", "Tool"]).reset_index(drop=True)
    return grouped

def plot_combined_entity_f1(datasets):
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(20, 14))
    axes = axes.flatten()

    for i, ds in enumerate(datasets):
        grouped = load_and_prepare(ds)
        ax = axes[i]

        if grouped.empty:
            ax.axis("off")
            ax.set_title(f"{ds} (no data)", fontsize=14)
            continue

        pivot = grouped.pivot(index="Entity", columns="Tool", values="F1")
        pivot.plot(kind="bar", ax=ax, width=0.8)

        ax.set_title(f"{ds}", fontsize=16, fontweight="bold")
        ax.set_ylabel("F1 Score", fontsize=14)
        ax.set_xlabel("Entity Type", fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.tick_params(axis="x", labelrotation=30, labelsize=10)
        ax.tick_params(axis="y", labelsize=12)

    # Hide unused axes if fewer datasets than subplots
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    # Put legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODEL_ORDER), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUTPUT_DIR, "combined_entity_f1.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved improved combined entity-type F1 chart → {out_path}")

def main():
    datasets = ["conll2003", "wnut17", "bc5cdr", "jnlpba"]
    plot_combined_entity_f1(datasets)

if __name__ == "__main__":
    main()
