import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("outputs/metrics/model_comparison.csv")


sns.set(style="whitegrid")


plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=df,
    x="dataset",
    y="f1",
    hue="model"
)


plt.title("F1 Score Comparison by Model and Dataset")
plt.ylabel("F1 Score")
plt.xlabel("Dataset")
plt.ylim(0, 1.05)
plt.legend(title="Model", loc="upper right")
plt.tight_layout()


output_path = "outputs/graphs/model_comparison_f1_bar.png"
plt.savefig(output_path)
plt.show()

print(f"✅ Clustered bar chart saved to: {output_path}")
