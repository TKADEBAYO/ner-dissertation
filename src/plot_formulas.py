import matplotlib.pyplot as plt

# Define the text
formulas = r"""
Precision = \frac{True\ Positives}{True\ Positives + False\ Positives}

Recall = \frac{True\ Positives}{True\ Positives + False\ Negatives}

F1\ Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}
"""


plt.figure(figsize=(8, 4))
plt.axis('off')
plt.text(0.5, 0.5, formulas, fontsize=16, ha='center', va='center', wrap=True)

plt.tight_layout()
plt.savefig("outputs/graphs/f1_precision_recall_formulae.png", dpi=300)
plt.show()
