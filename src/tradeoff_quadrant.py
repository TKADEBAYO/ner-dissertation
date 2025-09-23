import matplotlib.pyplot as plt


models = [
    "Fine-tuned BERT",
    "HuggingFace baseline",
    "spaCy baseline",
    "Rule-based/CRF"
]


accuracy = [
    0.95, 
    0.88,  
    0.79, 
    0.70   
]


interpretability = [
    0.2,   
    0.3,   
    0.7,   
    0.9    
]


plt.figure(figsize=(8, 6))
plt.scatter(interpretability, accuracy, s=200, c=['blue','orange','green','red'], alpha=0.7)


for i, model in enumerate(models):
    plt.text(interpretability[i] + 0.02, accuracy[i], model, fontsize=10, weight="bold")


plt.axhline(y=0.8, color="gray", linestyle="--", alpha=0.7) 
plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7)  

plt.title("Trade-off Quadrant: Accuracy vs Interpretability", fontsize=14, weight="bold")
plt.xlabel("Interpretability (Low → High)")
plt.ylabel("Accuracy (F1 Score)")
plt.grid(True, linestyle="--", alpha=0.5)


plt.tight_layout()
plt.savefig("outputs/graphs/tradeoff_quadrant.png")
plt.show()
