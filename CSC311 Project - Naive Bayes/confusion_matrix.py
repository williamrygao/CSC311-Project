import numpy as np
import matplotlib.pyplot as plt

# Confusion matrix (rows = true labels, cols = predicted labels)
cm = np.array([
    [35, 4, 1],
    [10, 18, 5],
    [9, 12, 13]
])

labels = ["ChatGPT", "Claude", "Gemini"]

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.colorbar()

# Tick labels
plt.xticks(np.arange(len(labels)), labels, rotation=45)
plt.yticks(np.arange(len(labels)), labels)

# Annotate each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
