import numpy as np
import matplotlib.pyplot as plt

def show_samples(X: np.ndarray, y: np.ndarray, n: int = 16):
    idx = np.random.choice(len(X), size=min(n, len(X)), replace=False)
    cols = int(np.sqrt(n))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols*2, rows*2))
    for i, j in enumerate(idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(X[j].reshape(28, 28), cmap="gray")
        plt.title(str(y[j]))
        plt.axis("off")
    plt.tight_layout()

def plot_confusion_matrix(cm: np.ndarray, class_names=None):
    if class_names is None:
        class_names = list(map(str, range(cm.shape[0])))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # annotate
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    return ax
