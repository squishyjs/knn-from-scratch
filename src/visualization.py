"""
Visualization utilities for KNN model evaluation
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names=None, save_path=None, figsize=(10, 8)):
    """
    Plot confusion matrix using seaborn heatmap.

    Args:
        cm: numpy array, confusion matrix
        class_names: list, names of classes
        save_path: str or None, path to save figure
        figsize: tuple, figure size
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=figsize)

    # heatmap (to be plotted)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_accuracy_by_k(k_values, accuracies, save_path=None, figsize=(10, 6)):
    """
    Plot accuracy scores for different k values.

    Args:
        k_values: list, k values tested
        accuracies: list, corresponding accuracy scores
        save_path: str or None, path to save figure
        figsize: tuple, figure size
    """
    plt.figure(figsize=figsize)

    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('K (Number of Neighbors)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('KNN Accuracy vs K Value', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"K-accuracy plot saved to {save_path}")

    plt.show()


def plot_sample_images(X, y, predictions=None, n_samples=10, image_shape=(28, 28), save_path=None):
    """
    Plot sample images with their labels and predictions.

    Args:
        X: numpy array, image data
        y: numpy array, true labels
        predictions: numpy array or None, predicted labels
        n_samples: int, number of samples to plot
        image_shape: tuple, shape to reshape images to
        save_path: str or None, path to save figure
    """
    n_samples = min(n_samples, len(X))
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]

    for i in range(n_samples):
        img = X[i].reshape(image_shape)
        axes[i].imshow(img, cmap='gray')

        if predictions is not None:
            title = f"True: {y[i]}, Pred: {predictions[i]}"
            color = 'green' if y[i] == predictions[i] else 'red'
            axes[i].set_title(title, fontsize=10, color=color)
        else:
            axes[i].set_title(f"Label: {y[i]}", fontsize=10)

        axes[i].axis('off')

    # subplots (hide)
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images saved to {save_path}")

    plt.show()


def plot_class_distribution(y, class_names=None, save_path=None, figsize=(10, 6)):
    """
    Plot distribution of classes in dataset.

    Args:
        y: numpy array, labels
        class_names: list or None, names of classes
        save_path: str or None, path to save figure
        figsize: tuple, figure size
    """
    unique_classes, counts = np.unique(y, return_counts=True)

    if class_names is None:
        class_names = [str(c) for c in unique_classes]

    plt.figure(figsize=figsize)
    plt.bar(class_names, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Class Distribution', fontsize=16, fontweight='bold')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")

    plt.show()