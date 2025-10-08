"""
Metrics for evaluating model performance.
"""
import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score.

    Args:
        y_true: numpy array of true labels
        y_pred: numpy array of predicted labels

    Returns:
        float: Accuracy score
    """
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix.

    Args:
        y_true: numpy array of true labels
        y_pred: numpy array of predicted labels

    Returns:
        numpy array: Confusion matrix
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Create mapping from class labels to indices
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1

    return cm


def precision_score(y_true, y_pred, average='macro'):
    """
    Calculate precision score.

    Args:
        y_true: numpy array of true labels
        y_pred: numpy array of predicted labels
        average: str, averaging method ('macro', 'micro', 'weighted')

    Returns:
        float: Precision score
    """
    classes = np.unique(y_true)
    precisions = []

    for cls in classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_positives = np.sum((y_true != cls) & (y_pred == cls))

        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        precisions.append(precision)

    if average == 'macro':
        return np.mean(precisions)
    elif average == 'weighted':
        weights = [np.sum(y_true == cls) for cls in classes]
        return np.average(precisions, weights=weights)
    else:
        return np.mean(precisions)


def recall_score(y_true, y_pred, average='macro'):
    """
    Calculate recall score.

    Args:
        y_true: numpy array of true labels
        y_pred: numpy array of predicted labels
        average: str, averaging method ('macro', 'micro', 'weighted')

    Returns:
        float: Recall score
    """
    classes = np.unique(y_true)
    recalls = []

    for cls in classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_negatives = np.sum((y_true == cls) & (y_pred != cls))

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        recalls.append(recall)

    if average == 'macro':
        return np.mean(recalls)
    elif average == 'weighted':
        weights = [np.sum(y_true == cls) for cls in classes]
        return np.average(recalls, weights=weights)
    else:
        return np.mean(recalls)


def f1_score(y_true, y_pred, average='macro'):
    """
    Calculate F1 score.

    Args:
        y_true: numpy array of true labels
        y_pred: numpy array of predicted labels
        average: str, averaging method ('macro', 'micro', 'weighted')

    Returns:
        float: F1 score
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def classification_report(y_true, y_pred):
    """
    Generate a text report showing main classification metrics.

    Args:
        y_true: numpy array of true labels
        y_pred: numpy array of predicted labels

    Returns:
        str: Classification report
    """
    classes = np.unique(y_true)
    report = []

    # Header
    report.append(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    report.append("-" * 60)

    # Per-class metrics
    for cls in classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_positives = np.sum((y_true != cls) & (y_pred == cls))
        false_negatives = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)

        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        report.append(f"{cls:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")

    report.append("-" * 60)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_avg_precision = precision_score(y_true, y_pred, average='macro')
    macro_avg_recall = recall_score(y_true, y_pred, average='macro')
    macro_avg_f1 = f1_score(y_true, y_pred, average='macro')

    report.append(f"{'Accuracy':<10} {'':<12} {'':<12} {accuracy:<12.4f} {len(y_true):<10}")
    report.append(f"{'Macro Avg':<10} {macro_avg_precision:<12.4f} {macro_avg_recall:<12.4f} {macro_avg_f1:<12.4f} {len(y_true):<10}")

    return "\n".join(report)
