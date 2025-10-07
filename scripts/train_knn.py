import os
import argparse
import numpy as np
from src.dataio import build_dataset
from src.knn import KNNClassifier
from src.metrics import accuracy, precision_recall_f1_macro, confusion_matrix
from src.utils import set_seed, Timer

def save_model(path: str, model: KNNClassifier, X_train: np.ndarray, y_train: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        X_train=X_train, y_train=y_train,
        k=model.k, metric=model.metric, weights=model.weights
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "manhattan"])
    parser.add_argument("--weights", type=str, default="uniform", choices=["uniform", "distance"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_out", type=str, default="./experiments/knn_model.npz")
    args = parser.parse_args()

    set_seed(args.seed)
    data = build_dataset()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    clf = KNNClassifier(k=args.k, metric=args.metric, weights=args.weights).fit(X_train, y_train)

    with Timer() as t:
        y_val_pred = clf.predict(X_val, batch_size=2048)
    val_acc = accuracy(y_val, y_val_pred)
    p, r, f1 = precision_recall_f1_macro(y_val, y_val_pred, n_classes=10)
    print(f"[VAL] acc={val_acc:.4f} macroP={p:.4f} macroR={r:.4f} macroF1={f1:.4f} (pred time {t.elapsed:.2f}s)")
    cm_val = confusion_matrix(y_val, y_val_pred, 10)
    print("[VAL] Confusion matrix:\n", cm_val)

    with Timer() as t2:
        y_test_pred = clf.predict(X_test, batch_size=2048)
    test_acc = accuracy(y_test, y_test_pred)
    p2, r2, f12 = precision_recall_f1_macro(y_test, y_test_pred, n_classes=10)
    print(f"[TEST] acc={test_acc:.4f} macroP={p2:.4f} macroR={r2:.4f} macroF1={f12:.4f} (pred time {t2.elapsed:.2f}s)")
    cm_test = confusion_matrix(y_test, y_test_pred, 10)
    print("[TEST] Confusion matrix:\n", cm_test)

    save_model(args.model_out, clf, X_train, y_train)
    print(f"Saved model to: {args.model_out}")

if __name__ == "__main__":
    main()
