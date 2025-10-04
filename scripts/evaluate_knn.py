import argparse
import numpy as np
from src.dataio import build_dataset
from src.knn import KNNClassifier
from src.metrics import accuracy, precision_recall_f1_macro, confusion_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./experiments/knn_model.npz")
    args = parser.parse_args()

    # Load data
    data = build_dataset()
    X_test, y_test = data["X_test"], data["y_test"]

    # Load saved "model"
    blob = np.load(args.model)
    X_train = blob["X_train"]
    y_train = blob["y_train"]
    k = int(blob["k"]); metric = str(blob["metric"]); weights = str(blob["weights"])

    clf = KNNClassifier(k=k, metric=metric, weights=weights).fit(X_train, y_train)
    y_pred = clf.predict(X_test, batch_size=2048)

    acc = accuracy(y_test, y_pred)
    p, r, f1 = precision_recall_f1_macro(y_test, y_pred, n_classes=10)
    cm = confusion_matrix(y_test, y_pred, 10)
    print(f"[TEST] acc={acc:.4f} macroP={p:.4f} macroR={r:.4f} macroF1={f1:.4f}")
    print("[TEST] Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
