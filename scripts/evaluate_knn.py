"""
Script to evaluate trained KNN on TEST data
"""
import os
import sys
import argparse
import numpy as np

# parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataio import load_model, load_dataset_from_directory
from src.utils import train_test_split
from src.metrics import accuracy_score, classification_report, confusion_matrix
from src.visualization import plot_confusion_matrix, plot_sample_images


def main():
    parser = argparse.ArgumentParser(description='Evaluate KNN model')
    parser.add_argument('--model_path', type=str, default='./models/knn_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing data')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per class to load')
    parser.add_argument('--image_size', type=int, default=28,
                        help='Size to resize images to')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--show_samples', type=int, default=20,
                        help='Number of sample predictions to display')

    args = parser.parse_args()

    print("="*70)
    print("KNN MODEL EVALUATION")
    print("="*70)
    print(f"Model path: {args.model_path}")
    print(f"Data directory: {args.data_dir}")
    print("="*70)

    # load model
    print("\n[1/4] Loading trained model...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first using train_knn.py")
        return

    model = load_model(args.model_path)
    print(f"Model info: k={model.k}, distance_metric={model.distance_metric}")

    print("\n[2/4] Loading dataset...") # load
    X, y = load_dataset_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        max_samples_per_class=args.max_samples
    )

    # Split data (use same seed as training)
    print("\n[3/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )
    print(f"Test samples: {len(X_test)}")

    # predict with KNN
    print("\n[4/4] Making predictions...")
    y_pred = model.predict(X_test)

    # metrics (performance)
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_pred))

    # confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])

    if args.show_samples > 0: # GET and DISPLAY predictions
        print(f"\nDisplaying {args.show_samples} sample predictions...")
        correct_indices = np.where(y_test == y_pred)[0]
        incorrect_indices = np.where(y_test != y_pred)[0]

        n_correct = min(args.show_samples // 2, len(correct_indices))
        n_incorrect = min(args.show_samples - n_correct, len(incorrect_indices))

        sample_indices = np.concatenate([
            np.random.choice(correct_indices, n_correct, replace=False) if len(correct_indices) > 0 else [],
            np.random.choice(incorrect_indices, n_incorrect, replace=False) if len(incorrect_indices) > 0 else []
        ])

        plot_sample_images(
            X_test[sample_indices],
            y_test[sample_indices],
            y_pred[sample_indices],
            n_samples=len(sample_indices),
            image_shape=(args.image_size, args.image_size)
        )

    # summary
    correct_predictions = np.sum(y_test == y_pred)
    incorrect_predictions = np.sum(y_test != y_pred)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total test samples: {len(y_test)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")
    print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("="*70)


if __name__ == "__main__":
    main()