"""
Script to train KNN model on handwritten digits dataset.
"""
import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knn import KNNClassifier
from src.dataio import load_dataset_from_directory, save_model
from src.utils import train_test_split
from src.metrics import accuracy_score, classification_report, confusion_matrix
from src.visualization import plot_confusion_matrix, plot_class_distribution


def main():
    parser = argparse.ArgumentParser(description='Train KNN model on handwritten digits')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing training data')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of neighbors for KNN')
    parser.add_argument('--distance', type=str, default='euclidean',
                        choices=['euclidean', 'manhattan', 'cosine', 'minkowski'],
                        help='Distance metric to use')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per class (None for all)')
    parser.add_argument('--model_path', type=str, default='./models/knn_model.pkl',
                        help='Path to save trained model')
    parser.add_argument('--image_size', type=int, default=28,
                        help='Size to resize images to (square)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-invert', action='store_true',
                        help='Do not invert image colors (use if images are already black-on-white)')

    args = parser.parse_args()

    print("="*70)
    print("KNN CLASSIFIER TRAINING")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"K value: {args.k}")
    print(f"Distance metric: {args.distance}")
    print(f"Test size: {args.test_size}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Max samples per class: {args.max_samples if args.max_samples else 'All'}")
    print("="*70)

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    X, y = load_dataset_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        max_samples_per_class=args.max_samples
    )

    # Visualize class distribution
    print("\n[2/5] Analyzing class distribution...")
    plot_class_distribution(y, class_names=[str(i) for i in range(10)])

    # Split data
    print("\n[3/5] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train model
    print(f"\n[4/5] Training KNN model with k={args.k}...")
    model = KNNClassifier(k=args.k, distance_metric=args.distance)
    model.fit(X_train, y_train)
    print("Model training complete!")

    # Evaluate model
    print("\n[5/5] Evaluating model...")
    print("\nTraining set performance:")
    train_accuracy = model.score(X_train, y_train)
    print(f"Accuracy: {train_accuracy:.4f}")

    print("\nTest set performance:")
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {test_accuracy:.4f}")

    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])

    # Save model
    print(f"\nSaving model to {args.model_path}...")
    save_model(model, args.model_path)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {args.model_path}")
    print("="*70)


if __name__ == "__main__":
    main()
