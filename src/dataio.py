"""
Data input/output utilities for loading and preprocessing image data
"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image(image_path, target_size=(28, 28), grayscale=True, invert=True):
    """
    Load and preprocess a single image.
    Handles transparent backgrounds by converting to white background.

    Args:
        image_path: str, path to image file
        target_size: tuple, target size (height, width)
        grayscale: bool, whether to convert to grayscale
        invert: bool, whether to invert colors (for white-on-black images)

    Returns:
        numpy array: Flattened image array
    """
    try:
        img = Image.open(image_path)

        '''
        Handle transparency by transforming
        background into WHITE
        '''
        if img.mode in ('RGBA', 'LA', 'PA'):
            background = Image.new('RGB', img.size, (255, 255, 255)) # white background
            # use alpha channel as mask (white background)
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])  # the alpha channel
            elif img.mode == 'LA':
                background.paste(img, mask=img.split()[1])
            else:
                background.paste(img)
            img = background

        # grayscale conversion
        if grayscale and img.mode != 'L':
            img = img.convert('L')

        # resize image
        img = img.resize(target_size, Image.LANCZOS)
        img_array = np.array(img) # convert numpy

        # check if image is mostly dark (white digits on black background)
        # (if mean pixel value < 128 probably inverted)
        if invert and np.mean(img_array) < 128:
            img_array = 255 - img_array  # Invert: black becomes white, white becomes black

        # flatten
        img_flat = img_array.flatten()

        # normalize to [0, 1]
        img_flat = img_flat.astype(np.float32) / 255.0

        return img_flat

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_dataset_from_directory(data_dir, target_size=(28, 28), max_samples_per_class=None, invert=True):
    """
    Load dataset from directory structure.
    Expected structure:
        data_dir/
            0/
                1.png
                2.png
                ...
            1/
                1.png
                2.png
                ...
            ...

    Args:
        data_dir: str, root directory containing class subdirectories
        target_size: tuple, target size for images
        max_samples_per_class: int or None, maximum samples to load per class
        invert: bool, whether to invert colors if needed

    Returns:
        tuple: (X, y) where X is features array and y is labels array
    """
    X = []
    y = []

    # target directories (data: 0 - 9)
    all_dirs = [d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))]

    # filter labels (0 - 9)
    class_dirs = sorted([d for d in all_dirs if d.isdigit()])

    print(f"Found {len(class_dirs)} valid classes: {class_dirs}")
    if len(all_dirs) > len(class_dirs):
        excluded = set(all_dirs) - set(class_dirs)
        print(f"Excluding non-numeric directories: {excluded}")

    for class_label in class_dirs:
        class_path = os.path.join(data_dir, class_label)
        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Limit samples if specified
        if max_samples_per_class is not None:
            image_files = image_files[:max_samples_per_class]

        print(f"Loading {len(image_files)} images from class '{class_label}'...")

        for image_file in tqdm(image_files, desc=f"Class {class_label}"):
            image_path = os.path.join(class_path, image_file)
            img_array = load_image(image_path, target_size=target_size, invert=invert)

            if img_array is not None:
                X.append(img_array)
                y.append(int(class_label))

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Label distribution: {np.bincount(y)}")

    return X, y


def save_model(model, filepath):
    """
    Save KNN model to file.

    Args:
        model: KNNClassifier instance
        filepath: str, path to save file
    """
    import joblib
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load KNN model from file.

    Args:
        filepath: str, path to model file

    Returns:
        KNNClassifier instance
    """
    import joblib
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
