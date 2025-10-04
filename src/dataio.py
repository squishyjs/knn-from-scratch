import os
import glob
import numpy as np
from PIL import Image
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(".", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CACHE_PATH = os.path.join(PROCESSED_DIR, "digits_28x28.npz")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _iter_image_paths(root: str):
    """
    Yields (label, path) for any PNG under ./data/<label>/.../*.png
    Handles nested folders (e.g., ./data/0/0/1.png) gracefully.
    """
    for label in map(str, range(10)):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        # recursive glob
        for p in glob.glob(os.path.join(label_dir, "**", "*.png"), recursive=True):
            yield int(label), p

def _load_png_to_array(p: str) -> np.ndarray:
    # Convert to grayscale (L), enforce 28x28, return float32 [0,1] flat (784,)
    img = Image.open(p).convert("L")
    if img.size != (28, 28):
        img = img.resize((28, 28), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)  # flatten 28*28 = 784

def build_dataset(
    data_dir: str = DATA_DIR,
    cache_path: str = CACHE_PATH,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    force_rebuild: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Build or load cached dataset.
    Returns dict with: X_train, y_train, X_val, y_val, X_test, y_test
    """
    _ensure_dir(PROCESSED_DIR)

    if (not force_rebuild) and os.path.isfile(cache_path):
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    # Load all images
    xs, ys = [], []
    for label, path in _iter_image_paths(data_dir):
        xs.append(_load_png_to_array(path))
        ys.append(label)

    X = np.stack(xs, axis=0)  # (N, 784)
    y = np.array(ys, dtype=np.int64)

    # Stratified train / temp split first
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=seed, stratify=y
    )
    # Split temp into val / test with the right proportions
    val_ratio_of_temp = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_of_temp), random_state=seed, stratify=y_temp
    )

    np.savez_compressed(
        cache_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }
