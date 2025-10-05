# 🧠 Building a KNN from Scratch
Author: <span style="color:gold;">John Sciuto</span>
## ✍️ Handwritten Digit Classification (non-MNIST)

In this project, I implement a **K-Nearest Neighbours (KNN)** model classifier ***from scratch*** (using **NumPy only** for the algorithm), and evaluate it on the **Handwritten Digits Dataset (not in MNIST)**.  
The implementation demonstrates machine learning theory → code mapping, clear task I/O, and rigorous evaluation, in alignment with **A2: Study, Implement, Present a Machine Learning Model**.

---

### 📂 Dataset
**Source:** Handwritten Digits Dataset (not in MNIST) by *jcprogjava*  
**Kaggle:** [https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist](https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist)

---

## Project Goals and Scope

**🎯 Goal:**  
Implement a KNN classifier from first principles and apply it to a real, non-toy handwritten digit classification task.

**📝 Task:**  
10-class classification of 28×28 grayscale digit images (classes 0–9).

**Emphasis:**
- Correct, readable, and fully vectorized implementation  
- Explicit choices for distance metrics and voting strategies  
- Proper evaluation (accuracy, macro-precision, recall, F1, confusion matrix)  
- Reproducibility via caching and fixed random seeds  

> ⚠️ `scikit-learn`’s `KNeighborsClassifier` **is not used**.  
> The library is imported only for utilities (e.g., stratified train/val/test splits).

---

## Task Definition (A2: Criterion A ✅️)

**Input (Training & Inference):**  
A 28×28 grayscale image, flattened to a 784-dimensional `float32` vector with values scaled to `[0, 1]`.

**Output:**  
A class label in `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}`.  
Optionally, `predict_proba` returns per-class vote or weight proportions.

**Assumptions & Data Quirks:**  
Kaggle’s auto-unzip may create nested subfolders.  
This implementation recursively scans `./data/<digit>/**/*.png` and automatically resizes non-28×28 images.

---

## 📁 Repository Structure

```graphql
knn-from-scratch/               # Root directory
├─ README.md
├─ LICENCE                      # License file (project-specific)
├─ requirements.txt
├─ colab/
│  └─ knn_a2_demo.ipynb         # Self-contained demo notebook (Colab-friendly)
├─ data/
│  ├─ 0/ ...                    # PNGs for class 0 (possibly nested subfolders)
│  ├─ 1/ ...
│  └─ 9/ ...
│  └─ processed/
│     └─ digits_28x28.npz       # Cached arrays (auto-created)
├─ scripts/
│  ├─ train_knn.py              # Build data → fit → eval (val/test) → save model bundle
│  └─ evaluate_knn.py           # Load model bundle → evaluate on test
├─ src/
│  ├─ dataio.py                 # PNG → arrays; normalization; stratified splits; cache
│  ├─ distances.py              # Pairwise Euclidean/Manhattan (vectorized)
│  ├─ knn.py                    # From-scratch KNN: fit/predict/predict_proba
│  ├─ metrics.py                # Accuracy, macro P/R/F1, confusion matrix
│  ├─ utils.py                  # Seeding, simple timing
│  └─ visualization.py          # Sample display & confusion-matrix plotting (matplotlib)
└─ venv/                        # Local virtual environment (optional)
```

## Environment Setup
### Local (I strongly recommend Python 3.10 🚨)
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements     # recursively install all
                                # modules and libraries in use
```

Open `colab/knn_a2_demo.ipynb` in Google Colab and run all cells. The notebook will install all of the needed dependencies needed and can either **(a)** download the dataset automatically or **(b)** assume it’s uploaded to your Drive/session.

## 📥 Dataset Acquisition
### Option A: Manual
1. Download from Kaggle and extract
2. Place folders `"./data/0/1"`, `"./data/9/"` under `"./data/"`.
Paths like ./data/0/0.png, ./data/0/10772.png, … are valid; nested subfolders are also handled.

### Option B: Programmatic via KaggleHub
```python
import kagglehub
path = kagglehub.dataset_download("jcprogjava/handwritten-digits-dataset-not-in-mnist")
print("Path to dataset files:", path)
# move the digit folders 0..9 into ./data/
```

# Data Processing Pipeline
Implemented in `src/dataio.py`:

Discovery: Recursively scan `./data/<label>/**.png` for `label ∈ {0,…,9}.`

**Load & preprocess**:

- Convert to grayscale; resize to 28×28 if necessary.
- Convert to `float32` and scale to `[0,1]`.
- Flatten to vectors of size 784.

**Stratified splits**:
Default: `train/val/test = 70% / 15% / 15%` with a fixed seed.

Caching:
Arrays are saved to `./data/processed/digits_28x28.npz` for fast subsequent runs.

## KNN Implementation (A2: Crtierion B ✅️)
Implemented in `src/knn.py`:

**Classifier**: `KNNClassifier(k=3, metric="euclidean", weights="uniform")`

**Supported distances**: Euclidean, Manhattan (`src/distances.py`).

**Voting**:
- **Uniform** — majority vote among the `k` neighbors.
- **Distance** — weights are `1 / (dist + ε)`; ties broken by smallest class id.

**API**:
- `fit(X, y)` — stores the training set (non-parametric).
- `predict(X, batch_size=1024)` — batched inference to manage memory.
- `predict_proba(X)` — per-class vote/weight proportions.

**Complexity**:
Time ≈ `O(n_test × n_train × d)` per evaluation (non-indexed brute force). Batched computation controls memory footprint.

## Evaluation (A2: Criterion C ✅️)
Metrics implemented in `src/metrics.py`:

- **Accuracy**
- **Macro Precision / Recall / F1**
- **Confusion matrix**

Documented experiments (see ***report***)

- Varying `k` ∈ {1, 3, 5, 7}
- Euclidean vs. Manhattan
- Uniform vs. distance weighting

# 💡 How to Run the Code
### Train and Evaluate (CLI)
From the repo root:
```bash
# first run builds the cache (data/processed/digits_28x28.npz), then evaluates.
python scripts/train_knn.py \
  --k 5 \
  --metric euclidean \        # options: euclidean, manhattan
  --weights uniform \         # options: uniform, distance
  --seed 42 \
  --model_out ./experiments/knn_model.npz
```

This script does the following:

1. Builds/loads the cached dataset.
2. Fits KNN (stores X_train, y_train).
3. Evaluates on validation and test sets; prints metrics and confusion matrices.
4. Saves a model bundle (experiments/knn_model.npz) with training arrays and config.

Evaluate later using the saved bundle:
```bash
python scripts/evaluate_knn.py --model ./experiments/knn_model.npz
```

### Colab Notebook
Open and run `colab/knn_a2_demo.ipynb`.
The notebook mirrors the CLI flow: dependency install → data acquisition → processing → KNN build → evaluation → figures.

## Visualization
`src/visualization.py` provides:

- `show_samples(X, y, n=16)` — quick grid of random digits.
- `plot_confusion_matrix(cm, class_names)` — heatmap with per-cell counts.

Use within notebooks or ad-hoc scripts to generate figures for your report.

## Limitations and Notes
- **Memory/compute**: Brute-force KNN is `O(n_train)` per test sample. If you encounter memory pressure, reduce the `predict` `batch_size` (default in scripts is 2048; lower to 512/256 if needed).

- **High-dimensional sensitivity**: Distance metrics can degrade in higher dimensions; normalization and simple dimensionality reduction (e.g., PCA for visualization only) may help interpretability, but the core algorithm here remains classic KNN.

- **No index structures**: For clarity, this implementation omits KD-Trees/ball trees/ANN libraries.

## 📋 Requirements
Minimal dependencies required for this project (see requirements.txt):

- numpy, pandas
- Pillow (image IO)
- matplotlib
- scikit-learn (utilities: stratified splits; optional comparisons)
- kagglehub (optional; programmatic dataset download)

```bash
pip install -r requirements.txt
```

# 📜 License and Credits

See [LICENCE](LICENCE).

- **Dataset**: *Handwritten Digits Dataset* (not in MNIST) by ***jcprogjava*** on Kaggle.

Please review and comply with the dataset’s license/terms on the Kaggle page.
