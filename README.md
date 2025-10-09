# 🧠 Writing KNN Model From Scratch
Author: <span style="color:gold;">John Sciuto</span>
## ✍️ Handwritten Digit Classification (non-MNIST)

In this project, I implement a **K-Nearest Neighbours (KNN)** classifier ***from scratch*** (using **NumPy only** for the algorithm), and evaluate it on the **Handwritten Digits Dataset (not in MNIST)**.
The implementation demonstrates varying machine learning theory, including → code mapping, clear task I/O, and thorough evaluation, in alignment with **A2: Study, Implement, Present a Machine Learning Model**.

---

### 📂 Dataset
**Source:** Handwritten Digits Dataset (not in MNIST) by *jcprogjava*
**Kaggle:** [https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist](https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist)

---

## **Project Goals and Scope**

### **🎯 Goal**
Implement a KNN machine learning model classifier from first principles and apply it to a real, non-toy handwritten digit classification task.

### **📝 Task**
10-class classification of 28×28 grayscale digit images (classes 0–9).

#### **Emphasis**
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

## 📁 **Repository Structure**

```graphql
knn-from-scratch/
├─ .gitignore
├─ requirements.txt
├─ README.md
├─ LICENCE
├─ data/
│  ├─ 0/ ...                   # PNG/JPG per class (0..9)
│  └─ 9/ ...
├─ experiments/
│  └─ example_output.txt       # Sample console output
├─ scripts/
│  ├─ train_knn.py             # Load → split → train → evaluate → save model
│  ├─ evaluate_knn.py          # Load model → evaluate → visualize
│  ├─ check_images.py          # Visual sanity-check and invert recommendation
│  └─ mod_data_dir.py          # Flatten nested class folders if needed
└─ src/
   ├─ dataio.py                # Image loading, preprocessing, directory scanning
   ├─ distances.py             # euclidean, manhattan, cosine, minkowski
   ├─ knn.py                   # KNNClassifier: fit/predict/predict_proba/score
   ├─ metrics.py               # accuracy_score, classification_report, confusion_matrix, etc.
   ├─ utils.py                 # normalize/standardize, simple train_test_split
   └─ visualization.py         # plots (confusion matrix, samples, class distribution)
```

## Environment setup

Recommended: Python 3.10+

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 📋 Requirements
I have used minimal dependenices for this proejct, please match `requirements.txt`:

- numpy
- Pillow
- scikit-learn
- matplotlib
- seaborn
- tqdm
- joblib

<br>

Install via:

```bash
pip install -r requirements.txt
```
---

## 📥 **Dataset acquisition**

- **Manual** <br>
  Manually download the Kaggle dataset and place the images under:
  - ./data/0, ./data/1, … ./data/9
  - PNG and JPG are supported; nested numeric subfolders are allowed

***NOTE***: If the extracted structure is **"./data/0/0/*.png"**, use `scripts/mod_data_dir.py` to flatten it via:

```bash
python scripts/mod_data_dir.py
```

---

## 📈 Data processing pipeline

Implemented in `src/dataio.py`:
- Load PNG/JPG; compose transparent backgrounds over white
- Convert to grayscale (L), resize to target size (default 28×28)
- Optional inversion for white-on-black images (default invert=True; pass `--no-invert` to skip)
- Normalize to [0,1]; flatten to 784-dim

Train/test split uses a simple random split implemented in `src/utils.py` (not stratified).

To visually verify image polarity and formatting, run:

```bash
python scripts/check_images.py --data_dir ./data --n_samples 3
```

If digits look correct without inversion, use `--no-invert` for training/evaluation.

---

## 🛠 **KNN Implementation**

Implemented in `src/knn.py`:

- Class: `KNNClassifier(k=3, distance_metric='euclidean')`
- Supported distances: euclidean, manhattan, cosine, minkowski (`src/distances.py`)
- Voting: uniform majority among k neighbors (no distance weighting)
- API:
  - fit(X, y)
  - predict(X)
  - predict_proba(X)  // frequency-based probabilities over k neighbors
  - score(X, y)       // accuracy

***Complexity***: brute-force **`O(n_test × n_train × d)`** per evaluation (no tree/ANN index; intentionally simple).

---

## ✅ Evaluation

Metrics in `src/metrics.py`:
- accuracy_score
- precision_score, recall_score, f1_score (macro/weighted supported)
- classification_report (text table)
- confusion_matrix

Visualizations in `src/visualization.py`:
- plot_confusion_matrix
- plot_class_distribution
- plot_sample_images

---

## 💡 **How To Run (CLI)**

Train and evaluate (saves a pickled model with joblib):

```bash
python scripts/train_knn.py \
  --data_dir ./data \
  --k 5 \
  --distance euclidean \   # euclidean | manhattan | cosine | minkowski
  --test_size 0.2 \
  --max_samples 1000 \     # optional cap per class
  --image_size 28 \
  --random_seed 42 \
  --model_path ./models/knn_model.pkl \
  --no-invert              # add this if your digits are already black-on-white
```

Evaluate a saved model and visualize:

```bash
python scripts/evaluate_knn.py \
  --model_path ./models/knn_model.pkl \
  --data_dir ./data \
  --test_size 0.2 \
  --max_samples 1000 \
  --image_size 28 \
  --random_seed 42 \
  --show_samples 20
```

***What you’ll see***:
- Per-class loading progress (tqdm)
- Class distribution bar chart
- Train/test sizes printed
- Classification report (precision/recall/F1 per class and macro averages)
- Confusion matrix heatmap
- Final accuracy

See sample output in [example_output.txt](experiments/example_output.txt).

---

# 📜 License and Credits

See [LICENCE](LICENCE).

- **Dataset**: *Handwritten Digits Dataset* (not in MNIST) by ***jcprogjava*** on Kaggle.

Please review and comply with the dataset’s license/terms on the Kaggle page.
