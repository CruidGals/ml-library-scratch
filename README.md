# ml-library-scratch

Educational implementations of classic machine learning algorithms and a small **NumPy-only neural network stack** (training loop, autograd-style backward passes, optimizers, and checkpoints). The goal is to understand algorithms by implementing them with minimal dependencies—**no PyTorch or scikit-learn** in the core library code.

The CNN path uses **im2col + GEMM** for convolutions rather than naive Python loops; see [results.md](results.md) for a rough CPU speedup comparison on the LeNet-style demo.

---

## Repository layout

```
ml-library-scratch/
├── README.md
├── results.md                 # CNN timing notes (loop vs im2col)
├── datasets/                  # Example CSVs (MBA, salary, heart)
│   ├── MBA.csv
│   ├── Salary_dataset.csv
│   └── heart.csv
└── src/
    ├── classic/               # Traditional ML (mostly NumPy)
    │   ├── models.py
    │   └── utils.py
    └── nn/                    # Mini “framework” + LeNet MNIST script
        ├── modules.py         # Layers, activations, Parameter
        ├── optim.py           # Adam
        ├── loss.py            # Cross-entropy (+ softmax)
        ├── dataloaders.py     # Shuffled minibatch iterator
        ├── param_init.py      # Weight initialization
        ├── util.py            # save/load state_dict, YAML arch
        └── model.py           # End-to-end LeNet-on-MNIST example
```

There is no packaged `setup.py` or `pyproject.toml` yet; treat `src/classic` and `src/nn` as **runnable script directories** (see below).

---

## Dependencies

| Area | Packages |
|------|----------|
| **Core** | `numpy` |
| **Neural net utilities** | `pyyaml` (for `util.get_modules_info` / `load_modules_info`) |
| **MNIST demo only** (`src/nn/model.py`) | `tensorflow` — used only to call `tf.keras.datasets.mnist.load_data()`; all training is NumPy |

The original note mentioned **pandas** for data handling; the current Python sources do not import it. The `datasets/` CSVs are there for your own experiments if you add loading code.

Suggested install for the full repo including the MNIST script:

```bash
pip install numpy pyyaml tensorflow
```

For classic-only work:

```bash
pip install numpy
```

---

## Classic machine learning (`src/classic`)

Implemented in `models.py` with helpers in `utils.py` (Gini impurity, bootstrap sampling, losses, `DTNode`).

| Component | Description |
|-----------|-------------|
| **`DecisionTreeClassifier`** | Greedy splits minimizing weighted Gini. Supports **numeric** thresholds (split points between sorted unique values) and **categorical** splits (equality to a category). Optional `categorical_features` column indices. |
| **`RandomForestClassifier`** | Ensemble of trees fit on **bootstrap** samples; prediction by **majority vote** across trees. |
| **`LinearDiscriminantAnalysis`** | Fits class means and a pooled covariance matrix in `__init__`. **`predict` is not implemented yet** (stub). |
| **`UnivariateLinearRegression`** | Single-feature linear model with **gradient descent** on MSE. |
| **`LogisticRegression`** | Binary logistic regression with **Adam** updates and BCE-style monitoring. |
| **`pca`** | PCA via covariance **SVD**; chooses `k` by cumulative explained variance vs. `epsilon` (see docstring: normalize inputs first). |
| **`LinearSVM`** | Linear SVM-style hinge objective with **Adam** (labels expected in ±1 form for the hinge). |

**How to run / import:** files use `from utils import *`, so run Python with `src/classic` as the working directory, or adjust imports if you later turn this into a package.

```bash
cd src/classic
python -c "import numpy as np; from models import DecisionTreeClassifier; ..."
```

---

## Neural network library (`src/nn`)

Design is **PyTorch-inspired**: each building block subclasses `Module` with `forward`, `backward`, optional `predict` (inference without storing full training caches where applicable), `get_params`, `load_params`, and `get_info`.

### Layers and ops (`modules.py`)

- **`Parameter`** — holds `value`, `grad`, and (when used with Adam) optimizer state on the object.
- **`Linear`** — fully connected: `X @ W + b`.
- **`Conv2D`** — 2D convolution, **NCHW** tensors `(batch, channels, height, width)`; **im2col + GEMM** forward; backward through `col2im`.
- **`MaxPool2D`** — max pooling with the same stride/padding ideas; backward scatters gradients to max indices.
- **`Flatten`**, **`ReLU`**, **`Sigmoid`**, **`Dropout`** — inverted dropout in training forward.

### Training stack

- **`CrossEntropyLoss`** (`loss.py`) — numerically stable softmax + cross-entropy; backward starts with `(softmax - labels) / batch_size` and walks modules in reverse.
- **`Adam`** (`optim.py`) — standard Adam with optional **weight decay** (L2-style term on the gradient).
- **`DataLoader`** (`dataloaders.py`) — shuffles indices each epoch, serves contiguous batches via `next()` / `reset()`.

### Checkpoints (`util.py`)

- **`save_state_dict` / `load_state_dict`** — NumPy `.npz`-compatible dict of per-layer parameters.
- **`get_modules_info` / `load_modules_info`** — YAML architecture metadata for reconstruction. **`load_modules_info`** currently handles Linear, Conv2D, MaxPool2D, ReLU, and Sigmoid only (no Flatten/Dropout branches yet—extend if you need full round-trips).

### MNIST LeNet demo (`model.py`)

Trains a **LeNet-style** stack on MNIST: two conv blocks (ReLU + max pool), then three linear layers, **cross-entropy loss**, **Adam**, simple **learning-rate decay** per epoch, then test accuracy and saves to `saved/model.npz` and `saved/model.yaml`.

Run from `src/nn` (same relative-import pattern as the rest of this folder):

```bash
cd src/nn
python model.py
```

Ensure `saved/` is writable (it is listed in `.gitignore`).

### Initialization (`param_init.py`)

Default layer init uses **`normal`** sampling (small std) on weight tensors.

---

## Datasets folder

CSV files are included for **heart**, **salary**, and **MBA**-style tabular problems. They are not wired into the library by default; use them with your own preprocessing or with pandas if you add a script.

---

## Roadmap / known gaps

- **`LinearDiscriminantAnalysis.predict`** — not implemented.
- **`load_modules_info`** — incomplete vs. full architectures (e.g. Flatten, Dropout).
- **Packaging** — no `requirements.txt` / installable package yet; imports assume directory-local execution.
- **`RandomForestClassifier.predict`** — currently prints predictions (you may want to remove or gate that for library use).

---

## License

This project is released under the [MIT License](LICENSE).
