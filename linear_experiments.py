"""
Felidae Rosenblatt — hyperparameter experiments for the report.

  1. Learning rate sweep — impact of lr on convergence
  2. Epochs sweep        — under/over-training
  3. Seed variance       — stability across random inits
  4. Confusion matrix    — per-class performance of the best config

Plots saved under report_plots/rosenblatt_<name>.png
Run from the PA root:
    python rosenblatt_experiments.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cffi import FFI
from PIL import Image

ffi = FFI()
ffi.cdef("""
    void* create_rosenblatt_classifier(
        size_t n_features, float learning_rate,
        float bias_cat, float bias_lion, float bias_cheetah,
        uint64_t seed
    );
    void train_classifier(
        void* classifier,
        const float* x, size_t rows, size_t cols,
        const size_t* y, size_t y_len,
        size_t epochs
    );
    size_t* predict_classifier(
        void* classifier,
        const float* x, size_t rows, size_t cols,
        const size_t* y, size_t y_len
    );
    void extract_features(const float* image_data, float* out, size_t out_len);
""", override=True)

BASE = os.path.abspath("felidae_classifier")
if os.name != "nt":
    lib = ffi.dlopen(os.path.join(BASE, "target/release/libfelidae_classifier.so"))
else:
    lib = ffi.dlopen(os.path.join(BASE, "target/release/felidae_classifier.dll"))

FEATURE_MODE = "extract"     # "extract" ou "flatten"
IMG_SIZE = (32, 32)
DATASET_ROOT = os.path.join(BASE, "dataset")
CLASS_NAMES = ["Cat", "Lion", "Cheetah"]
N_FEATURES = 20 if FEATURE_MODE == "extract" else 3072

PLOTS_DIR = os.path.abspath("report_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
np.random.seed(42)


def features_from_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.ascontiguousarray(np.array(img, dtype=np.float32).flatten() / 255.0, dtype=np.float32)
    if FEATURE_MODE == "extract":
        out = ffi.new("float[]", N_FEATURES)
        lib.extract_features(ffi.from_buffer("float[]", arr), out, N_FEATURES)
        return np.array([out[i] for i in range(N_FEATURES)], dtype=np.float32)
    return arr


def load_felidae(max_per_class):
    class_folders = {"cat": 0, "lion": 1, "cheetah": 2}
    X, Y = [], []
    for folder, label in class_folders.items():
        folder_path = os.path.join(DATASET_ROOT, folder)
        if not os.path.exists(folder_path):
            print(f"  WARNING: folder not found: {folder_path}")
            continue
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith((".jpg", ".png"))][:max_per_class]
        for fname in files:
            try:
                X.append(features_from_image(os.path.join(folder_path, fname)))
                Y.append(label)
            except Exception as e:
                print(f"  Failed on {fname}: {e}")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.uint64)


def split_train_test(X, Y, test_ratio=0.2):
    idx = np.random.permutation(len(X))
    split = int((1 - test_ratio) * len(X))
    tr, te = idx[:split], idx[split:]
    return (X[tr], Y[tr]), (X[te], Y[te])


def train_rosenblatt(X_train, Y_train, lr, epochs, seed=42):
    clf = lib.create_rosenblatt_classifier(
        X_train.shape[1], float(lr), 0.0, 0.0, 0.0, int(seed)
    )
    X_c = np.ascontiguousarray(X_train, dtype=np.float32)
    Y_c = np.ascontiguousarray(Y_train, dtype=np.uint64)
    lib.train_classifier(
        clf,
        ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1],
        ffi.from_buffer("size_t[]", Y_c), len(Y_c),
        int(epochs)
    )
    return clf


def accuracy_of(clf, X, Y):
    X_c = np.ascontiguousarray(X, dtype=np.float32)
    Y_c = np.ascontiguousarray(Y, dtype=np.uint64)
    preds_ptr = lib.predict_classifier(
        clf,
        ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1],
        ffi.from_buffer("size_t[]", Y_c), len(Y_c)
    )
    preds = np.array([preds_ptr[i] for i in range(X_c.shape[0])])
    return float(np.mean(preds == Y) * 100.0), preds


def save_plot(fig, name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# --- Experiment 1: learning rate sweep ---
def experiment_lr_sweep(data):
    print("\n=== Rosenblatt Experiment 1 — Learning rate sweep ===")
    (X_train, Y_train), (X_test, Y_test) = data
    lrs = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
    epochs = 500

    train_accs, test_accs = [], []
    for lr in lrs:
        clf = train_rosenblatt(X_train, Y_train, lr, epochs)
        tr, _ = accuracy_of(clf, X_train, Y_train)
        te, _ = accuracy_of(clf, X_test, Y_test)
        train_accs.append(tr)
        test_accs.append(te)
        print(f"  lr={lr:<8} train={tr:.1f}% test={te:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lrs, train_accs, "o-", label="Train", color="steelblue")
    ax.plot(lrs, test_accs, "o-", label="Test", color="orange")
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Rosenblatt — Impact of learning rate ({epochs} epochs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "rosenblatt_lr_sweep")


# --- Experiment 2: epochs sweep ---
def experiment_epochs_sweep(data):
    print("\n=== Rosenblatt Experiment 2 — Epochs sweep ===")
    (X_train, Y_train), (X_test, Y_test) = data
    epoch_values = [10, 50, 100, 250, 500, 1000, 2000, 5000]
    lr = 0.01

    train_accs, test_accs = [], []
    for ep in epoch_values:
        clf = train_rosenblatt(X_train, Y_train, lr, ep)
        tr, _ = accuracy_of(clf, X_train, Y_train)
        te, _ = accuracy_of(clf, X_test, Y_test)
        train_accs.append(tr)
        test_accs.append(te)
        print(f"  epochs={ep:<6} train={tr:.1f}% test={te:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epoch_values, train_accs, "o-", label="Train", color="steelblue")
    ax.plot(epoch_values, test_accs, "o-", label="Test", color="orange")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Rosenblatt — Impact of training epochs (lr={lr})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "rosenblatt_epochs_sweep")


# --- Experiment 3: seed variance ---
def experiment_seed_variance(data):
    print("\n=== Rosenblatt Experiment 3 — Seed variance ===")
    (X_train, Y_train), (X_test, Y_test) = data
    seeds = [1, 7, 42, 123, 999, 2024, 31337, 65536]
    lr, epochs = 0.01, 500

    test_accs = []
    for s in seeds:
        clf = train_rosenblatt(X_train, Y_train, lr, epochs, seed=s)
        te, _ = accuracy_of(clf, X_test, Y_test)
        test_accs.append(te)
        print(f"  seed={s:<7} test={te:.1f}%")

    mean, std = np.mean(test_accs), np.std(test_accs)
    print(f"  Mean test accuracy: {mean:.1f}% ± {std:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(seeds)), test_accs, color="steelblue")
    ax.axhline(mean, color="orange", linestyle="--", label=f"Mean = {mean:.1f}%")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Rosenblatt — Stability across seeds (lr={lr}, {epochs} epochs)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "rosenblatt_seed_variance")


# --- Experiment 4: confusion matrix ---
def experiment_confusion(data):
    print("\n=== Rosenblatt Experiment 4 — Confusion matrix ===")
    (X_train, Y_train), (X_test, Y_test) = data
    lr, epochs = 0.01, 500

    clf = train_rosenblatt(X_train, Y_train, lr, epochs)
    te, preds = accuracy_of(clf, X_test, Y_test)

    n = 3
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(Y_test, preds):
        cm[int(t)][int(p)] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Rosenblatt — Confusion matrix (lr={lr}, {epochs} epochs)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    save_plot(fig, "rosenblatt_confusion")

    print(f"  Overall test accuracy: {te:.1f}%")
    for i, name in enumerate(CLASS_NAMES):
        mask = Y_test == i
        if mask.sum() > 0:
            print(f"    {name}: {np.mean(preds[mask] == i) * 100:.1f}%")


def main():
    print("Loading dataset (3000/class)...")
    X, Y = load_felidae(max_per_class=3000)
    print(f"Loaded {len(X)} samples")
    data = split_train_test(X, Y, test_ratio=0.2)
    print(f"Train: {len(data[0][0])} | Test: {len(data[1][0])}")

    experiment_lr_sweep(data)
    experiment_epochs_sweep(data)
    experiment_seed_variance(data)
    experiment_confusion(data)

    print("\nAll Rosenblatt experiments complete.")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()