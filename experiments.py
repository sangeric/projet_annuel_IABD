import os
import numpy as np
import matplotlib.pyplot as plt
from cffi import FFI
from PIL import Image

ffi = FFI()
ffi.cdef("""
    void* mlp_create(const uint32_t* layer_sizes, size_t n_layers, const char* activation, const char* output_activation);
    void mlp_train(void* mlp, const float* x_data, size_t x_rows, size_t x_cols, const float* y_data, size_t y_rows, size_t y_cols, size_t epochs, float learning_rate, const char* log_dir);
    void mlp_predict_classes(const void* mlp, const float* x_data, size_t x_rows, size_t x_cols, uint32_t* out_predictions);
    void mlp_destroy(void* mlp);
    void extract_features(const float* image_data, float* out, size_t out_len);
""", override=True)

BASE = os.path.abspath("felidae_classifier")
lib = ffi.dlopen(os.path.join(BASE, "target/release/libfelidae_classifier.so"))

N_FEATURES = 20
IMG_SIZE = (32, 32)
DATASET_ROOT = os.path.join(BASE, "dataset_clean")
CLASS_NAMES = ["Cat", "Lion", "Cheetah"]

RUNS_DIR = os.path.abspath("runs/experiments")
PLOTS_DIR = os.path.abspath("report_plots")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

np.random.seed(42)


def load_felidae(max_per_class):
    """Load the felidae dataset through the Rust extract_features pipeline."""
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
                img = Image.open(os.path.join(folder_path, fname)).convert("RGB").resize(IMG_SIZE)
                arr = np.array(img, dtype=np.float32).flatten() / 255.0
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                out = ffi.new("float[]", N_FEATURES)
                lib.extract_features(ffi.from_buffer("float[]", arr), out, N_FEATURES)
                X.append(np.array([out[i] for i in range(N_FEATURES)], dtype=np.float32))
                Y.append(label)
            except Exception as e:
                print(f"  Failed on {fname}: {e}")
                continue
    X = np.array(X, dtype=np.float32)
    labels = np.array(Y)
    Y_onehot = np.full((len(Y), 3), -1.0, dtype=np.float32)
    for i, label in enumerate(Y):
        Y_onehot[i][label] = 1.0
    return X, Y_onehot, labels


def split_train_test(X, Y_onehot, labels, test_ratio=0.2):
    idx = np.random.permutation(len(X))
    split = int((1 - test_ratio) * len(X))
    tr, te = idx[:split], idx[split:]
    return (X[tr], Y_onehot[tr], labels[tr]), (X[te], Y_onehot[te], labels[te])


def train_eval(X_train, Y_train, X_test, Y_test, arch, epochs, lr, log_dir):
    """Create, train (logging to log_dir), and evaluate an MLP. Returns (train_acc, test_acc)."""
    os.makedirs(log_dir, exist_ok=True)

    layers = ffi.new("uint32_t[]", arch)
    mlp = lib.mlp_create(layers, len(arch), b"tanh", b"tanh")
    if mlp == ffi.NULL:
        raise RuntimeError(f"mlp_create failed for arch={arch}")

    X_train_c = np.ascontiguousarray(X_train, dtype=np.float32)
    Y_train_c = np.ascontiguousarray(Y_train, dtype=np.float32)

    lib.mlp_train(
        mlp,
        ffi.from_buffer("float[]", X_train_c), X_train_c.shape[0], X_train_c.shape[1],
        ffi.from_buffer("float[]", Y_train_c), Y_train_c.shape[0], Y_train_c.shape[1],
        epochs, lr, log_dir.encode()
    )

    def accuracy(X, Y_onehot):
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        out = ffi.new("uint32_t[]", X_c.shape[0])
        lib.mlp_predict_classes(mlp, ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1], out)
        preds = np.array([out[i] for i in range(X_c.shape[0])])
        true = np.argmax(Y_onehot, axis=1)
        return float(np.mean(preds == true) * 100.0)

    tr = accuracy(X_train, Y_train)
    te = accuracy(X_test, Y_test)
    lib.mlp_destroy(mlp)
    return tr, te


def predict_labels(X_train, Y_train, X_test, arch, epochs, lr, log_dir):
    """Train and return predicted class indices for X_test (used for confusion matrix)."""
    os.makedirs(log_dir, exist_ok=True)
    layers = ffi.new("uint32_t[]", arch)
    mlp = lib.mlp_create(layers, len(arch), b"tanh", b"tanh")

    X_train_c = np.ascontiguousarray(X_train, dtype=np.float32)
    Y_train_c = np.ascontiguousarray(Y_train, dtype=np.float32)
    lib.mlp_train(
        mlp,
        ffi.from_buffer("float[]", X_train_c), X_train_c.shape[0], X_train_c.shape[1],
        ffi.from_buffer("float[]", Y_train_c), Y_train_c.shape[0], Y_train_c.shape[1],
        epochs, lr, log_dir.encode()
    )

    X_test_c = np.ascontiguousarray(X_test, dtype=np.float32)
    out = ffi.new("uint32_t[]", X_test_c.shape[0])
    lib.mlp_predict_classes(mlp, ffi.from_buffer("float[]", X_test_c), X_test_c.shape[0], X_test_c.shape[1], out)
    preds = np.array([out[i] for i in range(X_test_c.shape[0])])
    lib.mlp_destroy(mlp)
    return preds


def save_plot(fig, name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def experiment_1_learning_rate(data):
    print("\n=== Experiment 1 — Learning rate ===")
    (X_train, Y_train, _), (X_test, Y_test, _) = data
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    epochs = 200
    arch = [N_FEATURES, 32, 16, 3]

    train_accs, test_accs = [], []
    for lr in learning_rates:
        log_dir = os.path.join(RUNS_DIR, "exp1_learning_rate", f"lr_{lr}")
        tr, te = train_eval(X_train, Y_train, X_test, Y_test, arch, epochs, lr, log_dir)
        train_accs.append(tr)
        test_accs.append(te)
        print(f"  lr={lr:<6} train={tr:.1f}% test={te:.1f}%")

    x = np.arange(len(learning_rates))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, train_accs, w, label="Train", color="steelblue")
    ax.bar(x + w/2, test_accs, w, label="Test", color="orange")
    ax.set_xticks(x); ax.set_xticklabels([str(lr) for lr in learning_rates])
    ax.set_xlabel("Learning rate"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Experiment 1 — Impact of learning rate")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "exp1_learning_rate")


def experiment_2_epochs(data):
    print("\n=== Experiment 2 — Epochs (under → overfitting) ===")
    (X_train, Y_train, _), (X_test, Y_test, _) = data
    epoch_values = [10, 25, 50, 100, 250, 500, 1000, 2000]
    arch = [N_FEATURES, 32, 16, 3]
    lr = 0.01

    train_accs, test_accs = [], []
    for ep in epoch_values:
        log_dir = os.path.join(RUNS_DIR, "exp2_epochs", f"epochs_{ep}")
        tr, te = train_eval(X_train, Y_train, X_test, Y_test, arch, ep, lr, log_dir)
        train_accs.append(tr); test_accs.append(te)
        print(f"  epochs={ep:<6} train={tr:.1f}% test={te:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epoch_values, train_accs, "o-", label="Train", color="steelblue")
    ax.plot(epoch_values, test_accs, "o-", label="Test", color="orange")
    ax.set_xscale("log")
    ax.set_xlabel("Epochs (log scale)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Experiment 2 — Underfitting to overfitting over epochs")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "exp2_epochs")


def experiment_3_depth(data):
    print("\n=== Experiment 3 — Architecture depth ===")
    (X_train, Y_train, _), (X_test, Y_test, _) = data
    architectures = [
        [N_FEATURES, 3],
        [N_FEATURES, 8, 3],
        [N_FEATURES, 16, 3],
        [N_FEATURES, 32, 16, 3],
        [N_FEATURES, 64, 32, 16, 3],
        [N_FEATURES, 128, 64, 32, 3],
    ]
    epochs = 300
    lr = 0.01

    labels = [str(a[1:-1]) for a in architectures]
    train_accs, test_accs = [], []
    for arch in architectures:
        label = "x".join(str(s) for s in arch)
        log_dir = os.path.join(RUNS_DIR, "exp3_depth", label)
        tr, te = train_eval(X_train, Y_train, X_test, Y_test, arch, epochs, lr, log_dir)
        train_accs.append(tr); test_accs.append(te)
        print(f"  arch={arch} train={tr:.1f}% test={te:.1f}%")

    x = np.arange(len(architectures))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, train_accs, w, label="Train", color="steelblue")
    ax.bar(x + w/2, test_accs, w, label="Test", color="orange")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel("Hidden layers"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Experiment 3 — Impact of architecture depth")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "exp3_depth")


def experiment_4_dataset_size():
    print("\n=== Experiment 4 — Dataset size ===")
    X_big, Y_big, labels_big = load_felidae(max_per_class=3000)
    y_idx = np.argmax(Y_big, axis=1)

    sample_sizes = [50, 100, 200, 500, 1000, 2000, 3000]
    arch = [N_FEATURES, 32, 16, 3]
    epochs = 200
    lr = 0.01

    train_accs, test_accs, gaps = [], [], []
    for n in sample_sizes:
        X_tr_parts, Y_tr_parts, X_te_parts, Y_te_parts = [], [], [], []
        for c in range(3):
            idx_c = np.where(y_idx == c)[0].copy()
            np.random.shuffle(idx_c)
            X_tr_parts.append(X_big[idx_c[:n]])
            Y_tr_parts.append(Y_big[idx_c[:n]])
            X_te_parts.append(X_big[idx_c[n:n + 200]])
            Y_te_parts.append(Y_big[idx_c[n:n + 200]])
        Xtr = np.concatenate(X_tr_parts); Ytr = np.concatenate(Y_tr_parts)
        Xte = np.concatenate(X_te_parts); Yte = np.concatenate(Y_te_parts)

        log_dir = os.path.join(RUNS_DIR, "exp4_dataset_size", f"n_{n}")
        tr, te = train_eval(Xtr, Ytr, Xte, Yte, arch, epochs, lr, log_dir)
        train_accs.append(tr); test_accs.append(te); gaps.append(tr - te)
        print(f"  n={n:<6}/class train={tr:.1f}% test={te:.1f}% gap={tr-te:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(sample_sizes, train_accs, "o-", label="Train", color="steelblue")
    ax1.plot(sample_sizes, test_accs, "o-", label="Test", color="orange")
    ax1.set_xlabel("Samples per class"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy vs dataset size"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(sample_sizes, gaps, "o-", color="red", label="Train - Test gap")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Samples per class"); ax2.set_ylabel("Gap (%)")
    ax2.set_title("Overfitting gap vs dataset size"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("Experiment 4 — Dataset size impact", fontsize=13)
    save_plot(fig, "exp4_dataset_size")


def experiment_5_overfitting():
    print("\n=== Experiment 5 — Overfitting demonstration ===")
    X_small, Y_small, labels_small = load_felidae(max_per_class=50)
    (X_train, Y_train, _), (X_test, Y_test, _) = split_train_test(X_small, Y_small, labels_small)

    arch = [N_FEATURES, 128, 64, 32, 3]
    epoch_values = [10, 25, 50, 100, 250, 500, 1000, 2000]
    lr = 0.01

    train_accs, test_accs = [], []
    for ep in epoch_values:
        log_dir = os.path.join(RUNS_DIR, "exp5_overfitting", f"epochs_{ep}")
        tr, te = train_eval(X_train, Y_train, X_test, Y_test, arch, ep, lr, log_dir)
        train_accs.append(tr); test_accs.append(te)
        print(f"  epochs={ep:<6} train={tr:.1f}% test={te:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epoch_values, train_accs, "o-", label="Train", color="steelblue")
    ax.plot(epoch_values, test_accs, "o-", label="Test", color="orange")
    ax.fill_between(epoch_values, train_accs, test_accs,
                    where=[a > b for a, b in zip(train_accs, test_accs)],
                    alpha=0.15, color="red", label="Overfitting zone")
    ax.set_xscale("log")
    ax.set_xlabel("Epochs (log scale)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Experiment 5 — Overfitting (arch={arch}, 50 samples/class)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "exp5_overfitting")


def experiment_6_underfitting(data):
    print("\n=== Experiment 6 — Underfitting demonstration ===")
    (X_train, Y_train, _), (X_test, Y_test, _) = data
    arch = [N_FEATURES, 2, 3]
    epoch_values = [50, 100, 250, 500, 1000, 2000, 5000]
    lr = 0.01

    train_accs, test_accs = [], []
    for ep in epoch_values:
        log_dir = os.path.join(RUNS_DIR, "exp6_underfitting", f"epochs_{ep}")
        tr, te = train_eval(X_train, Y_train, X_test, Y_test, arch, ep, lr, log_dir)
        train_accs.append(tr); test_accs.append(te)
        print(f"  epochs={ep:<6} train={tr:.1f}% test={te:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epoch_values, train_accs, "o-", label="Train", color="steelblue")
    ax.plot(epoch_values, test_accs, "o-", label="Test", color="orange")
    ax.axhline(33.3, color="gray", linestyle="--", linewidth=0.8, label="Random baseline (33%)")
    ax.set_xscale("log")
    ax.set_xlabel("Epochs (log scale)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Experiment 6 — Underfitting (arch={arch})")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "exp6_underfitting")


def experiment_7_sweet_spot(data):
    print("\n=== Experiment 7 — Sweet spot per architecture ===")
    (X_train, Y_train, _), (X_test, Y_test, _) = data
    architectures = [
        [N_FEATURES, 8, 3],
        [N_FEATURES, 32, 16, 3],
        [N_FEATURES, 64, 32, 3],
        [N_FEATURES, 128, 64, 32, 3],
    ]
    epoch_values = [50, 100, 250, 500, 1000]
    lr = 0.01

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, arch in enumerate(architectures):
        train_accs, test_accs = [], []
        for ep in epoch_values:
            label = "x".join(str(s) for s in arch)
            log_dir = os.path.join(RUNS_DIR, "exp7_sweet_spot", f"{label}_ep{ep}")
            tr, te = train_eval(X_train, Y_train, X_test, Y_test, arch, ep, lr, log_dir)
            train_accs.append(tr); test_accs.append(te)
        axes[i].plot(epoch_values, train_accs, "o-", label="Train", color="steelblue")
        axes[i].plot(epoch_values, test_accs, "o-", label="Test", color="orange")
        axes[i].set_xscale("log")
        axes[i].set_title(f"arch={arch[1:-1]}")
        axes[i].set_xlabel("Epochs"); axes[i].set_ylabel("Accuracy (%)")
        axes[i].legend(); axes[i].grid(True, alpha=0.3)
        print(f"  arch={arch} done")

    fig.suptitle("Experiment 7 — Finding the sweet spot per architecture", fontsize=13)
    save_plot(fig, "exp7_sweet_spot")


def experiment_8_confusion(data):
    print("\n=== Experiment 8 — Confusion matrix (best config) ===")
    (X_train, Y_train, _), (X_test, Y_test, labels_test) = data
    arch = [N_FEATURES, 32, 16, 3]
    epochs = 100
    lr = 0.01

    log_dir = os.path.join(RUNS_DIR, "exp8_confusion", "best")
    preds = predict_labels(X_train, Y_train, X_test, arch, epochs, lr, log_dir)
    true = np.argmax(Y_test, axis=1)

    n = 3
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true, preds):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Experiment 8 — Confusion matrix (best MLP)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    save_plot(fig, "exp8_confusion")

    overall = np.mean(preds == true) * 100
    print(f"  Overall test accuracy: {overall:.1f}%")
    for i, name in enumerate(CLASS_NAMES):
        mask = true == i
        if mask.sum() > 0:
            print(f"    {name}: {np.mean(preds[mask] == i) * 100:.1f}%")


def main():
    print("Loading main dataset (1000/class)...")
    X, Y, labels = load_felidae(max_per_class=1000)
    print(f"Loaded {len(X)} samples")
    data = split_train_test(X, Y, labels, test_ratio=0.2)
    (X_train, _, _), (X_test, _, _) = data
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    experiment_1_learning_rate(data)
    experiment_2_epochs(data)
    experiment_3_depth(data)
    experiment_4_dataset_size()
    experiment_5_overfitting()
    experiment_6_underfitting(data)
    experiment_7_sweet_spot(data)
    experiment_8_confusion(data)

    print("\n" + "=" * 60)
    print("All experiments complete.")
    print(f"Plots saved to : {PLOTS_DIR}")
    print(f"TB logs saved to: {RUNS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
