import os
import numpy as np
import matplotlib.pyplot as plt
from cffi import FFI
from PIL import Image

ffi = FFI()
ffi.cdef("""
    void* svm_train(const float* x_data, size_t x_rows, size_t x_cols,
                    const uint32_t* labels, size_t n_labels, size_t n_classes,
                    uint32_t kernel_type, float gamma, float c);
    void svm_predict_classes(const void* model, const float* x_data, size_t x_rows, size_t x_cols,
                             uint32_t* out_predictions);
    void svm_destroy(void* model);
    void extract_features(const float* image_data, float* out, size_t out_len);
""", override=True)

BASE = os.path.abspath("felidae_classifier")
lib = ffi.dlopen(os.path.join(BASE, "target/release/libfelidae_classifier.so"))

N_FEATURES = 3072
IMG_SIZE = (32, 32)
DATASET_ROOT = os.path.join(BASE, "dataset_clean")
CLASS_NAMES = ["Cat", "Cheetah", "Lion"]

KERNEL_LINEAR = 0
KERNEL_RBF = 1

PLOTS_DIR = os.path.abspath("report_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

np.random.seed(42)


def load_felidae(max_per_class):
    class_folders = {"cat": 0, "cheetah": 1, "lion": 2}
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
                X.append(arr)
                Y.append(label)
            except Exception as e:
                print(f"  Failed on {fname}: {e}")
                continue
    X = np.array(X, dtype=np.float32)
    return X, np.array(Y, dtype=np.uint32)


def split_train_test(X, labels, test_ratio=0.5):
    idx = np.random.permutation(len(X))
    split = int((1 - test_ratio) * len(X))
    tr, te = idx[:split], idx[split:]
    return (X[tr], labels[tr]), (X[te], labels[te])


def train_svm(X_train, labels_train, kernel_type, gamma, c, n_classes=3):
    X_c = np.ascontiguousarray(X_train, dtype=np.float32)
    labels_c = np.ascontiguousarray(labels_train, dtype=np.uint32)
    return lib.svm_train(
        ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1],
        ffi.from_buffer("uint32_t[]", labels_c), labels_c.shape[0], n_classes,
        kernel_type, float(gamma), float(c)
    )


def accuracy_of(model, X, labels):
    X_c = np.ascontiguousarray(X, dtype=np.float32)
    out = ffi.new("uint32_t[]", X_c.shape[0])
    lib.svm_predict_classes(model, ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1], out)
    preds = np.array([out[i] for i in range(X_c.shape[0])])
    return float(np.mean(preds == labels) * 100.0), preds


def save_plot(fig, name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def experiment_gamma_sweep(data):
    print("\n=== SVM Experiment 1 — Gamma sweep (RBF, C=1.0) ===")
    (X_train, l_train), (X_test, l_test) = data
    gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    c = 1.0

    train_accs, test_accs = [], []
    for g in gammas:
        model = train_svm(X_train, l_train, KERNEL_RBF, g, c)
        tr, _ = accuracy_of(model, X_train, l_train)
        te, _ = accuracy_of(model, X_test, l_test)
        lib.svm_destroy(model)
        train_accs.append(tr)
        test_accs.append(te)
        print(f"  gamma={g:<7} train={tr:.1f}% test={te:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gammas, train_accs, "o-", label="Train", color="steelblue")
    ax.plot(gammas, test_accs, "o-", label="Test", color="orange")
    ax.set_xscale("log")
    ax.set_xlabel("Gamma (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SVM — Impact of gamma (RBF kernel, C=1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "svm_gamma_sweep")


def experiment_c_sweep(data):
    print("\n=== SVM Experiment 2 — C sweep (RBF, gamma=0.01) ===")
    (X_train, l_train), (X_test, l_test) = data
    c_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
    gamma = 0.01

    train_accs, test_accs = [], []
    for c in c_values:
        model = train_svm(X_train, l_train, KERNEL_RBF, gamma, c)
        tr, _ = accuracy_of(model, X_train, l_train)
        te, _ = accuracy_of(model, X_test, l_test)
        lib.svm_destroy(model)
        train_accs.append(tr)
        test_accs.append(te)
        print(f"  C={c:<7} train={tr:.1f}% test={te:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(c_values, train_accs, "o-", label="Train", color="steelblue")
    ax.plot(c_values, test_accs, "o-", label="Test", color="orange")
    ax.set_xscale("log")
    ax.set_xlabel("C, soft-margin penalty (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SVM — Impact of C (RBF kernel, gamma=0.01)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "svm_c_sweep")


def experiment_gamma_c_heatmap(data):
    print("\n=== SVM Experiment 3 — Gamma x C heatmap (test accuracy) ===")
    (X_train, l_train), (X_test, l_test) = data
    gammas = [0.001, 0.01, 0.1, 1.0]
    c_values = [0.01, 0.1, 1.0, 10.0]

    grid = np.zeros((len(gammas), len(c_values)))
    best = (0.0, gammas[0], c_values[0])
    for gi, g in enumerate(gammas):
        for ci, c in enumerate(c_values):
            model = train_svm(X_train, l_train, KERNEL_RBF, g, c)
            te, _ = accuracy_of(model, X_test, l_test)
            lib.svm_destroy(model)
            grid[gi][ci] = te
            if te > best[0]:
                best = (te, g, c)
            print(f"  gamma={g:<6} C={c:<6} test={te:.1f}%")

    print(f"  Best: gamma={best[1]}, C={best[2]} -> {best[0]:.1f}% test")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(c_values)))
    ax.set_yticks(range(len(gammas)))
    ax.set_xticklabels([str(c) for c in c_values])
    ax.set_yticklabels([str(g) for g in gammas])
    ax.set_xlabel("C")
    ax.set_ylabel("Gamma")
    ax.set_title("SVM — Test accuracy over gamma x C")
    for gi in range(len(gammas)):
        for ci in range(len(c_values)):
            ax.text(ci, gi, f"{grid[gi][ci]:.0f}", ha="center", va="center",
                    color="white" if grid[gi][ci] < grid.max() * 0.7 else "black")
    fig.colorbar(im, label="Test accuracy (%)")
    save_plot(fig, "svm_gamma_c_heatmap")

    return best[1], best[2]


def experiment_linear_vs_rbf(data):
    print("\n=== SVM Experiment 4 — Linear vs RBF kernel ===")
    (X_train, l_train), (X_test, l_test) = data
    c = 1.0

    results = {}

    try:
        model = train_svm(X_train, l_train, KERNEL_LINEAR, 0.0, c)
        tr, _ = accuracy_of(model, X_train, l_train)
        te, _ = accuracy_of(model, X_test, l_test)
        lib.svm_destroy(model)
        results["Linear"] = (tr, te)
        print(f"  Linear  train={tr:.1f}% test={te:.1f}%")
    except Exception as e:
        print(f"  Linear kernel failed: {e}")
        results["Linear"] = (0.0, 0.0)

    model = train_svm(X_train, l_train, KERNEL_RBF, 0.01, c)
    tr, _ = accuracy_of(model, X_train, l_train)
    te, _ = accuracy_of(model, X_test, l_test)
    lib.svm_destroy(model)
    results["RBF"] = (tr, te)
    print(f"  RBF     train={tr:.1f}% test={te:.1f}%")

    labels_bar = list(results.keys())
    train_vals = [results[k][0] for k in labels_bar]
    test_vals = [results[k][1] for k in labels_bar]

    x = np.arange(len(labels_bar))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, train_vals, w, label="Train", color="steelblue")
    ax.bar(x + w/2, test_vals, w, label="Test", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SVM — Linear vs RBF kernel (C=1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "svm_linear_vs_rbf")


def experiment_confusion(data, gamma, c):
    print("\n=== SVM Experiment 5 — Confusion matrix (best config from heatmap) ===")
    (X_train, l_train), (X_test, l_test) = data

    model = train_svm(X_train, l_train, KERNEL_RBF, gamma, c)
    te, preds = accuracy_of(model, X_test, l_test)
    lib.svm_destroy(model)

    n = 3
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(l_test, preds):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"SVM — Confusion matrix (RBF, gamma={gamma}, C={c})")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    save_plot(fig, "svm_confusion")

    print(f"  Overall test accuracy: {te:.1f}%")
    for i, name in enumerate(CLASS_NAMES):
        mask = l_test == i
        if mask.sum() > 0:
            print(f"    {name}: {np.mean(preds[mask] == i) * 100:.1f}%")


def main():
    print("Loading dataset (200/class, flatten pixels)...")
    X, labels = load_felidae(max_per_class=200)
    print(f"Loaded {len(X)} samples")

    data = split_train_test(X, labels, test_ratio=0.5)
    (X_train, _), (X_test, _) = data
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    experiment_gamma_sweep(data)
    experiment_c_sweep(data)
    best_gamma, best_c = experiment_gamma_c_heatmap(data)
    experiment_linear_vs_rbf(data)
    experiment_confusion(data, best_gamma, best_c)

    print("\n" + "=" * 60)
    print("All SVM experiments complete.")
    print(f"Plots saved to: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
