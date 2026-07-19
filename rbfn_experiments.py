import os
import numpy as np
import matplotlib.pyplot as plt
from cffi import FFI
from PIL import Image

ffi = FFI()
ffi.cdef("""
    void* rbfn_train(const float* x_data, size_t x_rows, size_t x_cols,
                     const float* y_data, size_t y_rows, size_t y_cols,
                     size_t k, float gamma, size_t kmeans_iters, uint64_t seed);
    void* rbfn_train_naive(const float* x_data, size_t x_rows, size_t x_cols,
                           const float* y_data, size_t y_rows, size_t y_cols,
                           float gamma);
    void rbfn_predict_classes(const void* model, const float* x_data, size_t x_rows, size_t x_cols,
                              uint32_t* out_predictions);
    void rbfn_destroy(void* model);
    void extract_features(const float* image_data, float* out, size_t out_len);
""", override=True)

BASE = os.path.abspath("felidae_classifier")
lib = ffi.dlopen(os.path.join(BASE, "target/release/libfelidae_classifier.so"))


FEATURE_MODE = "flatten"
FLATTEN_DIM = 32
FLATTEN_GRAYSCALE = False

IMG_SIZE = (32, 32)
DATASET_ROOT = os.path.join(BASE, "dataset_clean")
CLASS_NAMES = ["Cat", "Lion", "Cheetah"]

if FEATURE_MODE == "extract":
    N_FEATURES = 20
elif FEATURE_MODE == "flatten":
    channels = 1 if FLATTEN_GRAYSCALE else 3
    N_FEATURES = FLATTEN_DIM * FLATTEN_DIM * channels
else:
    raise ValueError(f"Unknown FEATURE_MODE: {FEATURE_MODE}")

print(f"Feature mode: {FEATURE_MODE} ({N_FEATURES} features)")

PLOTS_DIR = os.path.abspath("report_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

np.random.seed(42)


def features_from_image(path):

    if FEATURE_MODE == "extract":
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        arr = np.ascontiguousarray(np.array(img, dtype=np.float32).flatten() / 255.0, dtype=np.float32)
        out = ffi.new("float[]", N_FEATURES)
        lib.extract_features(ffi.from_buffer("float[]", arr), out, N_FEATURES)
        return np.array([out[i] for i in range(N_FEATURES)], dtype=np.float32)
    else:
        mode = "L" if FLATTEN_GRAYSCALE else "RGB"
        img = Image.open(path).convert(mode).resize((FLATTEN_DIM, FLATTEN_DIM))
        return np.array(img, dtype=np.float32).flatten() / 255.0


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
                X.append(features_from_image(os.path.join(folder_path, fname)))
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


def accuracy_of(model, X, Y_onehot):
    X_c = np.ascontiguousarray(X, dtype=np.float32)
    out = ffi.new("uint32_t[]", X_c.shape[0])
    lib.rbfn_predict_classes(model, ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1], out)
    preds = np.array([out[i] for i in range(X_c.shape[0])])
    true = np.argmax(Y_onehot, axis=1)
    return float(np.mean(preds == true) * 100.0), preds


def train_kcenters(X_train, Y_train, k, gamma, kmeans_iters=20, seed=0):
    X_c = np.ascontiguousarray(X_train, dtype=np.float32)
    Y_c = np.ascontiguousarray(Y_train, dtype=np.float32)
    return lib.rbfn_train(
        ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1],
        ffi.from_buffer("float[]", Y_c), Y_c.shape[0], Y_c.shape[1],
        k, gamma, kmeans_iters, seed
    )


def train_naive(X_train, Y_train, gamma):
    X_c = np.ascontiguousarray(X_train, dtype=np.float32)
    Y_c = np.ascontiguousarray(Y_train, dtype=np.float32)
    return lib.rbfn_train_naive(
        ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1],
        ffi.from_buffer("float[]", Y_c), Y_c.shape[0], Y_c.shape[1],
        gamma
    )


def save_plot(fig, name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")



def experiment_gamma_sweep(data):
    print("\n=== RBFN Experiment 1 — Gamma sweep ===")
    (X_train, Y_train, _), (X_test, Y_test, _) = data
    gammas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    k = 50

    train_accs, test_accs = [], []
    for g in gammas:
        model = train_kcenters(X_train, Y_train, k, g)
        tr, _ = accuracy_of(model, X_train, Y_train)
        te, _ = accuracy_of(model, X_test, Y_test)
        lib.rbfn_destroy(model)
        train_accs.append(tr)
        test_accs.append(te)
        print(f"  gamma={g:<7} train={tr:.1f}% test={te:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gammas, train_accs, "o-", label="Train", color="steelblue")
    ax.plot(gammas, test_accs, "o-", label="Test", color="orange")
    ax.set_xscale("log")
    ax.set_xlabel("Gamma (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"RBFN — Impact of gamma (K={k} centers)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "rbfn_gamma_sweep")


def experiment_k_sweep(data):
    print("\n=== RBFN Experiment 2 — K (number of centers) sweep ===")
    (X_train, Y_train, _), (X_test, Y_test, _) = data
    k_values = [2, 5, 10, 25, 50, 100, 200, 400]
    gamma = 0.1

    train_accs, test_accs = [], []
    for k in k_values:
        model = train_kcenters(X_train, Y_train, k, gamma)
        tr, _ = accuracy_of(model, X_train, Y_train)
        te, _ = accuracy_of(model, X_test, Y_test)
        lib.rbfn_destroy(model)
        train_accs.append(tr)
        test_accs.append(te)
        print(f"  K={k:<5} train={tr:.1f}% test={te:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_values, train_accs, "o-", label="Train", color="steelblue")
    ax.plot(k_values, test_accs, "o-", label="Test", color="orange")
    ax.set_xlabel("Number of centers K")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"RBFN — Impact of number of centers (gamma={gamma})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "rbfn_k_sweep")


def experiment_naive_vs_kcenters():
    print("\n=== RBFN Experiment 3 — Naive vs K-centers ===")

    X, Y, labels = load_felidae(max_per_class=100)
    (X_train, Y_train, _), (X_test, Y_test, _) = split_train_test(X, Y, labels)
    gamma = 0.1


    naive = train_naive(X_train, Y_train, gamma)
    naive_train, _ = accuracy_of(naive, X_train, Y_train)
    naive_test, _ = accuracy_of(naive, X_test, Y_test)
    lib.rbfn_destroy(naive)
    print(f"  Naive (N={len(X_train)} centers): train={naive_train:.1f}% test={naive_test:.1f}%")


    k = 30
    kc = train_kcenters(X_train, Y_train, k, gamma)
    kc_train, _ = accuracy_of(kc, X_train, Y_train)
    kc_test, _ = accuracy_of(kc, X_test, Y_test)
    lib.rbfn_destroy(kc)
    print(f"  K-centers (K={k}):        train={kc_train:.1f}% test={kc_test:.1f}%")

    labels_bar = [f"Naive\n(N={len(X_train)})", f"K-centers\n(K={k})"]
    train_vals = [naive_train, kc_train]
    test_vals = [kc_train and naive_test, kc_test]
    test_vals = [naive_test, kc_test]

    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, train_vals, w, label="Train", color="steelblue")
    ax.bar(x + w/2, test_vals, w, label="Test", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("RBFN — Naive (memorization) vs K-centers (generalization)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "rbfn_naive_vs_kcenters")


def experiment_confusion(data):
    print("\n=== RBFN Experiment 4 — Confusion matrix (best config) ===")
    (X_train, Y_train, _), (X_test, Y_test, labels_test) = data
    k = 50
    gamma = 0.1

    model = train_kcenters(X_train, Y_train, k, gamma)
    te, preds = accuracy_of(model, X_test, Y_test)
    lib.rbfn_destroy(model)
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
    ax.set_title(f"RBFN — Confusion matrix (K={k}, gamma={gamma})")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    save_plot(fig, "rbfn_confusion")

    print(f"  Overall test accuracy: {te:.1f}%")
    for i, name in enumerate(CLASS_NAMES):
        mask = true == i
        if mask.sum() > 0:
            print(f"    {name}: {np.mean(preds[mask] == i) * 100:.1f}%")


def main():
    print("Loading main dataset (3000/class)...")
    X, Y, labels = load_felidae(max_per_class=3000)
    print(f"Loaded {len(X)} samples")
    data = split_train_test(X, Y, labels, test_ratio=0.2)
    (X_train, _, _), (X_test, _, _) = data
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    experiment_gamma_sweep(data)
    experiment_k_sweep(data)
    experiment_naive_vs_kcenters()
    experiment_confusion(data)

    print("\n" + "=" * 60)
    print("All RBFN experiments complete.")
    print(f"Plots saved to: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
