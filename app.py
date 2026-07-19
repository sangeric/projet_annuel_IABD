import gradio as gr
import numpy as np
from PIL import Image
from cffi import FFI
import os
import json

ffi = FFI()
ffi.cdef("""
    void* mlp_create(const uint32_t* layer_sizes, size_t n_layers, const char* activation, const char* output_activation);
    void mlp_train(void* mlp, const float* x_data, size_t x_rows, size_t x_cols, const float* y_data, size_t y_rows, size_t y_cols, size_t epochs, float learning_rate);
    void mlp_predict_classes(const void* mlp, const float* x_data, size_t x_rows, size_t x_cols, uint32_t* out_predictions);
    void mlp_predict_raw(const void* mlp, const float* x_data, size_t x_rows, size_t x_cols, float* out_predictions);
    int mlp_save(const void* mlp, const char* path);
    void* mlp_load(const char* path);
    void mlp_destroy(void* mlp);
    void extract_features(const float* image_data, float* out, size_t out_len);

    void* rbfn_train(const float* x_data, size_t x_rows, size_t x_cols,
                     const float* y_data, size_t y_rows, size_t y_cols,
                     size_t k, float gamma, size_t kmeans_iters, uint64_t seed);
    void rbfn_predict_classes(const void* model, const float* x_data, size_t x_rows, size_t x_cols,
                              uint32_t* out_predictions);
    int rbfn_save(const void* model, const char* path);
    void* rbfn_load(const char* path);
    void rbfn_destroy(void* model);

    void* svm_train(const float* x_data, size_t x_rows, size_t x_cols,
                    const uint32_t* labels, size_t n_labels, size_t n_classes,
                    uint32_t kernel_type, float gamma, float c);
    void svm_predict_classes(const void* model, const float* x_data, size_t x_rows, size_t x_cols,
                             uint32_t* out_predictions);
    int svm_save(const void* model, const char* path);
    void* svm_load(const char* path);
    void svm_destroy(void* model);
    
    void* create_rosenblatt_classifier(
        size_t n_features,
        float learning_rate,
        float bias_cat,
        float bias_lion,
        float bias_cheetah,
        uint64_t seed
    );
    void train_classifier_rosen(
        void* classifier,
        const float* x,
        size_t rows,
        size_t cols,
        const size_t* y,
        size_t y_len,
        size_t epochs
    );
    size_t* predict_classifier(
        void* classifier,
        const float* x,
        size_t rows,
        size_t cols,
        const size_t* y,
        size_t y_len
    );
    float* get_weights_classifier(void* classifier);

    typedef struct {
        float bias;
        float learning_rate;
    } RosenblattParams;

    typedef struct RosenblattClassifier RosenblattClassifier;

    RosenblattClassifier* rosenblatt_load(
        const char* path,
        uint64_t* out_seeds,
        RosenblattParams* out_params,
        size_t* out_rows,
        size_t* out_cols,
        float* out_cat_weights,
        float* out_lion_weights,
        float* out_cheetah_weights
    );
    int rosenblatt_save(const void* classifier, const char* path, const uint64_t* seeds);
    
""", override=True)

BASE = os.path.abspath("felidae_classifier")
if os.name != "nt":
    lib = ffi.dlopen(os.path.join(BASE, "target/release/libfelidae_classifier.so"))
else:
    lib = ffi.dlopen(os.path.join(BASE, "target/release/felidae_classifier.dll"))

MODEL_DIR = os.path.join(BASE, "saved_models")
for subfolder in ["mlp", "linear", "svm", "rbfn"]:
    os.makedirs(os.path.join(MODEL_DIR, subfolder), exist_ok=True)

CLASS_NAMES = ["Cat", "Lion", "Cheetah"]
N_FEATURES = 20
N_FLATTEN = 3072
IMG_SIZE = (32, 32)
DATASET_ROOT = os.path.join(BASE, "dataset_clean")
DATASET_ROOT_NOTCLEANED = os.path.join(BASE, "dataset")

# --- Metadata helpers ---
# Each model stores its own metadata (which feature mode it was trained with,
# plus any model-specific hyperparameters) so inference can reproduce the exact
# preprocessing used at training time.
def save_model_meta(model_name, meta):
    with open(os.path.join(MODEL_DIR, model_name, f"{model_name}_meta.json"), "w") as f:
        json.dump(meta, f)

def load_model_meta(model_name):
    meta_path = os.path.join(MODEL_DIR, model_name, f"{model_name}_meta.json")
    if not os.path.exists(meta_path):
        return {"feature_mode": "Extract (20 features)"}
    with open(meta_path) as f:
        return json.load(f)


def preprocess(pil_image, use_extract=True):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32).flatten() / 255.0
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    if use_extract:
        out = ffi.new("float[]", N_FEATURES)
        lib.extract_features(ffi.from_buffer("float[]", arr), out, N_FEATURES)
        result = np.array([out[i] for i in range(N_FEATURES)], dtype=np.float32)
    else:
        result = arr

    return np.ascontiguousarray(result.reshape(1, -1), dtype=np.float32)


def load_dataset(max_samples=3000, use_extract=True):
    class_folders = {"cat": 0, "lion": 1, "cheetah": 2}
    X, Y = [], []
    for folder, label in class_folders.items():
        folder_path = os.path.join(DATASET_ROOT, folder)
        if not os.path.exists(folder_path):
            continue
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
        files = files[:max_samples]
        for fname in files:
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert("RGB").resize(IMG_SIZE)
                arr = np.array(img, dtype=np.float32).flatten() / 255.0
                arr = np.ascontiguousarray(arr, dtype=np.float32)

                if use_extract:
                    out = ffi.new("float[]", N_FEATURES)
                    lib.extract_features(ffi.from_buffer("float[]", arr), out, N_FEATURES)
                    features = np.array([out[i] for i in range(N_FEATURES)], dtype=np.float32)
                else:
                    features = arr

                X.append(features)
                Y.append(label)
            except Exception as e:
                print(f"Failed on {fname}: {e}")
                continue

    X = np.array(X, dtype=np.float32)
    Y_onehot = np.zeros((len(Y), 3), dtype=np.float32)
    for i, label in enumerate(Y):
        Y_onehot[i] = [-1, -1, -1]
        Y_onehot[i][label] = 1
    return X, Y_onehot


def load_dataset_linear(max_samples=3000, use_extract=True):
    class_folders = {"cat": 0, "lion": 1, "cheetah": 2}
    X, Y = [], []
    for folder, label in class_folders.items():
        folder_path = os.path.join(DATASET_ROOT, folder)
        if not os.path.exists(folder_path):
            continue
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
        files = files[:max_samples]
        for fname in files:
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert("RGB").resize(IMG_SIZE)
                arr = np.array(img, dtype=np.float32).flatten() / 255.0
                arr = np.ascontiguousarray(arr, dtype=np.float32)

                if use_extract:
                    out = ffi.new("float[]", N_FEATURES)
                    lib.extract_features(ffi.from_buffer("float[]", arr), out, N_FEATURES)
                    features = np.array([out[i] for i in range(N_FEATURES)], dtype=np.float32)
                else:
                    features = arr

                X.append(features)
                Y.append(label)
            except Exception as e:
                print(f"Failed on {fname}: {e}")
                continue

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.uint32)

    return X, Y

def train_and_save_mlp(epochs, learning_rate, max_samples_per_class, hidden_layers_str, feature_mode, progress=gr.Progress()):
    use_extract = "Extract" in feature_mode
    n_inputs = N_FEATURES if use_extract else N_FLATTEN

    progress(0, desc="Loading dataset...")
    X, Y = load_dataset(max_samples=int(max_samples_per_class), use_extract=use_extract)
    if len(X) == 0:
        return "No dataset found at: " + DATASET_ROOT

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    try:
        hidden = [int(x.strip()) for x in hidden_layers_str.split(",")]
    except ValueError:
        return "Invalid hidden layers — use comma-separated integers e.g. '32, 16'"

    arch = [n_inputs] + hidden + [3]

    progress(0.1, desc="Creating model...")
    layers = ffi.new("uint32_t[]", arch)
    mlp = lib.mlp_create(layers, len(arch), b"tanh", b"tanh")
    if mlp == ffi.NULL:
        return "Failed to create MLP"

    X_ptr = ffi.from_buffer("float[]", np.ascontiguousarray(X_train, dtype=np.float32))
    Y_ptr = ffi.from_buffer("float[]", np.ascontiguousarray(Y_train, dtype=np.float32))

    progress(0.2, desc=f"Training {int(epochs)} epochs at lr={learning_rate}...")
    lib.mlp_train(
        mlp,
        X_ptr, X_train.shape[0], X_train.shape[1],
        Y_ptr, Y_train.shape[0], Y_train.shape[1],
        int(epochs), float(learning_rate)
    )

    def acc(Xset, Yset):
        Xc = np.ascontiguousarray(Xset, dtype=np.float32)
        out = ffi.new("uint32_t[]", Xc.shape[0])
        lib.mlp_predict_classes(mlp, ffi.from_buffer("float[]", Xc), Xc.shape[0], Xc.shape[1], out)
        preds = np.array([out[i] for i in range(Xc.shape[0])])
        true = np.argmax(Yset, axis=1)
        return float(np.mean(preds == true) * 100.0)

    train_acc = acc(X_train, Y_train)
    test_acc = acc(X_test, Y_test)

    save_path = os.path.join(MODEL_DIR, "mlp", "mlp.bin").encode()
    ret = lib.mlp_save(mlp, save_path)
    lib.mlp_destroy(mlp)

    if ret != 0:
        return "Training done but save failed"

    save_model_meta("mlp", {"feature_mode": feature_mode})
    progress(1.0, desc="Done")
    return (
        f"Training complete.\n"
        f"Input mode    : {feature_mode}\n"
        f"Architecture  : {arch}\n"
        f"Epochs        : {int(epochs)}\n"
        f"Learning rate : {learning_rate}\n"
        f"Samples       : {len(X_train)} train / {len(X_test)} test\n"
        f"Train accuracy: {train_acc:.1f}%\n"
        f"Test accuracy : {test_acc:.1f}%\n"
        f"Saved to      : {save_path.decode()}"
    )


def predict_mlp(X):
    model_path = os.path.join(MODEL_DIR, "mlp", "mlp.bin")
    if not os.path.exists(model_path):
        return None, "No trained MLP found — train it first in the Train tab"

    mlp = lib.mlp_load(model_path.encode())
    if mlp == ffi.NULL:
        return None, "Failed to load MLP"

    out_raw = ffi.new("float[]", 3)
    out_cls = ffi.new("uint32_t[]", 1)
    lib.mlp_predict_raw(mlp, ffi.from_buffer("float[]", X), 1, X.shape[1], out_raw)
    lib.mlp_predict_classes(mlp, ffi.from_buffer("float[]", X), 1, X.shape[1], out_cls)
    lib.mlp_destroy(mlp)

    scores_raw = np.array([float(out_raw[i]) for i in range(3)], dtype=np.float64)
    exp = np.exp(scores_raw - np.max(scores_raw))
    probs = exp / exp.sum()

    return {CLASS_NAMES[i]: round(float(probs[i]), 3) for i in range(3)}, None


def train_and_save_rbfn(k, gamma, kmeans_iters, max_samples_per_class, feature_mode, progress=gr.Progress()):
    use_extract = "Extract" in feature_mode

    progress(0, desc="Loading dataset...")
    X, Y = load_dataset(max_samples=int(max_samples_per_class), use_extract=use_extract)
    if len(X) == 0:
        return "No dataset found at: " + DATASET_ROOT

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    progress(0.3, desc=f"Training RBFN (K={int(k)}, gamma={gamma})...")
    X_c = np.ascontiguousarray(X_train, dtype=np.float32)
    Y_c = np.ascontiguousarray(Y_train, dtype=np.float32)
    model = lib.rbfn_train(
        ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1],
        ffi.from_buffer("float[]", Y_c), Y_c.shape[0], Y_c.shape[1],
        int(k), float(gamma), int(kmeans_iters), 0
    )
    if model == ffi.NULL:
        return "Failed to train RBFN"

    def acc(Xset, Yset):
        Xc = np.ascontiguousarray(Xset, dtype=np.float32)
        out = ffi.new("uint32_t[]", Xc.shape[0])
        lib.rbfn_predict_classes(model, ffi.from_buffer("float[]", Xc), Xc.shape[0], Xc.shape[1], out)
        preds = np.array([out[i] for i in range(Xc.shape[0])])
        true = np.argmax(Yset, axis=1)
        return float(np.mean(preds == true) * 100.0)

    train_acc = acc(X_train, Y_train)
    test_acc = acc(X_test, Y_test)

    save_path = os.path.join(MODEL_DIR, "rbfn", "rbfn.bin").encode()
    ret = lib.rbfn_save(model, save_path)
    lib.rbfn_destroy(model)

    if ret != 0:
        return "Training done but save failed"

    save_model_meta("rbfn", {"feature_mode": feature_mode, "k": int(k), "gamma": float(gamma)})
    progress(1.0, desc="Done")
    return (
        f"Training complete.\n"
        f"Input mode    : {feature_mode}\n"
        f"Centers K     : {int(k)}\n"
        f"Gamma         : {gamma}\n"
        f"Samples       : {len(X_train)} train / {len(X_test)} test\n"
        f"Train accuracy: {train_acc:.1f}%\n"
        f"Test accuracy : {test_acc:.1f}%\n"
        f"Saved to      : {save_path.decode()}"
    )


def predict_rbfn(X):
    model_path = os.path.join(MODEL_DIR, "rbfn", "rbfn.bin")
    if not os.path.exists(model_path):
        return None, "No trained RBFN found — train it first in the Train tab"

    model = lib.rbfn_load(model_path.encode())
    if model == ffi.NULL:
        return None, "Failed to load RBFN"

    out_cls = ffi.new("uint32_t[]", 1)
    lib.rbfn_predict_classes(model, ffi.from_buffer("float[]", X), 1, X.shape[1], out_cls)
    lib.rbfn_destroy(model)

    predicted = int(out_cls[0])
    # RBFN exposes only the class index, so we show a hard 1.0 on the winner.
    scores = {name: (1.0 if i == predicted else 0.0) for i, name in enumerate(CLASS_NAMES)}
    return scores, None


KERNEL_LINEAR = 0
KERNEL_RBF = 1


def train_and_save_svm(kernel_choice, gamma, c, max_samples_per_class, feature_mode, progress=gr.Progress()):
    use_extract = "Extract" in feature_mode
    kernel_type = KERNEL_RBF if "RBF" in kernel_choice else KERNEL_LINEAR

    progress(0, desc="Loading dataset...")
    X, Y = load_dataset(max_samples=int(max_samples_per_class), use_extract=use_extract)
    if len(X) == 0:
        return "No dataset found at: " + DATASET_ROOT

    # SVM labels are class indices, not one-hot. Recover them from the one-hot Y.
    labels = np.argmax(Y, axis=1).astype(np.uint32)

    # hold out 20% as a test set the model never trains on
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, labels_train = X[train_idx], labels[train_idx]
    X_test, labels_test = X[test_idx], labels[test_idx]

    progress(0.3, desc=f"Training SVM ({kernel_choice}, gamma={gamma}, C={c})...")
    X_c = np.ascontiguousarray(X_train, dtype=np.float32)
    labels_c = np.ascontiguousarray(labels_train, dtype=np.uint32)
    model = lib.svm_train(
        ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1],
        ffi.from_buffer("uint32_t[]", labels_c), labels_c.shape[0], 3,
        kernel_type, float(gamma), float(c)
    )
    if model == ffi.NULL:
        return "Failed to train SVM"

    def acc(Xset, labelset):
        Xc = np.ascontiguousarray(Xset, dtype=np.float32)
        out = ffi.new("uint32_t[]", Xc.shape[0])
        lib.svm_predict_classes(model, ffi.from_buffer("float[]", Xc), Xc.shape[0], Xc.shape[1], out)
        preds = np.array([out[i] for i in range(Xc.shape[0])])
        return float(np.mean(preds == labelset) * 100.0)

    train_acc = acc(X_train, labels_train)
    test_acc = acc(X_test, labels_test)

    save_path = os.path.join(MODEL_DIR, "svm", "svm.bin").encode()
    ret = lib.svm_save(model, save_path)
    lib.svm_destroy(model)

    if ret != 0:
        return "Training done but save failed"

    save_model_meta("svm", {
        "feature_mode": feature_mode,
        "kernel": kernel_choice,
        "gamma": float(gamma),
        "c": float(c),
    })
    progress(1.0, desc="Done")
    return (
        f"Training complete.\n"
        f"Input mode    : {feature_mode}\n"
        f"Kernel        : {kernel_choice}\n"
        f"Gamma         : {gamma}\n"
        f"C             : {c}\n"
        f"Samples       : {len(X_train)} train / {len(X_test)} test\n"
        f"Train accuracy: {train_acc:.1f}%\n"
        f"Test accuracy : {test_acc:.1f}%\n"
        f"Saved to      : {save_path.decode()}"
    )


def predict_svm(X):
    model_path = os.path.join(MODEL_DIR, "svm", "svm.bin")
    if not os.path.exists(model_path):
        return None, "No trained SVM found — train it first in the Train tab"

    model = lib.svm_load(model_path.encode())
    if model == ffi.NULL:
        return None, "Failed to load SVM"

    out_cls = ffi.new("uint32_t[]", 1)
    lib.svm_predict_classes(model, ffi.from_buffer("float[]", X), 1, X.shape[1], out_cls)
    lib.svm_destroy(model)

    predicted = int(out_cls[0])
    # SVM exposes only the class index, so we show a hard 1.0 on the winner.
    scores = {name: (1.0 if i == predicted else 0.0) for i, name in enumerate(CLASS_NAMES)}
    return scores, None


def train_and_save_rosenblatt(epochs, learning_rate, max_samples_per_class, feature_mode, seed, progress=gr.Progress()):
    use_extract = "Extract" in feature_mode
    n_features = N_FEATURES if use_extract else N_FLATTEN

    progress(0, desc="Loading dataset...")
    X, Y = load_dataset_linear(max_samples=int(max_samples_per_class), use_extract=use_extract)
    if len(X) == 0:
        return "No dataset found at: " + DATASET_ROOT

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    progress(0.1, desc="Creating classifier...")
    classifier = lib.create_rosenblatt_classifier(
        n_features, float(learning_rate),
        1.0, 1.0, 1.0,
        int(seed)
    )
    if classifier == ffi.NULL:
        return "Failed to create Rosenblatt classifier"

    X_c = np.ascontiguousarray(X_train, dtype=np.float32)
    Y_c = np.ascontiguousarray(Y_train, dtype=np.uint64)

    progress(0.2, desc=f"Training {int(epochs)} epochs at lr={learning_rate}...")
    lib.train_classifier_rosen(
        classifier,
        ffi.from_buffer("float[]", X_c), X_c.shape[0], X_c.shape[1],
        ffi.from_buffer("size_t[]", Y_c), len(Y_c),
        int(epochs)
    )

    def acc(Xset, Yset):
        Xc = np.ascontiguousarray(Xset, dtype=np.float32)
        Yc = np.ascontiguousarray(Yset, dtype=np.uint64)
        preds_ptr = lib.predict_classifier(
            classifier,
            ffi.from_buffer("float[]", Xc), Xc.shape[0], Xc.shape[1],
            ffi.from_buffer("size_t[]", Yc), len(Yc)
        )
        preds = np.array([preds_ptr[i] for i in range(Xc.shape[0])])
        return float(np.mean(preds == Yset) * 100.0)

    train_acc = acc(X_train, Y_train)
    test_acc = acc(X_test, Y_test)

    save_path = os.path.join(MODEL_DIR, "linear", "rosenblatt.bin").encode()
    seeds_buf = ffi.new("uint64_t[3]", [int(seed), int(seed), int(seed)])
    ret = lib.rosenblatt_save(classifier, save_path, seeds_buf)

    if ret != 0:
        return "Training done but save failed"

    save_model_meta("linear", {"feature_mode": feature_mode, "seed": int(seed)})
    progress(1.0, desc="Done")
    return (
        f"Training complete.\n"
        f"Input mode    : {feature_mode}\n"
        f"Features      : {n_features}\n"
        f"Epochs        : {int(epochs)}\n"
        f"Learning rate : {learning_rate}\n"
        f"Samples       : {len(X_train)} train / {len(X_test)} test\n"
        f"Train accuracy: {train_acc:.1f}%\n"
        f"Test accuracy : {test_acc:.1f}%\n"
        f"Saved to      : {save_path.decode()}"
    )


N_LINEAR_MAX = 3072

def predict_rosenblatt(X):
    model_path = os.path.join(MODEL_DIR, "linear", "rosenblatt.bin")
    if not os.path.exists(model_path):
        return None, "No trained Linear (Rosenblatt) model found — train it first in the Train tab"

    seeds = ffi.new("uint64_t[3]")
    params = ffi.new("RosenblattParams*")
    rows = ffi.new("size_t*")
    cols = ffi.new("size_t*")
    cat_w = ffi.new("float[]", N_LINEAR_MAX)
    lion_w = ffi.new("float[]", N_LINEAR_MAX)
    cheetah_w = ffi.new("float[]", N_LINEAR_MAX)

    classifier_ptr = lib.rosenblatt_load(
        model_path.encode(), seeds, params, rows, cols,
        cat_w, lion_w, cheetah_w
    )
    if classifier_ptr == ffi.NULL:
        return None, "Failed to load Rosenblatt classifier"

    Xc = np.ascontiguousarray(X, dtype=np.float32)
    dummy_y = np.zeros(1, dtype=np.uint64)

    preds_ptr = lib.predict_classifier(
        ffi.cast("void*", classifier_ptr),
        ffi.from_buffer("float[]", Xc), 1, Xc.shape[1],
        ffi.from_buffer("size_t[]", dummy_y), 1
    )

    predicted = int(preds_ptr[0])
    scores = {name: (1.0 if i == predicted else 0.0) for i, name in enumerate(CLASS_NAMES)}
    return scores, None

def predict(pil_image, model_choice):
    if pil_image is None:
        return None, "Please upload an image"

    if model_choice == "MLP":
        meta = load_model_meta("mlp")
        use_extract = "Extract" in meta["feature_mode"]
        X = preprocess(pil_image, use_extract=use_extract)
        return predict_mlp(X)
    elif model_choice == "RBFN":
        meta = load_model_meta("rbfn")
        use_extract = "Extract" in meta["feature_mode"]
        X = preprocess(pil_image, use_extract=use_extract)
        return predict_rbfn(X)
    elif model_choice == "Linear (Rosenblatt)":
        meta = load_model_meta("linear")
        use_extract = "Extract" in meta["feature_mode"]
        X = preprocess(pil_image, use_extract=use_extract)
        return predict_rosenblatt(X)
    elif model_choice == "SVM":
        meta = load_model_meta("svm")
        use_extract = "Extract" in meta["feature_mode"]
        X = preprocess(pil_image, use_extract=use_extract)
        return predict_svm(X)
    else:
        return None, f"{model_choice} not yet implemented"


def classify(pil_image, model_choice):
    scores, err = predict(pil_image, model_choice)
    if err:
        return None, f"⚠️ {err}"
    return scores, ""


def check_model_status():
    lines = []

    mlp_path = os.path.join(MODEL_DIR, "mlp", "mlp.bin")
    if os.path.exists(mlp_path):
        mode = load_model_meta("mlp")["feature_mode"]
        lines.append(f"✅ MLP — ready ({mode})")
    else:
        lines.append("❌ MLP — not trained")

    rbfn_path = os.path.join(MODEL_DIR, "rbfn", "rbfn.bin")
    if os.path.exists(rbfn_path):
        meta = load_model_meta("rbfn")
        lines.append(f"✅ RBFN - ready ({meta['feature_mode']}, K={meta.get('k', '?')}, gamma={meta.get('gamma', '?')})")
    else:
        lines.append("❌ RBFN — not trained")

    linear_path = os.path.join(MODEL_DIR, "linear", "rosenblatt.bin")
    if os.path.exists(linear_path):
        mode = load_model_meta("linear")["feature_mode"]
        lines.append(f"✅ Linear (Rosenblatt) - ready ({mode})")
    else:
        lines.append("❌ Linear (Rosenblatt) - not trained")

    svm_path = os.path.join(MODEL_DIR, "svm", "svm.bin")
    if os.path.exists(svm_path):
        meta = load_model_meta("svm")
        lines.append(f"✅ SVM — ready ({meta['feature_mode']}, {meta.get('kernel', '?')}, gamma={meta.get('gamma', '?')}, C={meta.get('c', '?')})")
    else:
        lines.append("❌ SVM — not trained")
    return "\n".join(lines)


with gr.Blocks(title="Felidae Classifier") as demo:
    gr.Markdown("# 🐱 Felidae Classifier")
    gr.Markdown("Classify images of cats, lions, and cheetahs using models built from scratch in Rust.")

    with gr.Tabs():

        with gr.Tab("🔍 Classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload image")
                    model_dropdown = gr.Dropdown(
                        choices=["MLP", "Linear (Rosenblatt)", "SVM", "RBFN"],
                        value="MLP",
                        label="Model"
                    )
                    classify_btn = gr.Button("Classify", variant="primary")

                with gr.Column(scale=1):
                    output_label = gr.Label(label="Prediction", num_top_classes=3)
                    error_text = gr.Markdown("")

            classify_btn.click(
                fn=classify,
                inputs=[image_input, model_dropdown],
                outputs=[output_label, error_text]
            )

        with gr.Tab("🏋️ Train"):
            with gr.Tabs():
                with gr.Tab("MLP"):
                    gr.Markdown("### Train and save an MLP")
                    gr.Markdown("The input and output layer sizes are set automatically based on the input mode.")

                    with gr.Row():
                        with gr.Column():
                            feature_mode = gr.Radio(
                                choices=["Extract (20 features)", "Flatten (3072 pixels)"],
                                value="Extract (20 features)",
                                label="Input mode"
                            )
                            hidden_layers_input = gr.Textbox(
                                value="32, 16",
                                label="Hidden layers (comma-separated sizes)"
                            )
                            arch_preview = gr.Textbox(
                                value="[20, 32, 16, 3]",
                                label="Full architecture (preview)",
                                interactive=False
                            )

                        with gr.Column():
                            epochs_slider = gr.Slider(minimum=10, maximum=10000, value=100, step=10, label="Epochs")
                            lr_slider = gr.Slider(minimum=0.001, maximum=0.5, value=0.01, step=0.001, label="Learning rate")
                            sample_size_slider = gr.Slider(minimum=100, maximum=5000, value=3000, step=100, label="Max samples per class")

                    train_btn = gr.Button("Train MLP", variant="primary")
                    train_output = gr.Textbox(label="Training log", interactive=False, lines=8)

                    def update_arch_preview(mode, hidden_str):
                        n_inputs = N_FEATURES if "Extract" in mode else N_FLATTEN
                        try:
                            hidden = [int(x.strip()) for x in hidden_str.split(",")]
                            arch = [n_inputs] + hidden + [3]
                            return str(arch)
                        except ValueError:
                            return "Invalid hidden layers"

                    def update_hidden_default(mode):
                        return "32, 16" if "Extract" in mode else "128, 64"

                    feature_mode.change(
                        fn=update_hidden_default,
                        inputs=[feature_mode],
                        outputs=[hidden_layers_input]
                    )
                    feature_mode.change(
                        fn=update_arch_preview,
                        inputs=[feature_mode, hidden_layers_input],
                        outputs=[arch_preview]
                    )
                    hidden_layers_input.change(
                        fn=update_arch_preview,
                        inputs=[feature_mode, hidden_layers_input],
                        outputs=[arch_preview]
                    )

                    train_btn.click(
                        fn=train_and_save_mlp,
                        inputs=[epochs_slider, lr_slider, sample_size_slider, hidden_layers_input, feature_mode],
                        outputs=train_output
                    )

                with gr.Tab("RBFN"):
                    gr.Markdown("### Train and save an RBFN")
                    gr.Markdown("The RBFN trains in one shot: k-means elects K centers, then a single linear solve finds the weights.")

                    with gr.Row():
                        with gr.Column():
                            rbfn_feature_mode = gr.Radio(
                                choices=["Extract (20 features)", "Flatten (3072 pixels)"],
                                value="Extract (20 features)",
                                label="Input mode"
                            )
                            rbfn_k_slider = gr.Slider(minimum=2, maximum=400, value=50, step=1, label="Number of centers K")
                            rbfn_gamma_slider = gr.Slider(minimum=0.001, maximum=10.0, value=0.1, step=0.001, label="Gamma (bump width)")

                        with gr.Column():
                            rbfn_iters_slider = gr.Slider(minimum=5, maximum=100, value=20, step=5, label="K-means iterations")
                            rbfn_sample_slider = gr.Slider(minimum=100, maximum=5000, value=3000, step=100, label="Max samples per class")

                    rbfn_train_btn = gr.Button("Train RBFN", variant="primary")
                    rbfn_train_output = gr.Textbox(label="Training log", interactive=False, lines=8)

                    rbfn_train_btn.click(
                        fn=train_and_save_rbfn,
                        inputs=[rbfn_k_slider, rbfn_gamma_slider, rbfn_iters_slider, rbfn_sample_slider, rbfn_feature_mode],
                        outputs=rbfn_train_output
                    )
                with gr.Tab("Linear (Rosenblatt)"):
                    gr.Markdown("### Train and save a Rosenblatt classifier")
                    gr.Markdown("Three perceptrons (one-vs-all) trained on cat / lion / cheetah.")

                    with gr.Row():
                        with gr.Column():
                            lin_feature_mode = gr.Radio(
                                choices=["Extract (20 features)", "Flatten (3072 pixels)"],
                                value="Extract (20 features)",
                                label="Input mode"
                            )
                            lin_seed = gr.Number(value=42, label="Seed", precision=0)

                        with gr.Column():
                            lin_epochs = gr.Slider(minimum=10, maximum=10000, value=500, step=10, label="Epochs")
                            lin_lr = gr.Slider(minimum=0.001, maximum=0.5, value=0.01, step=0.001, label="Learning rate")
                            lin_samples = gr.Slider(minimum=100, maximum=5000, value=3000, step=100, label="Max samples per class")

                    lin_train_btn = gr.Button("Train Rosenblatt", variant="primary")
                    lin_train_output = gr.Textbox(label="Training log", interactive=False, lines=8)

                    lin_train_btn.click(
                        fn=train_and_save_rosenblatt,
                        inputs=[lin_epochs, lin_lr, lin_samples, lin_feature_mode, lin_seed],
                        outputs=lin_train_output
                    )

                with gr.Tab("SVM"):
                    gr.Markdown("### Train and save an SVM")
                    gr.Markdown("One-vs-rest support vector machine. The dense QP solve is costly, so keep the sample count modest.")

                    with gr.Row():
                        with gr.Column():
                            svm_feature_mode = gr.Radio(
                                choices=["Extract (20 features)", "Flatten (3072 pixels)"],
                                value="Extract (20 features)",
                                label="Input mode"
                            )
                            svm_kernel = gr.Radio(
                                choices=["RBF", "Linear"],
                                value="RBF",
                                label="Kernel"
                            )
                            svm_gamma_slider = gr.Slider(minimum=0.001, maximum=10.0, value=0.01, step=0.001, label="Gamma (RBF only)")

                        with gr.Column():
                            svm_c_slider = gr.Slider(minimum=0.1, maximum=100.0, value=10.0, step=0.1, label="C (soft-margin penalty)")
                            svm_sample_slider = gr.Slider(minimum=50, maximum=1000, value=200, step=50, label="Max samples per class")

                    svm_train_btn = gr.Button("Train SVM", variant="primary")
                    svm_train_output = gr.Textbox(label="Training log", interactive=False, lines=8)

                    svm_train_btn.click(
                        fn=train_and_save_svm,
                        inputs=[svm_kernel, svm_gamma_slider, svm_c_slider, svm_sample_slider, svm_feature_mode],
                        outputs=svm_train_output
                    )

        with gr.Tab("📊 Status"):
            gr.Markdown("### Model status")
            status_output = gr.Textbox(label="Models", interactive=False, lines=5)
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(fn=check_model_status, inputs=[], outputs=status_output)
            demo.load(fn=check_model_status, outputs=status_output)

demo.launch(theme=gr.themes.Soft())
