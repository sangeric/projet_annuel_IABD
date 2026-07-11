import gradio as gr
import numpy as np
from PIL import Image
from cffi import FFI
import os
import json

# --- FFI setup ---
ffi = FFI()
ffi.cdef("""
    void* mlp_create(const uint32_t* layer_sizes, size_t n_layers, const char* activation, const char* output_activation);
    void mlp_train(void* mlp, const float* x_data, size_t x_rows, size_t x_cols, const float* y_data, size_t y_rows, size_t y_cols, size_t epochs, float learning_rate, const char* log_dir);
    void mlp_predict_classes(const void* mlp, const float* x_data, size_t x_rows, size_t x_cols, uint32_t* out_predictions);
    void mlp_predict_raw(const void* mlp, const float* x_data, size_t x_rows, size_t x_cols, float* out_predictions);
    int mlp_save(const void* mlp, const char* path);
    void* mlp_load(const char* path);
    void mlp_destroy(void* mlp);
    void extract_features(const float* image_data, float* out, size_t out_len);
""", override=True)

BASE = os.path.abspath("felidae_classifier")
lib = ffi.dlopen(os.path.join(BASE, "target/release/libfelidae_classifier.so"))
MODEL_DIR = os.path.join(BASE, "saved_models")
for subfolder in ["mlp", "linear", "svm", "rbfn"]:
    os.makedirs(os.path.join(MODEL_DIR, subfolder), exist_ok=True)

CLASS_NAMES = ["Cat", "Lion", "Cheetah"]
N_FEATURES = 20
N_FLATTEN = 3072
IMG_SIZE = (32, 32)
DATASET_ROOT = os.path.join(BASE, "dataset")


# --- Metadata helpers ---
def save_model_meta(feature_mode):
    meta = {"feature_mode": feature_mode}
    with open(os.path.join(MODEL_DIR, "mlp", "mlp_meta.json"), "w") as f:
        json.dump(meta, f)

def load_model_meta():
    meta_path = os.path.join(MODEL_DIR, "mlp", "mlp_meta.json")
    if not os.path.exists(meta_path):
        return "Extract (20 features)"
    with open(meta_path) as f:
        return json.load(f)["feature_mode"]


# --- Preprocessing ---
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


# --- Dataset loading ---
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


# --- MLP train and save ---
def train_and_save_mlp(epochs, learning_rate, max_samples_per_class, hidden_layers_str, feature_mode, progress=gr.Progress()):
    use_extract = "Extract" in feature_mode
    n_inputs = N_FEATURES if use_extract else N_FLATTEN

    progress(0, desc="Loading dataset...")
    X, Y = load_dataset(max_samples=int(max_samples_per_class), use_extract=use_extract)
    if len(X) == 0:
        return "No dataset found at: " + DATASET_ROOT

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

    X_ptr = ffi.from_buffer("float[]", X)
    Y_ptr = ffi.from_buffer("float[]", Y)

    progress(0.2, desc=f"Training {int(epochs)} epochs at lr={learning_rate}...")
    lib.mlp_train(mlp, X_ptr, X.shape[0], X.shape[1], Y_ptr, Y.shape[0], Y.shape[1], int(epochs), float(learning_rate), b"./runs/logdir")

    save_path = os.path.join(MODEL_DIR, "mlp", "mlp.bin").encode()
    ret = lib.mlp_save(mlp, save_path)
    lib.mlp_destroy(mlp)

    if ret != 0:
        return "Training done but save failed"

    save_model_meta(feature_mode)
    progress(1.0, desc="Done")
    return (
        f"Training complete.\n"
        f"Input mode   : {feature_mode}\n"
        f"Architecture : {arch}\n"
        f"Epochs       : {int(epochs)}\n"
        f"Learning rate: {learning_rate}\n"
        f"Samples      : {len(X)} ({len(X) // 3} per class)\n"
        f"Saved to     : {save_path.decode()}"
    )


# --- MLP predict ---
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

    scores = [float(out_raw[i]) for i in range(3)]
    min_s, max_s = min(scores), max(scores)
    if max_s != min_s:
        scores = [(s - min_s) / (max_s - min_s) for s in scores]

    return {CLASS_NAMES[i]: round(scores[i], 3) for i in range(3)}, None


# --- Main predict dispatcher ---
def predict(pil_image, model_choice):
    if pil_image is None:
        return None, "Please upload an image"

    feature_mode = load_model_meta()
    use_extract = "Extract" in feature_mode
    X = preprocess(pil_image, use_extract=use_extract)

    if model_choice == "MLP":
        scores, err = predict_mlp(X)
        if err:
            return None, err
        return scores, None
    else:
        return None, f"{model_choice} not yet implemented"


def classify(pil_image, model_choice):
    scores, err = predict(pil_image, model_choice)
    if err:
        return None, f"⚠️ {err}"
    return scores, ""


# --- Model status ---
def check_model_status():
    lines = []
    mlp_path = os.path.join(MODEL_DIR, "mlp", "mlp.bin")
    meta_path = os.path.join(MODEL_DIR, "mlp", "mlp_meta.json")

    if os.path.exists(mlp_path):
        mode = load_model_meta() if os.path.exists(meta_path) else "unknown"
        lines.append(f"✅ MLP — ready ({mode})")
    else:
        lines.append("❌ MLP — not trained")

    lines.append("❌ Linear (Rosenblatt) — not yet implemented")
    lines.append("❌ SVM — not yet implemented")
    lines.append("❌ RBFN — not yet implemented")
    return "\n".join(lines)


# --- UI ---
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
            gr.Markdown("### Train and save a model")
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
                    lr_slider = gr.Slider(minimum=0.01, maximum=1, value=0.01, step=0.01, label="Learning rate")
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

        with gr.Tab("📊 Status"):
            gr.Markdown("### Model status")
            status_output = gr.Textbox(label="Models", interactive=False, lines=5)
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(fn=check_model_status, inputs=[], outputs=status_output)
            demo.load(fn=check_model_status, outputs=status_output)

demo.launch(theme=gr.themes.Soft())
