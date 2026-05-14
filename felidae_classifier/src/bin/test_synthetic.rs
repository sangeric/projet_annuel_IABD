// src/bin/test_synthetic.rs

use felidae_classifier::tensor::Matrix;

fn main() {
    println!("=== Synthetic Test Cases ===\n");

    // ----------------------------------------------------------------
    // DATASET 1 — Linearly separable
    // 3 clusters of points, one per class, clearly separated
    // Class 0 = cat   → bottom-left cluster
    // Class 1 = lion  → top cluster
    // Class 2 = cheetah → bottom-right cluster
    // ----------------------------------------------------------------
    println!("--- Dataset 1: Linearly Separable ---");

    let linear_data: Vec<([f32; 2], usize)> = vec![
        // class 0 — cat (bottom-left)
        ([0.1, 0.1], 0), ([0.2, 0.15], 0), ([0.15, 0.2], 0),
        ([0.1, 0.25], 0), ([0.25, 0.1], 0),
        // class 1 — lion (top)
        ([0.5, 0.9], 1), ([0.55, 0.85], 1), ([0.45, 0.88], 1),
        ([0.5, 0.95], 1), ([0.6, 0.9], 1),
        // class 2 — cheetah (bottom-right)
        ([0.9, 0.1], 2), ([0.85, 0.15], 2), ([0.88, 0.2], 2),
        ([0.95, 0.1], 2), ([0.9, 0.25], 2),
    ];

    print_dataset("Linearly Separable", &linear_data);

    // ----------------------------------------------------------------
    // DATASET 2 — Non-linearly separable (XOR-like)
    // Class 0 (cat) and class 2 (cheetah) are interleaved —
    // no straight line can separate them from class 1 (lion).
    // These are your KO cases for the linear model.
    // ----------------------------------------------------------------
    println!("\n--- Dataset 2: Non-Linearly Separable (KO cases) ---");

    let nonlinear_data: Vec<([f32; 2], usize)> = vec![
        // class 0 — cat (top-left and bottom-right)
        ([0.1, 0.9], 0), ([0.2, 0.8], 0), ([0.15, 0.85], 0),
        ([0.9, 0.1], 0), ([0.8, 0.2], 0), ([0.85, 0.15], 0),
        // class 1 — lion (top-right and bottom-left)
        ([0.9, 0.9], 1), ([0.8, 0.8], 1), ([0.85, 0.85], 1),
        ([0.1, 0.1], 1), ([0.2, 0.2], 1), ([0.15, 0.15], 1),
        // class 2 — cheetah (center)
        ([0.5, 0.5], 2), ([0.55, 0.45], 2), ([0.45, 0.55], 2),
        ([0.5, 0.6], 2), ([0.6, 0.5], 2),
    ];

    print_dataset("Non-Linearly Separable", &nonlinear_data);

    // ----------------------------------------------------------------
    // Represent both datasets as Matrix so the rest of the project
    // can consume them — models expect a Matrix of shape (N x features)
    // and a flat Vec<usize> of labels
    // ----------------------------------------------------------------
    println!("\n--- Converting to Matrix format ---");

    let (x_linear, y_linear) = to_matrix(&linear_data);
    let (x_nonlinear, y_nonlinear) = to_matrix(&nonlinear_data);

    println!(
        "Linear dataset:     X shape = {}x{}, y len = {}",
        x_linear.rows, x_linear.cols, y_linear.len()
    );
    println!(
        "Non-linear dataset: X shape = {}x{}, y len = {}",
        x_nonlinear.rows, x_nonlinear.cols, y_nonlinear.len()
    );

    println!("\nAll synthetic test cases loaded successfully.");
}

/// Pretty-prints a dataset — useful for the report
fn print_dataset(name: &str, data: &[([f32; 2], usize)]) {
    let class_names = ["cat", "lion", "cheetah"];
    println!("{}:", name);
    for (point, label) in data {
        println!(
            "  [{:.2}, {:.2}] → {}",
            point[0], point[1], class_names[*label]
        );
    }
}

/// Converts our raw Vec of (point, label) into:
/// - a Matrix of shape (N x 2) for the inputs  — what models call X
/// - a Vec<usize> of labels                     — what models call y
/// 
/// This is the format every model in the project will expect
fn to_matrix(data: &[([f32; 2], usize)]) -> (Matrix, Vec<usize>) {
    let rows = data.len();
    let cols = 2; // x and y coordinates

    // Flatten all points into a single Vec<f32> row by row
    // like numpy.array([p for p, _ in data]).flatten()
    let flat: Vec<f32> = data
        .iter()
        .flat_map(|(point, _)| point.iter().copied())
        .collect();

    let labels: Vec<usize> = data
        .iter()
        .map(|(_, label)| *label)
        .collect();

    (Matrix::from_vec(flat, rows, cols), labels)
}
