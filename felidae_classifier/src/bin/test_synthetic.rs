// src/bin/test_synthetic.rs

use felidae_classifier::tensor::Matrix;
use felidae_classifier::models::linear::LinearClassifier;
use felidae_classifier::models::linear::transform::{polynomial_transform, transformed_feature_count};
use felidae_classifier::models::perceptron::RosenblattClassifier;
use felidae_classifier::models::mlp::MLP;
use felidae_classifier::models::rbfn::RBFN;
use felidae_classifier::models::svm::kernel::Kernel;
use felidae_classifier::models::svm::multiclass::MulticlassSVM;

use felidae_classifier::data::one_hot_encode;

fn main() {
    println!("=== Synthetic Test Cases ===\n");

    // ----------------------------------------------------------------
    // DATASET 1 — Linearly separable
    // 3 clusters of points, one per class, clearly separated
    // Class 0 = cat     -> bottom-left cluster
    // Class 1 = lion    -> top cluster
    // Class 2 = cheetah -> bottom-right cluster
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
    // Class 0 (cat) and class 1 (lion) are interleaved so that no
    // straight line separates the three classes. These are the KO
    // cases for the raw linear model.
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
    // Convert both datasets to Matrix format (shared by all models)
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

    let nb_features = x_linear.cols;

    // ================================================================
    // LINEAR CLASSIFIER
    // ================================================================
    println!("\n\n############## LINEAR CLASSIFIER ##############");

    println!("\n=== Linear Classifier on Dataset 1 (linearly separable) ===\n");
    let mut clf1 = LinearClassifier::new(2, 3, 0.1);
    clf1.train(&x_linear, &y_linear, 1000);
    let preds1 = clf1.predict(&x_linear);
    println!("\nResults:");
    print_predictions(&preds1, &y_linear);
    println!("Final accuracy: {:.1}%", clf1.accuracy(&x_linear, &y_linear) * 100.0);

    println!("\n=== Linear Classifier on Dataset 2 (non-linearly separable) ===\n");
    let mut clf2 = LinearClassifier::new(2, 3, 0.1);
    clf2.train(&x_nonlinear, &y_nonlinear, 1000);
    let preds2 = clf2.predict(&x_nonlinear);
    println!("\nResults:");
    print_predictions(&preds2, &y_nonlinear);
    println!("Final accuracy: {:.1}%", clf2.accuracy(&x_nonlinear, &y_nonlinear) * 100.0);

    println!("\n=== KO Cases (misclassified by linear model on dataset 2) ===\n");
    print_ko_cases(&preds2, &y_nonlinear, &nonlinear_data);

    println!("\n=== Linear Classifier + Transform on Dataset 2 ===\n");
    let x_transformed = polynomial_transform(&x_nonlinear);
    println!("Transformed shape: {}x{}", x_transformed.rows, x_transformed.cols);
    let mut clf3 = LinearClassifier::new(transformed_feature_count(), 3, 0.1);
    clf3.train(&x_transformed, &y_nonlinear, 1000);
    let preds3 = clf3.predict(&x_transformed);
    println!("\nResults:");
    print_predictions(&preds3, &y_nonlinear);
    println!("Final accuracy: {:.1}%", clf3.accuracy(&x_transformed, &y_nonlinear) * 100.0);

    // ================================================================
    // ROSENBLATT (one-vs-rest)
    // ================================================================
    println!("\n\n############## ROSENBLATT ##############");

    println!("\n=== Rosenblatt on Dataset 1 (linearly separable) ===\n");
    let mut linear_rosen = RosenblattClassifier::new(nb_features, 0.01, 1.0, 1.0, 1.0, 500);
    linear_rosen.train(&x_linear, &y_linear, 100);
    linear_rosen.predict(&x_linear, &y_linear);

    println!("\n=== Rosenblatt on Dataset 2 (non-linearly separable) ===\n");
    let mut non_linear_rosen = RosenblattClassifier::new(nb_features, 0.01, 1.0, 1.0, 1.0, 500);
    non_linear_rosen.train(&x_nonlinear, &y_nonlinear, 100);
    non_linear_rosen.predict(&x_nonlinear, &y_nonlinear);

    println!("\n=== Rosenblatt + Transform on Dataset 2 ===\n");
    let nb_deg2 = x_nonlinear.cols * 2 + 1;
    let mut non_linear_transform_rosen = RosenblattClassifier::new(nb_deg2, 0.01, 1.0, 1.0, 1.0, 500);
    let x_non_linear_transform = non_linear_transform_rosen.transform(&x_nonlinear);
    println!("x transformed {:?}", x_non_linear_transform);
    non_linear_transform_rosen.train(&x_non_linear_transform, &y_nonlinear, 100);
    non_linear_transform_rosen.predict(&x_non_linear_transform, &y_nonlinear);

    // ================================================================
    // MLP
    // Network: 2 inputs -> 8 hidden -> 8 hidden -> 3 classes
    // Tanh activation throughout (as per the course slides)
    // ================================================================
    println!("\n\n############## MLP ##############");

    println!("\n=== MLP on Dataset 1 (linearly separable) ===\n");
    let y_linear_onehot = one_hot_encode(&y_linear, 3);
    let mut mlp1 = MLP::new(&[2, 8, 8, 3], "tanh", "tanh").expect("failed to build MLP");
    mlp1.train(&x_linear, &y_linear_onehot, &x_linear, &y_linear_onehot, 5000, 0.05, "/tmp/test_synthetic");
    let mlp_preds1 = mlp1.predict(&x_linear);
    println!("\nResults:");
    print_predictions(&mlp_preds1, &y_linear);
    println!("Final accuracy: {:.1}%", mlp1.accuracy(&x_linear, &y_linear_onehot) * 100.0);

    println!("\n=== MLP on Dataset 2 (non-linearly separable) ===\n");
    let y_nonlinear_onehot = one_hot_encode(&y_nonlinear, 3);
    let mut mlp2 = MLP::new(&[2, 8, 8, 3], "tanh", "tanh").expect("failed to build MLP");
    mlp2.train(&x_nonlinear, &y_nonlinear_onehot, &x_nonlinear, &y_nonlinear_onehot, 5000, 0.05, "/tmp/test_synthetic");
    let mlp_preds2 = mlp2.predict(&x_nonlinear);
    println!("\nResults:");
    print_predictions(&mlp_preds2, &y_nonlinear);
    println!("Final accuracy: {:.1}%", mlp2.accuracy(&x_nonlinear, &y_nonlinear_onehot) * 100.0);

    // ================================================================
    // RBFN
    // K Gaussian centers elected by k-means, then a single linear solve.
    // ================================================================
    println!("\n\n############## RBFN ##############");

    println!("\n=== RBFN on Dataset 1 (linearly separable) ===\n");
    let rbfn1 = RBFN::train(&x_linear, &y_linear_onehot, 6, 2.0, 20, 42);
    let rbfn_preds1 = rbfn1.predict(&x_linear);
    println!("\nResults:");
    print_predictions(&rbfn_preds1, &y_linear);
    println!("Final accuracy: {:.1}%", rbfn1.accuracy(&x_linear, &y_linear_onehot) * 100.0);

    println!("\n=== RBFN on Dataset 2 (non-linearly separable) ===\n");
    let rbfn2 = RBFN::train(&x_nonlinear, &y_nonlinear_onehot, 8, 2.0, 20, 42);
    let rbfn_preds2 = rbfn2.predict(&x_nonlinear);
    println!("\nResults:");
    print_predictions(&rbfn_preds2, &y_nonlinear);
    println!("Final accuracy: {:.1}%", rbfn2.accuracy(&x_nonlinear, &y_nonlinear_onehot) * 100.0);

    // ================================================================
    // SVM (one-vs-rest, linear and RBF kernels)
    // ================================================================
    println!("\n\n############## SVM ##############");

    println!("\n=== SVM (RBF kernel) on Dataset 1 (linearly separable) ===\n");
    let svm_rbf1 = MulticlassSVM::train(&x_linear, &y_linear, 3, Kernel::Rbf { gamma: 2.0 });
    let svm_preds1 = svm_rbf1.predict(&x_linear);
    println!("\nResults:");
    print_predictions(&svm_preds1, &y_linear);
    println!("Final accuracy: {:.1}%", svm_rbf1.accuracy(&x_linear, &y_linear) * 100.0);

    println!("\n=== SVM (RBF kernel) on Dataset 2 (non-linearly separable) ===\n");
    let svm_rbf2 = MulticlassSVM::train(&x_nonlinear, &y_nonlinear, 3, Kernel::Rbf { gamma: 2.0 });
    let svm_rbf_preds2 = svm_rbf2.predict(&x_nonlinear);
    println!("\nResults:");
    print_predictions(&svm_rbf_preds2, &y_nonlinear);
    println!("Final accuracy: {:.1}%", svm_rbf2.accuracy(&x_nonlinear, &y_nonlinear) * 100.0);}

/// Pretty-prints a dataset
fn print_dataset(name: &str, data: &[([f32; 2], usize)]) {
    let class_names = ["cat", "lion", "cheetah"];
    println!("{}:", name);
    for (point, label) in data {
        println!("  [{:.2}, {:.2}] -> {}", point[0], point[1], class_names[*label]);
    }
}

/// Prints predicted vs actual label for every sample
fn print_predictions(predictions: &[usize], labels: &[usize]) {
    let class_names = ["cat", "lion", "cheetah"];
    for i in 0..predictions.len() {
        let status = if predictions[i] == labels[i] { "OK" } else { "KO" };
        println!(
            "  Sample {:>2} | predicted: {:>7} | actual: {:>7} | {}",
            i, class_names[predictions[i]], class_names[labels[i]], status
        );
    }
}

/// Prints only the samples the linear model got wrong (the KO cases)
fn print_ko_cases(predictions: &[usize], labels: &[usize], data: &[([f32; 2], usize)]) {
    let class_names = ["cat", "lion", "cheetah"];
    let mut ko_count = 0;

    for i in 0..predictions.len() {
        if predictions[i] != labels[i] {
            ko_count += 1;
            println!(
                "  KO sample {:>2} | point [{:.2}, {:.2}] | predicted: {:>7} | actual: {:>7}",
                i, data[i].0[0], data[i].0[1],
                class_names[predictions[i]], class_names[labels[i]]
            );
        }
    }

    if ko_count == 0 {
        println!("  No KO cases — linear model solved this dataset perfectly.");
    } else {
        println!("\n  Total KO cases: {}/{}", ko_count, predictions.len());
    }
}

/// Converts raw Vec of (point, label) into a Matrix (N x 2) and a Vec<usize> of labels
fn to_matrix(data: &[([f32; 2], usize)]) -> (Matrix, Vec<usize>) {
    let rows = data.len();
    let cols = 2;

    let mut flat = Vec::new();
    for (point, _) in data {
        flat.push(point[0]);
        flat.push(point[1]);
    }

    let mut labels = Vec::new();
    for (_, label) in data {
        labels.push(*label);
    }

    (Matrix::from_vec(flat, rows, cols), labels)
}
