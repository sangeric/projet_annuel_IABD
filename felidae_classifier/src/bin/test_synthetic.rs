// src/bin/test_synthetic.rs

use std::cmp::max;
use std::ffi::c_void;
use felidae_classifier::tensor::Matrix;
use felidae_classifier::models::linear::LinearClassifier;
use felidae_classifier::models::perceptron::Rosenblatt;
use felidae_classifier::models::linear::Regression;


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
    // Convert both datasets to Matrix format
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

    // ----------------------------------------------------------------
    // LINEAR MODEL ON DATASET 1 — expect high accuracy
    // 2 input features, 3 classes, learning rate 0.1
    // ----------------------------------------------------------------
    println!("\n=== Linear Classifier on Dataset 1 (linearly separable) ===\n");

    let mut clf1 = LinearClassifier::new(2, 3, 0.1);
    clf1.train(&x_linear, &y_linear, 1000);

    let preds1 = clf1.predict(&x_linear);
    println!("\nResults:");
    print_predictions(&preds1, &y_linear);
    println!("Final accuracy: {:.1}%", clf1.accuracy(&x_linear, &y_linear) * 100.0);

    // ----------------------------------------------------------------
    // LINEAR MODEL ON DATASET 2 — expect poor accuracy (KO cases)
    // Same setup, but data is not linearly separable
    // ----------------------------------------------------------------
    println!("\n=== Linear Classifier on Dataset 2 (non-linearly separable) ===\n");

    let mut clf2 = LinearClassifier::new(2, 3, 0.1);
    clf2.train(&x_nonlinear, &y_nonlinear, 1000);

    let preds2 = clf2.predict(&x_nonlinear);
    println!("\nResults:");
    print_predictions(&preds2, &y_nonlinear);
    println!("Final accuracy: {:.1}%", clf2.accuracy(&x_nonlinear, &y_nonlinear) * 100.0);

    // ----------------------------------------------------------------
    // IDENTIFY KO CASES — which samples did the linear model get wrong?
    // These are the inputs transform.rs will need to fix
    // ----------------------------------------------------------------
    println!("\n=== KO Cases (misclassified by linear model on dataset 2) ===\n");
    print_ko_cases(&preds2, &y_nonlinear, &nonlinear_data);

    // ----------------------------------------------------------------
    // LINEAR MODEL + TRANSFORM ON DATASET 2 — fixing the KO cases
    // We apply polynomial_transform before training and predicting
    // ----------------------------------------------------------------
    println!("\n=== Linear Classifier + Transform on Dataset 2 ===\n");

    use felidae_classifier::models::linear::transform::{
        polynomial_transform,
        transformed_feature_count,
    };

    // Transform the input — goes from (17 × 2) to (17 × 5)
    let x_transformed = polynomial_transform(&x_nonlinear);
    println!("Transformed shape: {}x{}", x_transformed.rows, x_transformed.cols);

    // Note: n_features is now 5 instead of 2
    let mut clf3 = LinearClassifier::new(transformed_feature_count(), 3, 0.1);
    clf3.train(&x_transformed, &y_nonlinear, 1000);

    let preds3 = clf3.predict(&x_transformed);
    println!("\nResults:");
    print_predictions(&preds3, &y_nonlinear);
    println!("Final accuracy: {:.1}%", clf3.accuracy(&x_transformed, &y_nonlinear) * 100.0);


    //-------------------------------------------------------------------- Rosenblatt -------------------------------------------
    println!("\n\nStart of rosenblatt");

    println!("\n==============================");
    println!("  DATASET 1 — ROSENBLATT");
    println!("  Linearly Separable");
    println!("==============================\n");

    let linear_data_rosenblatt: Vec<([f32; 2], usize)> = vec![
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

    print_dataset("Lineary separable ",&linear_data_rosenblatt);

    let nonlinear_data_rosenblatt: Vec<([f32; 2], usize)> = vec![
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
    print_dataset("Non lineary separable", &nonlinear_data_rosenblatt);
    let (x_linear_rosen, y_linear_rosen) = to_matrix(&linear_data_rosenblatt);
    let (x_nonlinear_rosen, y_nonlinear_rosen) = to_matrix(&nonlinear_data_rosenblatt);
    println!(
        "Linear dataset : X shape = {}x{}, y len = {}",x_linear_rosen.rows, x_linear_rosen.cols, y_linear_rosen.len()
    );
    println!(
        "Non linear dataset : X shape = {}x{}, y len = {}", x_nonlinear_rosen.rows, x_nonlinear_rosen.cols, y_nonlinear_rosen.len()
    );
    let nb_features = x_linear_rosen.cols;



    println!("\n=== Linear Classifier on Dataset 1 (linearly separable) ===\n");
    let mut cat_perceptron_linear = Rosenblatt::new(nb_features,0.01, 1.0, 0, 500);
    let mut lion_perceptron_linear = Rosenblatt::new(nb_features,0.01, 1.0, 1, 1000);
    let mut cheetah_perceptron_linear = Rosenblatt::new(nb_features,0.01, 1.0, 2, 2000);
    println!("Cat :");
    cat_perceptron_linear.train(&x_linear_rosen, &y_linear_rosen, 500);
    println!();
    println!("Lion :");
    lion_perceptron_linear.train(&x_linear_rosen, &y_linear_rosen, 500);
    println!();
    println!("Cheetah :");
    cheetah_perceptron_linear.train(&x_linear_rosen, &y_linear_rosen, 500);
    println!();
    let cat_prediction_linear = cat_perceptron_linear.predict(&x_linear_rosen);
    let lion_prediction_linear = lion_perceptron_linear.predict(&x_linear_rosen);
    let cheetah_prediction_linear = cheetah_perceptron_linear.predict(&x_linear_rosen);

    println!("\n\nResult\n");
    let tab_argmax:Vec<usize> = argmax(cat_prediction_linear, lion_prediction_linear, cheetah_prediction_linear);
    Rosenblatt::print_prediction_result(tab_argmax, &y_linear_rosen);



    println!("\n=== Linear Classifier on Dataset 2 (non-linearly separable) ===\n");
    let mut cat_perceptron_non_linear = Rosenblatt::new(nb_features,0.01, 1.0, 0, 500);
    let mut lion_perceptron_non_linear = Rosenblatt::new(nb_features,0.01, 1.0, 1, 1000);
    let mut cheetah_perceptron_non_linear = Rosenblatt::new(nb_features,0.01, 1.0, 2, 2000);

    println!("Cat :");
    cat_perceptron_non_linear.train(&x_nonlinear_rosen, &y_nonlinear_rosen, 500);
    println!();
    println!("Lion :");
    lion_perceptron_non_linear.train(&x_nonlinear_rosen, &y_nonlinear_rosen, 500);
    println!();
    println!("Cheetah :");
    cheetah_perceptron_non_linear.train(&x_nonlinear_rosen, &y_nonlinear_rosen, 500);
    println!();
    println!("cat weights {:?}", cat_perceptron_non_linear.get_weight());
    println!("lion weights {:?}", lion_perceptron_non_linear.get_weight());
    println!("cheetah weights {:?}\n\n", cheetah_perceptron_non_linear.get_weight());

    let cat_perceptron_non_linear = cat_perceptron_non_linear.predict(&x_nonlinear_rosen);
    let lion_perceptron_non_linear = lion_perceptron_non_linear.predict(&x_nonlinear_rosen);
    let cheetah_perceptron_non_linear = cheetah_perceptron_non_linear.predict(&x_nonlinear_rosen);

    println!("{:?}\n {:?}\n {:?}", cat_perceptron_non_linear, lion_perceptron_non_linear, cheetah_perceptron_non_linear);
    println!("\n\nResult\n");
    let tab_argmax:Vec<usize> = argmax(cat_perceptron_non_linear, lion_perceptron_non_linear, cheetah_perceptron_non_linear);
    Rosenblatt::print_prediction_result(tab_argmax, &y_nonlinear_rosen);




    println!("\n=== Linear Classifier + Transform on Dataset 2 ===\n");
    let nb_deg2 = 5;
    let mut cat_perceptron_non_linear_transform = Rosenblatt::new(nb_deg2,0.01, 1.0, 0, 500);
    let mut lion_perceptron_non_linear_transform = Rosenblatt::new(nb_deg2,0.01, 1.0, 1, 1000);
    let mut cheetah_perceptron_non_linear_transform = Rosenblatt::new(nb_deg2,0.01, 1.0, 2, 2000);
    println!("{:?}", cat_perceptron_non_linear_transform.get_weight());
    //transformer x_nonlinear_rosen pour la transformation
    let mut x_cat_transform = cat_perceptron_non_linear_transform.transform(&x_nonlinear_rosen, y_nonlinear_rosen.len());
    let mut x_lion_transform = lion_perceptron_non_linear_transform.transform(&x_nonlinear_rosen,  y_nonlinear_rosen.len());
    let mut x_cheetah_transform = cheetah_perceptron_non_linear_transform.transform(&x_nonlinear_rosen,  y_nonlinear_rosen.len());

    cat_perceptron_non_linear_transform.train(&x_cat_transform, &y_nonlinear_rosen, 500);
    lion_perceptron_non_linear_transform.train(&x_lion_transform, &y_nonlinear_rosen, 500);
    cheetah_perceptron_non_linear_transform.train(&x_cheetah_transform, &y_nonlinear_rosen, 500);


    let cat_perceptron_transform = cat_perceptron_non_linear_transform.predict(&x_cat_transform);
    let lion_perceptron_transform = lion_perceptron_non_linear_transform.predict(&x_lion_transform);
    let cheetah_perceptron_transform = cheetah_perceptron_non_linear_transform.predict(&x_cheetah_transform);

    println!("\n\nResult\n");
    let tab_argmax_transform :Vec<usize> = argmax(cat_perceptron_transform, lion_perceptron_transform, cheetah_perceptron_transform);
    println!("tab arg max {:?}", tab_argmax_transform);
    Rosenblatt::print_prediction_result(tab_argmax_transform, &y_nonlinear_rosen);



    /*
    //Regression linéaire

    let linear_data_regression: Vec<([f32; 2], f32)> = vec![
        ([0.3, 0.2], 0.8),
        ([0.5, 0.4], 1.2),
        ([0.7, 0.6], 1.8),
        ([0.4, 0.3], 1.0),
        ([0.6, 0.5], 1.5),
        ([0.8, 0.7], 2.1),
        ([0.2, 0.1], 0.5),
        ([0.9, 0.8], 2.4),
        ([0.1, 0.1], 0.3),
        ([0.55, 0.45], 1.3),
        ([0.35, 0.25], 0.9),
        ([0.65, 0.55], 1.6),
        ([0.75, 0.65], 2.0),
        ([0.45, 0.35], 1.1),
        ([0.85, 0.75], 2.2),
        ([0.15, 0.05], 0.4),
        ([0.25, 0.15], 0.6),
        ([0.95, 0.85], 2.5),
        ([0.58, 0.48], 1.4),
        ([0.42, 0.32], 1.05),
    ];

    print_dataset_regression("Lineary separable regression ", &linear_data_regression);

    let non_linear_data_regression: Vec<([f32; 2], f32)> = vec![
        ([0.1, 0.0], 0.01),
        ([0.2, 0.0], 0.04),
        ([0.3, 0.0], 0.09),
        ([0.5, 0.0], 0.25),
        ([0.7, 0.0], 0.49),
        ([0.9, 0.0], 0.81),
        ([0.15, 0.0], 0.0225),
        ([0.25, 0.0], 0.0625),
        ([0.35, 0.0], 0.1225),
        ([0.45, 0.0], 0.2025),
        ([0.55, 0.0], 0.3025),
        ([0.65, 0.0], 0.4225),
        ([0.75, 0.0], 0.5625),
        ([0.85, 0.0], 0.7225),
        ([0.95, 0.0], 0.9025),
        ([0.12, 0.0], 0.0144),
        ([0.38, 0.0], 0.1444),
        ([0.62, 0.0], 0.3844),
        ([0.78, 0.0], 0.6084),
        ([0.92, 0.0], 0.8464),
    ];

    print_dataset_regression("Non lineary separable regression", &non_linear_data_regression);

    let (x_linear_regression, y_linear_regression) = to_matrix_regression(&linear_data_regression);
    let (x_nonlinear_regression, y_nonlinear_regression) = to_matrix_regression(&non_linear_data_regression);
    println!(
        "Linear dataset : X shape = {}x{}, y len = {}",x_linear_regression.rows, x_linear_regression.cols, y_linear_regression.cols
    );
    println!(
        "Non linear dataset : X shape = {}x{}, y len = {}", x_nonlinear_regression.rows, x_nonlinear_regression.cols, y_nonlinear_regression.cols
    );
    let nb_features_regression = 2;
    let bias :f32= 1.0;
    let mut regression_linear = Regression::new(nb_features_regression);
    regression_linear.fit(&x_linear_regression, &y_linear_regression, bias);
    println!("poids : {:?}", regression_linear.get_weight());
    let mut prediction_regression = regression_linear.predict(&x_linear_regression, bias);
    println!("\nprediction : {:?}", prediction_regression);

    let mse = mse(&y_linear_regression, &prediction_regression);
    let mae = mae(&y_linear_regression, &prediction_regression);
    println!("MSE: {:.6}", mse);
    println!("MAE: {:.6}", mae);
    */
}

/// Pretty-prints a dataset
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

fn print_dataset_regression(name : &str, data : &[([f32;2], f32)]){
    println!("{}", name);
    for (point, label) in data{
        println!(
            "  [{:.2}, {:.2}] → {}",
            point[0], point[1], label
        );
    }
}

/// Prints predicted vs actual label for every sample
fn print_predictions(predictions: &[usize], labels: &[usize]) {
    let class_names = ["cat", "lion", "cheetah"];
    for i in 0..predictions.len() {
        let status = if predictions[i] == labels[i] { "OK" } else { "KO" };
        println!(
            "  Sample {:>2} | predicted: {:>7} | actual: {:>7} | {}",
            i,
            class_names[predictions[i]],
            class_names[labels[i]],
            status
        );
    }
}

/// Prints only the samples the linear model got wrong
/// These are the KO cases that transform.rs will address
fn print_ko_cases(predictions: &[usize], labels: &[usize], data: &[([f32; 2], usize)]) {
    let class_names = ["cat", "lion", "cheetah"];
    let mut ko_count = 0;

    for i in 0..predictions.len() {
        if predictions[i] != labels[i] {
            ko_count += 1;
            println!(
                "  KO sample {:>2} | point [{:.2}, {:.2}] | predicted: {:>7} | actual: {:>7}",
                i,
                data[i].0[0],
                data[i].0[1],
                class_names[predictions[i]],
                class_names[labels[i]]
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

    // Flatten all points into a single Vec<f32> row by row
    let mut flat = Vec::new();
    for (point, _) in data {
        flat.push(point[0]);
        flat.push(point[1]);
    }

    // Collect labels into a separate Vec
    let mut labels = Vec::new();
    for (_, label) in data {
        labels.push(*label);
    }

    (Matrix::from_vec(flat, rows, cols), labels)
}


fn to_matrix_regression(data: &[([f32; 2], f32)]) -> (Matrix, Matrix) {
    let rows = data.len();
    let cols = 2;

    // Flatten all points into a single Vec<f32> row by row
    let mut flat = Vec::new();
    for (point, _) in data {
        flat.push(point[0]);
        flat.push(point[1]);
    }

    // Collect labels into a separate Vec
    let mut labels = Vec::new();
    for (_, label) in data {
        labels.push(*label);
    }

    (Matrix::from_vec(flat, rows, cols), Matrix::from_vec(labels,rows, 1))
}


fn argmax(cat : Vec<f32>, lion : Vec<f32>, cheetah : Vec<f32>) -> Vec<usize>{
    let mut argmax = Vec::new();

    for i in 0..cat.len() {
        let scores = [cat[i], lion[i],cheetah[i]];
        let mut max = max_argmax(scores);
        argmax.push(max);
    }

    argmax
}

fn max_argmax(scores: [f32; 3]) -> usize{
    let mut index_max : usize = 0;
    let mut max : f32 = scores[0];
    for i in 1..scores.len(){
        if max < scores[i]{
            max = scores[i];
            index_max = i;
        }
    }
    index_max
}


pub fn mse(y_true: &Matrix, y_pred: &Matrix) -> f32 {
    let n = y_true.rows as f32;
    let mut sum = 0.0;
    for i in 0..y_true.rows {
        let diff = y_true.get(i, 0) - y_pred.get(i, 0);
        sum += diff * diff;
    }
    sum / n
}

pub fn mae(y_true: &Matrix, y_pred: &Matrix) -> f32 {
    let n = y_true.rows as f32;
    let mut sum = 0.0;
    for i in 0..y_true.rows {
        sum += (y_true.get(i, 0) - y_pred.get(i, 0)).abs();
    }
    sum / n
}