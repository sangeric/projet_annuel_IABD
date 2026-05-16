// src/models/mlp/activation.rs

use cargo::tensor::Matrix;

pub fn relu(x: f32) -> f32 {
    if x <= 0.0 { 0.0 } else { x }
}

pub fn relu_derivative(x: f32) -> f32 {
    if x <= 0.0 { 0.0 } else { 1.0 }
}



/// Softmax — converts raw scores into probabilities
/// Each row sums to 1.0, like a probability distribution over classes
pub fn softmax(matrix: &Matrix) -> Matrix {
    let mut result = Matrix::zeros(matrix.rows, matrix.cols);

    for i in 0..matrix.rows {
        // Find the max score in this row for numerical stability
        // This prevents exp() from overflowing — standard trick
        let mut max = f32::NEG_INFINITY;
        for j in 0..matrix.cols {
            if matrix.get(i, j) > max {
                max = matrix.get(i, j);
            }
        }

        // Compute exp(score - max) for each class
        let mut exps = Vec::new();
        for j in 0..matrix.cols {
            exps.push((matrix.get(i, j) - max).exp());
        }

        // Sum all the exp values so we can normalize
        let mut sum = 0.0_f32;
        for val in &exps {
            sum += val;
        }

        // Divide each exp by the sum — now the row sums to 1.0
        for j in 0..matri.cols {
            result.set(i, j, exps[j] / sum);
        }
    }

    result
}
