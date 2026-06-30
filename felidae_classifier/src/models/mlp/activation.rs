// src/models/mlp/activation.rs

use crate::tensor::Matrix;

pub fn identity(x: f32) -> f32 {
    x
}

pub fn identity_derivative(_: f32) -> f32 {
    1.0
}

pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

pub fn tanh_derivative(x: f32) -> f32 {
    1.0 - (x.tanh() * x.tanh())
}

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
        for j in 0..matrix.cols {
            result.set(i, j, exps[j] / sum);
        }
    }

    result
}

pub fn lookup(name: &str) -> Result<(fn(f32) -> f32, fn(f32) -> f32), String> {
    match name {
        "tanh" => Ok((tanh, tanh_derivative)),
        "relu" => Ok((relu, relu_derivative)),
        "identity" => Ok((identity, identity_derivative)),
        _ => Err(format!("Unknown activation function: '{}'", name)),
    }
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_known_activations() {
        assert!(lookup("tanh").is_ok());
        assert!(lookup("relu").is_ok());
        assert!(lookup("identity").is_ok());
    }

    #[test]
    fn test_lookup_unknown_activation_errors() {
        assert!(lookup("not_a_real_function").is_err());
    }

    #[test]
    fn test_identity_is_pass_through() {
        assert_eq!(identity(0.5), 0.5);
        assert_eq!(identity(-3.0), -3.0);
        assert_eq!(identity_derivative(0.5), 1.0);
    }
}
