// src/training/loss.rs

use crate::tensor::Matrix;

pub fn mse_loss(output: &Matrix, expected: &Matrix) -> f32 {
    let mut total = 0.0_f32;
    for i in 0..output.rows {
        for j in 0..output.cols {
            let diff = output.get(i, j) - expected.get(i, j);
            total += diff * diff;
        }
    }
    total / (output.rows * output.cols) as f32
}
