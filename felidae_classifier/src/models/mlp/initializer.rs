// src/models/mlp/initializer.rs

use crate::tensor::Matrix;
use rand::RngExt;

pub fn he_init(n_inputs: usize, n_outputs: usize) -> Matrix {
    let mut rng = rand::rng();

    let mut data = Vec::new();
    for _ in 0..n_inputs * n_outputs {
        let value = rng.random_range(-0.01..0.01) * (2.0 / n_inputs as f32).sqrt();
        data.push(value);
    }

    Matrix::from_vec(data, n_inputs, n_outputs)
}
