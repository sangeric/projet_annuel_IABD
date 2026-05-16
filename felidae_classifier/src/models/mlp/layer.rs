// src/models/mlp/layer.rs

use cargo::tensor::Matrix;
use cargo::models::mlp::initializer::he_init;

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: fn(f32) -> f32,
    pub activation_derivative: fn(f32) -> f32,
}

impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize, activation: fn(f32) -> f32, activation_derivative: fn(f32) -> f32,) -> Self {
        Self {
            weights: he_init(n_inputs, n_outputs),
            biases: Matrix::zeros(1, n_outputs),
            activation,
            activation_derivative,
        }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        let z = x.dot(&self.weights).add_bias_row(&self.biases);
        z.map(self.activation)
    }
}
