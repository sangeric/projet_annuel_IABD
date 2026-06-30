// src/models/mlp/layer.rs

use crate::tensor::Matrix;
use crate::models::mlp::initializer::he_init;
use crate::models::mlp::activation::lookup;

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: fn(f32) -> f32,
    pub activation_derivative: fn(f32) -> f32,
    pub activation_name: String,
}

impl Layer {
    /// Creates a new layer.
    ///
    /// - `n_inputs`: number of inputs to each neuron
    /// - `n_outputs`: number of neurons (= outputs)
    /// - `activation_name`: one of "tanh", "relu", or "identity"
    ///
    /// Returns Err if `activation_name` isn't recognised.
    pub fn new(n_inputs: usize, n_outputs: usize, activation_name: &str,) -> Result<Layer, String> {
        let (activation, activation_derivative) = lookup(activation_name)?;

        Ok(Layer {
            weights: he_init(n_inputs, n_outputs),
            biases: Matrix::zeros(1, n_outputs),
            activation,
            activation_derivative,
            activation_name: activation_name.to_string(),
        })
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        let z = x.dot(&self.weights).add_bias_row(&self.biases);
        z.map(self.activation)
    }
}
