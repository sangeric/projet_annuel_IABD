use crate::tensor::Matrix;
use rand::RngExt;
mod rosenblatt;

use rand::SeedableRng;
//classification model perceptron
pub struct Rosenblatt {
    learning_rate: f32,
    target_class: usize,
    weights: Matrix,
    bias:f32,
}
