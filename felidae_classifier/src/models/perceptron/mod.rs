use crate::tensor::Matrix;
mod rosenblatt;
mod rosenblatt_classifier;

//classification model perceptron
pub struct Rosenblatt {
    learning_rate: f32,
    weights: Matrix,
    bias:f32,
}

pub struct RosenblattClassifier{
    cat : Rosenblatt,
    lion : Rosenblatt,
    cheetah : Rosenblatt,
}