// src/models/mlp/mod.rs

pub mod activation;
pub mod initializer;
pub mod layer;

use crate::tensor::Matrix;
use crate::models::mlp::layer::Layer;

pub struct MLP {
    layers: Vec<Layer>,
    output_activation: fn(&Matrix) -> Matrix,
}

impl MLP {
    pub fn new(layers_sizes: &[usize], activation: fn(f32) -> f32, activation_derivative: fn(f32) -> f32, output_activation: fn(&Matrix) -> Matrix) -> MLP {
        let mut new_layers: Vec<Layer> = Vec::new();

        for i in 0..layers_sizes.len() - 2 {
            new_layers.push(Layer::new(layers_sizes[i], layers_sizes[i + 1], activation, activation_derivative));
        }

        let last = layers_sizes.len() - 1;
        new_layers.push(Layer::new(layers_sizes[last - 1], layers_sizes[last], (|x: f32| x) as fn(f32) -> f32, (|_: f32| 1.0_f32) as fn(f32) -> f32));

        Self {
            layers: new_layers,
            output_activation,
        }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        let mut current = x.clone();
        for i in 0..self.layers.len() {
            current = self.layers[i].forward(&current);
        }
        (self.output_activation)(&current)
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        let probs = self.forward(x);
    
        let mut predictions = Vec::new();

        for i in 0..probs.rows {
            let mut best_class = 0;
            let mut best_score = probs.get(i, 0);

            for j in 1..probs.cols {
                let score = probs.get(i, j);
                if score > best_score {
                    best_score = score;
                    best_class = j;
                }
            }
            
            predictions.push(best_class);
        }

        predictions
    }

    pub fn accuracy(&self, x: &Matrix, labels: &[usize]) -> f32 {
        let predictions = self.predict(x);

        let mut correct = 0;
        for i in 0..predictions.len() {
            if predictions[i] == labels[i] {
                correct += 1;
            }
        }

        correct as f32 / labels.len() as f32
    }

    fn forward_with_cache(&self, x: &Matrix) -> Vec<Matrix> {
        let mut current = x.clone();
        let mut cache = Vec::new();

        cache.push(current.clone());

        for i in 0..self.layers.len() {
            current = self.layers[i].forward(&current);
            cache.push(current.clone());
        }

        cache.push((self.output_activation)(&current));

        cache
    }

    fn backward(&mut self, cache: &Vec<Matrix>, label: usize, learning_rate: f32) {
        // Step 1 — build the expected output vector
        // shape (1 × n_classes), -1.0 everywhere except +1.0 at label index
        
        let n_classes = self.layers[self.layers.len()-1].biases.cols;
        let mut expected_data = Vec::new();

        for j in 0..n_classes {
            if j == label {
                expected_data.push(1.0_f32);
            } else {
                expected_data.push(-1.0_f32);
            }
        }
        let expected = Matrix::from_vec(expected_data, 1, n_classes);

        // Step 2 — compute delta for the output layer
        // delta = activation_derivative(cache[last]) * (cache[last] - expected)
        // elementwise multiply
        


        // Step 3 — loop backwards through layers
        // for each layer:
        //   a) update weights: cache[l].transpose().dot(delta) scaled by learning_rate
        //   b) update biases: delta scaled by learning_rate  
        //   c) compute delta for previous layer:
        //      propagated = delta.dot(weights.transpose())
        //      delta_prev = propagated.mul_elementwise(cache[l].map(activation_derivative))

        // Note: cache[0] is the input x
        // cache[l] is the input TO layer l (not the output)
        // so cache[l+1] is the output of layer l
        
}

    // pub fn train(&mut self, x: Matrix, labels: &[usize], epochs: usize, learning_rate: f32) {
        
    // }
}

