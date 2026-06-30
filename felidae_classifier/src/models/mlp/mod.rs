// src/models/mlp/mod.r

pub mod activation;
pub mod initializer;
pub mod layer;
use crate::tensor::Matrix;
// use crate::tensor::Matrix;
// use crate::models::mlp::layer::Layer;
use rand::RngExt;
use crate::models::mlp::layer::Layer;

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(
        layers_sizes: &[usize],
        activation: fn(f32) -> f32,
        activation_derivative: fn(f32) -> f32,
        output_activation: fn(f32) -> f32,
        output_activation_derivative: fn(f32) -> f32,
    ) -> MLP {
        let mut new_layers: Vec<Layer> = Vec::new();

        for i in 0..layers_sizes.len() - 2 {
            new_layers.push(Layer::new(
                layers_sizes[i],
                layers_sizes[i + 1],
                activation,
                activation_derivative,
            ));
        }

        let last = layers_sizes.len() - 1;
        new_layers.push(Layer::new(
            layers_sizes[last - 1],
            layers_sizes[last],
            output_activation,
            output_activation_derivative,
        ));

        Self {
            layers: new_layers,
        }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        let mut current = x.clone();
        for i in 0..self.layers.len() {
            current = self.layers[i].forward(&current);
        }
        current
    }

    fn forward_with_cache(&self, x: &Matrix) -> Vec<Matrix> {
        let mut current = x.clone();
        let mut cache = Vec::new();

        cache.push(current.clone()); // cache[0] = input x

        for i in 0..self.layers.len() {
            current = self.layers[i].forward(&current);
            cache.push(current.clone()); // cache[i+1] = output of layer i
        }

        cache
    }

    fn backward(&mut self, cache: &Vec<Matrix>, label: usize, learning_rate: f32) {
        let n_classes = self.layers[self.layers.len() - 1].biases.cols;
        let mut expected_data = Vec::new();
        for j in 0..n_classes {
            if j == label {
                expected_data.push(1.0_f32);
            } else {
                expected_data.push(-1.0_f32);
            }
        }
        let expected = Matrix::from_vec(expected_data, 1, n_classes);

        let last_output = &cache[cache.len() - 1];
        let diff = last_output.sub(&expected);
        let deriv = last_output.map(self.layers[self.layers.len() - 1].activation_derivative);
        let mut delta = diff.mul_elementwise(&deriv);

        for l in (0..self.layers.len()).rev() {
            let layer_input = &cache[l];

            let old_weights = self.layers[l].weights.clone();

            let weight_grad = layer_input.transpose().dot(&delta);
            self.layers[l].weights = self.layers[l].weights.sub(&weight_grad.scale(learning_rate));

            self.layers[l].biases = self.layers[l].biases.sub(&delta.scale(learning_rate));

            if l > 0 {
                let propagated = delta.dot(&old_weights.transpose());
                let prev_deriv = cache[l].map(self.layers[l - 1].activation_derivative);
                delta = propagated.mul_elementwise(&prev_deriv);
            }
        }
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

    pub fn train(&mut self, x: &Matrix, labels: &[usize], epochs: usize, learning_rate: f32) {
        let mut rng = rand::rng();
        
        for epoch in 0..epochs {
            let i = rng.random_range(0..x.rows);
            
            let mut sample_data = Vec::new();

            for j in 0..x.cols {
                sample_data.push(x.get(i, j));
            }

            let sample = Matrix::from_vec(sample_data, 1, x.cols);

            let cache = self.forward_with_cache(&sample);

            self.backward(&cache, labels[i], learning_rate);

            if epoch % 100 == 0 {
                let acc = self.accuracy(x, labels);
                println!("Epoch {:>4} | Accuracy: {:.1}%", epoch, acc * 100.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::mlp::activation::{tanh, tanh_derivative};

    /// 3 clearly separated points, one per class — easy to learn
    fn make_simple_data() -> (Matrix, Vec<usize>) {
        let data = vec![
            0.1_f32, 0.1,  // cat
            0.5,     0.9,  // lion
            0.9,     0.1,  // cheetah
        ];
        (Matrix::from_vec(data, 3, 2), vec![0, 1, 2])
    }

    #[test]
    fn test_new_layer_count() {
        // sizes [2, 16, 8, 3] should produce 3 layers
        let mlp = MLP::new(&[2, 16, 8, 3], tanh, tanh_derivative, tanh, tanh_derivative);
        assert_eq!(mlp.layers.len(), 3);
    }

    #[test]
    fn test_forward_shape() {
        // 3 samples, 2 features in → 3 samples, 3 classes out
        let mlp = MLP::new(&[2, 16, 8, 3], tanh, tanh_derivative, tanh, tanh_derivative);
        let (x, _) = make_simple_data();
        let output = mlp.forward(&x);
        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 3);
    }

    #[test]
    fn test_forward_with_cache_length() {
        // cache should have 1 entry per layer + 1 for the input
        let mlp = MLP::new(&[2, 16, 8, 3], tanh, tanh_derivative, tanh, tanh_derivative);
        let (x, _) = make_simple_data();
        let cache = mlp.forward_with_cache(&x);
        // cache[0] = input, cache[1..=3] = layer outputs → 4 total
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn test_forward_with_cache_first_is_input() {
        // cache[0] must be the original input unchanged
        let mlp = MLP::new(&[2, 16, 8, 3], tanh, tanh_derivative, tanh, tanh_derivative);
        let (x, _) = make_simple_data();
        let cache = mlp.forward_with_cache(&x);
        assert_eq!(cache[0].get(0, 0), x.get(0, 0));
        assert_eq!(cache[0].get(2, 1), x.get(2, 1));
    }

    #[test]
    fn test_predict_returns_correct_length() {
        let mlp = MLP::new(&[2, 16, 8, 3], tanh, tanh_derivative, tanh, tanh_derivative);
        let (x, _) = make_simple_data();
        let preds = mlp.predict(&x);
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_predict_returns_valid_classes() {
        // every prediction must be a valid class index (0, 1 or 2)
        let mlp = MLP::new(&[2, 16, 8, 3], tanh, tanh_derivative, tanh, tanh_derivative);
        let (x, _) = make_simple_data();
        let preds = mlp.predict(&x);
        for p in &preds {
            assert!(*p < 3, "prediction {} is not a valid class", p);
        }
    }

    #[test]
    fn test_train_improves_accuracy() {
        // on 3 clearly separated points the MLP should reach 100%
        let mut mlp = MLP::new(&[2, 16, 8, 3], tanh, tanh_derivative, tanh, tanh_derivative);
        let (x, y) = make_simple_data();
        mlp.train(&x, &y, 5000, 0.05);
        assert_eq!(mlp.accuracy(&x, &y), 1.0);
    }

    #[test]
    fn test_accuracy_range() {
        // accuracy must always be between 0.0 and 1.0
        let mlp = MLP::new(&[2, 16, 8, 3], tanh, tanh_derivative, tanh, tanh_derivative);
        let (x, y) = make_simple_data();
        let acc = mlp.accuracy(&x, &y);
        assert!(acc >= 0.0 && acc <= 1.0);
    }
}
