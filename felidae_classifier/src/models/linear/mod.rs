// src/models/linear/mod.rs

use crate::tensor::Matrix;
use rand::RngExt;

pub mod transform;

/// A linear classifier: output = X · W + b
/// Trained with softmax + cross-entropy loss via gradient descent
pub struct LinearClassifier {
    /// Weight matrix, shape (n_features × n_classes)
    weights: Matrix,
    /// Bias vector, shape (1 × n_classes)
    bias: Matrix,
    /// Number of classes
    n_classes: usize,
    /// Learning rate for gradient descent
    learning_rate: f32,
}

impl LinearClassifier {
    /// Creates a new LinearClassifier with small random weights
    pub fn new(n_features: usize, n_classes: usize, learning_rate: f32) -> Self {
        let mut rng = rand::rng();

        // Small random weights to start — prevents any class from dominating early
        let mut weight_data = Vec::new();
        for _ in 0..n_features * n_classes {
            weight_data.push(rng.random_range(-0.01..0.01));
        }

        // Biases start at zero
        let mut bias_data = Vec::new();
        for _ in 0..n_classes {
            bias_data.push(0.0_f32);
        }

        Self {
            weights: Matrix::from_vec(weight_data, n_features, n_classes),
            bias: Matrix::from_vec(bias_data, 1, n_classes),
            n_classes,
            learning_rate,
        }
    }

    /// Forward pass — computes raw scores (logits) for each class
    /// Input shape:  (N × n_features)
    /// Output shape: (N × n_classes)
    fn forward(&self, x: &Matrix) -> Matrix {
        // X · W + b
        x.dot(&self.weights).add_bias_row(&self.bias)
    }

    /// Softmax — converts raw scores into probabilities
    /// Each row sums to 1.0, like a probability distribution over classes
    fn softmax(&self, logits: &Matrix) -> Matrix {
        let mut result = Matrix::zeros(logits.rows, logits.cols);

        for i in 0..logits.rows {
            // Find the max score in this row for numerical stability
            // This prevents exp() from overflowing — standard trick
            let mut max = f32::NEG_INFINITY;
            for j in 0..logits.cols {
                if logits.get(i, j) > max {
                    max = logits.get(i, j);
                }
            }

            // Compute exp(score - max) for each class
            let mut exps = Vec::new();
            for j in 0..logits.cols {
                exps.push((logits.get(i, j) - max).exp());
            }

            // Sum all the exp values so we can normalize
            let mut sum = 0.0_f32;
            for val in &exps {
                sum += val;
            }

            // Divide each exp by the sum — now the row sums to 1.0
            for j in 0..logits.cols {
                result.set(i, j, exps[j] / sum);
            }
        }

        result
    }

    /// Cross-entropy loss — measures how wrong the predictions are
    /// Returns a single f32 scalar — lower is better
    fn cross_entropy_loss(&self, probs: &Matrix, labels: &[usize]) -> f32 {
        let mut total_loss = 0.0_f32;

        for i in 0..labels.len() {
            let true_class = labels[i];
            // Clamp to avoid log(0) which would give -infinity
            let prob = probs.get(i, true_class).max(1e-7);
            total_loss -= prob.ln();
        }

        // Return the average loss over all samples
        total_loss / labels.len() as f32
    }

    /// Compute gradients and update weights + bias (one gradient descent step)
    /// The math:
    ///   dL/dW = X^T · (probs - one_hot(labels)) / N
    ///   dL/db = sum(probs - one_hot(labels), axis=0) / N
    fn backward(&mut self, x: &Matrix, probs: &Matrix, labels: &[usize]) {
        let n = labels.len() as f32;

        // Build the error matrix (probs - one_hot), shape (N × n_classes)
        // one_hot means: true class gets 1.0, all others get 0.0
        // So subtracting it gives the "error" per class per sample
        let mut delta = probs.clone();
        for i in 0..labels.len() {
            let true_class = labels[i];
            let current = delta.get(i, true_class);
            delta.set(i, true_class, current - 1.0);
        }

        // Weight gradient: X^T · delta / N — shape (n_features × n_classes)
        let weight_grad = x.transpose().dot(&delta).scale(1.0 / n);

        // Bias gradient: sum each column of delta / N — shape (1 × n_classes)
        let mut bias_grad = Matrix::zeros(1, self.n_classes);
        for j in 0..self.n_classes {
            let mut col_sum = 0.0_f32;
            for i in 0..delta.rows {
                col_sum += delta.get(i, j);
            }
            bias_grad.set(0, j, col_sum / n);
        }

        // Gradient descent update: W = W - lr * dL/dW
        self.weights = self.weights.sub(&weight_grad.scale(self.learning_rate));
        self.bias = self.bias.sub(&bias_grad.scale(self.learning_rate));
    }

    /// Full training loop — runs forward + backward for a number of epochs
    pub fn train(&mut self, x: &Matrix, labels: &[usize], epochs: usize) {
        for epoch in 0..epochs {
            let logits = self.forward(x);
            let probs = self.softmax(&logits);
            let loss = self.cross_entropy_loss(&probs, labels);
            self.backward(x, &probs, labels);

            // Print every 100 epochs so you can watch it converge
            if epoch % 100 == 0 {
                let acc = self.accuracy(x, labels);
                println!(
                    "Epoch {:>4} | Loss: {:.4} | Accuracy: {:.1}%",
                    epoch,
                    loss,
                    acc * 100.0
                );
            }
        }
    }

    /// Predict the class index for each sample
    /// For each row in X, returns the index of the highest probability class
    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        let logits = self.forward(x);
        let probs = self.softmax(&logits);

        let mut predictions = Vec::new();

        for i in 0..probs.rows {
            // Find which class has the highest probability for this sample
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

    /// Fraction of correct predictions
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_data() -> (Matrix, Vec<usize>) {
        // 3 clearly separated points, one per class
        let data = vec![
            0.1_f32, 0.1,  // cat
            0.5,     0.9,  // lion
            0.9,     0.1,  // cheetah
        ];
        (Matrix::from_vec(data, 3, 2), vec![0, 1, 2])
    }

    #[test]
    fn test_forward_shape() {
        let clf = LinearClassifier::new(2, 3, 0.1);
        let (x, _) = make_simple_data();
        let logits = clf.forward(&x);
        // (3 samples × 2 features) · (2 × 3 weights) = (3 × 3)
        assert_eq!(logits.rows, 3);
        assert_eq!(logits.cols, 3);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let clf = LinearClassifier::new(2, 3, 0.1);
        let (x, _) = make_simple_data();
        let logits = clf.forward(&x);
        let probs = clf.softmax(&logits);

        for i in 0..probs.rows {
            let mut row_sum = 0.0_f32;
            for j in 0..probs.cols {
                row_sum += probs.get(i, j);
            }
            assert!((row_sum - 1.0).abs() < 1e-5, "Row {} sums to {}", i, row_sum);
        }
    }

    #[test]
    fn test_train_improves_accuracy() {
        let mut clf = LinearClassifier::new(2, 3, 0.1);
        let (x, y) = make_simple_data();
        // After 500 epochs on 3 clearly separated points it should reach 100%
        clf.train(&x, &y, 500);
        assert_eq!(clf.accuracy(&x, &y), 1.0);
    }

    #[test]
    fn test_predict_returns_correct_length() {
        let clf = LinearClassifier::new(2, 3, 0.1);
        let (x, _) = make_simple_data();
        let preds = clf.predict(&x);
        assert_eq!(preds.len(), 3);
    }
}
