// src/models/linear/mod.rs

use crate::tensor::Matrix;
use rand::RngExt;

pub mod transform;
mod regression;


pub struct LinearClassifier {
    weights: Matrix,

    bias: Matrix,

    n_classes: usize,

    learning_rate: f32,
}

pub struct Regression{
    weights: Matrix,
}



impl LinearClassifier {

    pub fn new(n_features: usize, n_classes: usize, learning_rate: f32) -> Self {
        let mut rng = rand::rng();


        let mut weight_data = Vec::new();
        for _ in 0..n_features * n_classes {
            weight_data.push(rng.random_range(-0.01..0.01));
        }


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


    fn forward(&self, x: &Matrix) -> Matrix {
        x.dot(&self.weights).add_bias_row(&self.bias)
    }


    fn softmax(&self, logits: &Matrix) -> Matrix {
        let mut result = Matrix::zeros(logits.rows, logits.cols);

        for i in 0..logits.rows {
            let mut max = f32::NEG_INFINITY;
            for j in 0..logits.cols {
                if logits.get(i, j) > max {
                    max = logits.get(i, j);
                }
            }


            let mut exps = Vec::new();
            for j in 0..logits.cols {
                exps.push((logits.get(i, j) - max).exp());
            }


            let mut sum = 0.0_f32;
            for val in &exps {
                sum += val;
            }


            for j in 0..logits.cols {
                result.set(i, j, exps[j] / sum);
            }
        }

        result
    }


    fn cross_entropy_loss(&self, probs: &Matrix, labels: &[usize]) -> f32 {
        let mut total_loss = 0.0_f32;

        for i in 0..labels.len() {
            let true_class = labels[i];

            let prob = probs.get(i, true_class).max(1e-7);
            total_loss -= prob.ln();
        }


        total_loss / labels.len() as f32
    }


    fn backward(&mut self, x: &Matrix, probs: &Matrix, labels: &[usize]) {
        let n = labels.len() as f32;


        let mut delta = probs.clone();
        for i in 0..labels.len() {
            let true_class = labels[i];
            let current = delta.get(i, true_class);
            delta.set(i, true_class, current - 1.0);
        }


        let weight_grad = x.transpose().dot(&delta).scale(1.0 / n);


        let mut bias_grad = Matrix::zeros(1, self.n_classes);
        for j in 0..self.n_classes {
            let mut col_sum = 0.0_f32;
            for i in 0..delta.rows {
                col_sum += delta.get(i, j);
            }
            bias_grad.set(0, j, col_sum / n);
        }


        self.weights = self.weights.sub(&weight_grad.scale(self.learning_rate));
        self.bias = self.bias.sub(&bias_grad.scale(self.learning_rate));
    }


    pub fn train(&mut self, x: &Matrix, labels: &[usize], epochs: usize) {
        for epoch in 0..epochs {
            let logits = self.forward(x);
            let probs = self.softmax(&logits);
            let loss = self.cross_entropy_loss(&probs, labels);
            self.backward(x, &probs, labels);


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


    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        let logits = self.forward(x);
        let probs = self.softmax(&logits);

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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_data() -> (Matrix, Vec<usize>) {

        let data = vec![
            0.1_f32, 0.1,
            0.5,     0.9,
            0.9,     0.1,
        ];
        (Matrix::from_vec(data, 3, 2), vec![0, 1, 2])
    }

    #[test]
    fn test_forward_shape() {
        let clf = LinearClassifier::new(2, 3, 0.1);
        let (x, _) = make_simple_data();
        let logits = clf.forward(&x);

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
