// src/models/mlp/mod.rs

pub mod activation;
pub mod initializer;
pub mod layer;

use std::fs::File;
use std::io::{Read, Write};

use crate::tensor::Matrix;
use crate::models::mlp::layer::Layer;
use crate::models::mlp::activation::lookup;
use crate::training::loss::mse_loss;
use crate::training::shuffle::shuffled_indices;
use tensorboard_rs::summary_writer::SummaryWriter;

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    /// Constructs a new MLP.
    ///
    /// - layers_sizes — sizes of each layer, e.g. "&[20, 16, 3]"
    /// - activation — name of activation function for hidden layers
    /// - output_activation — name of activation for the final layer
    ///
    /// Both activation names must be recognised by activation::lookup
    /// (currently "tanh", "relu", or "identity").

    pub fn new(layers_sizes: &[usize], activation: &str, output_activation: &str) -> Result<MLP, String> {
        let mut new_layers: Vec<Layer> = Vec::new();

        for i in 0..layers_sizes.len() - 2 {
            new_layers.push(Layer::new(
                layers_sizes[i],
                layers_sizes[i + 1],
                activation,
            )?);
        }

        let last = layers_sizes.len() - 1;
        new_layers.push(Layer::new(
            layers_sizes[last - 1],
            layers_sizes[last],
            output_activation,
        )?);

        Ok(MLP { layers: new_layers })
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
        cache.push(current.clone());
        for i in 0..self.layers.len() {
            current = self.layers[i].forward(&current);
            cache.push(current.clone());
        }
        cache
    }


    fn backward(&mut self, cache: &Vec<Matrix>, expected: &Matrix, learning_rate: f32) {
        let last_output = &cache[cache.len() - 1];
        let diff = last_output.sub(expected);
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


    pub fn accuracy(&self, x: &Matrix, y: &Matrix) -> f32 {
        let predictions = self.predict(x);
        let mut correct = 0;

        for i in 0..predictions.len() {
            let mut true_class = 0;
            let mut best_score = y.get(i, 0);
            for j in 1..y.cols {
                let score = y.get(i, j);
                if score > best_score {
                    best_score = score;
                    true_class = j;
                }
            }
            if predictions[i] == true_class {
                correct += 1;
            }
        }

        correct as f32 / predictions.len() as f32
    }


    pub fn train(
        &mut self,
        x_train: &Matrix,
        y_train: &Matrix,
        x_test: &Matrix,
        y_test: &Matrix,
        epochs: usize,
        learning_rate: f32,
        log_dir: &str,
    ) {
        let mut writer = SummaryWriter::new(&log_dir.to_string());

        for epoch in 0..epochs {
            let order = shuffled_indices(x_train.rows, epoch as u64);
            let mut total_loss = 0.0_f32;

            for &i in &order {

                let mut sample_data = Vec::new();
                for j in 0..x_train.cols {
                    sample_data.push(x_train.get(i, j));
                }
                let sample = Matrix::from_vec(sample_data, 1, x_train.cols);


                let mut expected_data = Vec::new();
                for j in 0..y_train.cols {
                    expected_data.push(y_train.get(i, j));
                }
                let expected = Matrix::from_vec(expected_data, 1, y_train.cols);

                let cache = self.forward_with_cache(&sample);
                let output = &cache[cache.len() - 1];
                total_loss += mse_loss(output, &expected);

                self.backward(&cache, &expected, learning_rate);
            }

            let avg_loss = total_loss / x_train.rows as f32;
            let train_acc = self.accuracy(x_train, y_train);
            let test_acc = self.accuracy(x_test, y_test);

            writer.add_scalar("loss", avg_loss, epoch);
            writer.add_scalar("train_accuracy", train_acc, epoch);
            writer.add_scalar("test_accuracy", test_acc, epoch);
            writer.flush();

            println!(
                "Epoch {:>3} | Loss: {:.4} | Train: {:.1}% | Test: {:.1}%",
                epoch, avg_loss, train_acc * 100.0, test_acc * 100.0
            );
        }
    }


    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut file = match File::create(path) {
            Ok(f) => f,
            Err(e) => return Err(format!("Failed to create {}: {}", path, e)),
        };


        if let Err(e) = file.write_all(b"MLPv") {
            return Err(format!("write failed: {}", e));
        }


        let version: u32 = 1;
        if let Err(e) = file.write_all(&version.to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }


        let n_layers = self.layers.len() as u32;
        if let Err(e) = file.write_all(&n_layers.to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }


        for layer in &self.layers {
            let name_bytes = layer.activation_name.as_bytes();
            let name_len = name_bytes.len() as u32;
            if let Err(e) = file.write_all(&name_len.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
            if let Err(e) = file.write_all(name_bytes) {
                return Err(format!("write failed: {}", e));
            }

            let w_rows = layer.weights.rows as u32;
            let w_cols = layer.weights.cols as u32;
            if let Err(e) = file.write_all(&w_rows.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
            if let Err(e) = file.write_all(&w_cols.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
            for &value in &layer.weights.data {
                if let Err(e) = file.write_all(&value.to_le_bytes()) {
                    return Err(format!("write failed: {}", e));
                }
            }


            for &value in &layer.biases.data {
                if let Err(e) = file.write_all(&value.to_le_bytes()) {
                    return Err(format!("write failed: {}", e));
                }
            }
        }

        Ok(())
    }


    pub fn load(path: &str) -> Result<MLP, String> {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) => return Err(format!("Failed to open {}: {}", path, e)),
        };


        let read_u32 = |file: &mut File| -> Result<u32, String> {
            let mut buf = [0u8; 4];
            if let Err(e) = file.read_exact(&mut buf) {
                return Err(format!("read failed: {}", e));
            }
            Ok(u32::from_le_bytes(buf))
        };
        let read_f32 = |file: &mut File| -> Result<f32, String> {
            let mut buf = [0u8; 4];
            if let Err(e) = file.read_exact(&mut buf) {
                return Err(format!("read failed: {}", e));
            }
            Ok(f32::from_le_bytes(buf))
        };


        let mut magic = [0u8; 4];
        if let Err(e) = file.read_exact(&mut magic) {
            return Err(format!("read failed: {}", e));
        }
        if &magic != b"MLPv" {
            return Err(format!("Not an MLP model file: bad magic number {:?}", magic));
        }


        let version = read_u32(&mut file)?;
        if version != 1 {
            return Err(format!(
                "Unsupported MLP file version: {} (this build supports version 1)",
                version
            ));
        }

        let n_layers = read_u32(&mut file)? as usize;

        let mut layers: Vec<Layer> = Vec::new();
        for _ in 0..n_layers {

            let name_len = read_u32(&mut file)? as usize;
            let mut name_bytes = vec![0u8; name_len];
            if let Err(e) = file.read_exact(&mut name_bytes) {
                return Err(format!("read failed: {}", e));
            }
            let name = match String::from_utf8(name_bytes) {
                Ok(s) => s,
                Err(e) => return Err(format!("invalid utf8 in activation name: {}", e)),
            };


            let (activation, activation_derivative) = lookup(&name)?;


            let w_rows = read_u32(&mut file)? as usize;
            let w_cols = read_u32(&mut file)? as usize;
            let mut w_data = Vec::with_capacity(w_rows * w_cols);
            for _ in 0..(w_rows * w_cols) {
                w_data.push(read_f32(&mut file)?);
            }
            let weights = Matrix::from_vec(w_data, w_rows, w_cols);


            let mut b_data = Vec::with_capacity(w_cols);
            for _ in 0..w_cols {
                b_data.push(read_f32(&mut file)?);
            }
            let biases = Matrix::from_vec(b_data, 1, w_cols);

            layers.push(Layer {
                weights,
                biases,
                activation,
                activation_derivative,
                activation_name: name,
            });
        }

        Ok(MLP { layers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_data() -> (Matrix, Matrix) {
        let x_data = vec![
            0.1_f32, 0.1,
            0.5,     0.9,
            0.9,     0.1,
        ];

        let y_data = vec![
             1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
            -1.0, -1.0,  1.0,
        ];
        (
            Matrix::from_vec(x_data, 3, 2),
            Matrix::from_vec(y_data, 3, 3),
        )
    }

    #[test]
    fn test_new_layer_count() {
        let mlp = MLP::new(&[2, 16, 8, 3], "tanh", "tanh").unwrap();
        assert_eq!(mlp.layers.len(), 3);
    }

    #[test]
    fn test_new_unknown_activation_errors() {
        let result = MLP::new(&[2, 16, 8, 3], "fakeactivation", "tanh");
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_shape() {
        let mlp = MLP::new(&[2, 16, 8, 3], "tanh", "tanh").unwrap();
        let (x, _) = make_simple_data();
        let output = mlp.forward(&x);
        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 3);
    }

    #[test]
    fn test_forward_with_cache_length() {
        let mlp = MLP::new(&[2, 16, 8, 3], "tanh", "tanh").unwrap();
        let (x, _) = make_simple_data();
        let cache = mlp.forward_with_cache(&x);
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn test_forward_with_cache_first_is_input() {
        let mlp = MLP::new(&[2, 16, 8, 3], "tanh", "tanh").unwrap();
        let (x, _) = make_simple_data();
        let cache = mlp.forward_with_cache(&x);
        assert_eq!(cache[0].get(0, 0), x.get(0, 0));
        assert_eq!(cache[0].get(2, 1), x.get(2, 1));
    }

    #[test]
    fn test_predict_returns_correct_length() {
        let mlp = MLP::new(&[2, 16, 8, 3], "tanh", "tanh").unwrap();
        let (x, _) = make_simple_data();
        let preds = mlp.predict(&x);
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_predict_returns_valid_classes() {
        let mlp = MLP::new(&[2, 16, 8, 3], "tanh", "tanh").unwrap();
        let (x, _) = make_simple_data();
        let preds = mlp.predict(&x);
        for p in &preds {
            assert!(*p < 3, "prediction {} is not a valid class", p);
        }
    }

    #[test]
    fn test_train_improves_accuracy() {
        let mut mlp = MLP::new(&[2, 16, 8, 3], "tanh", "tanh").unwrap();
        let (x, y) = make_simple_data();
        mlp.train(&x, &y, &x, &y, 50, 0.05, "/tmp/test_runs");
        assert_eq!(mlp.accuracy(&x, &y), 1.0);
    }

    #[test]
    fn test_accuracy_range() {
        let mlp = MLP::new(&[2, 16, 8, 3], "tanh", "tanh").unwrap();
        let (x, y) = make_simple_data();
        let acc = mlp.accuracy(&x, &y);
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_save_and_load_roundtrip() {

        let mut original = MLP::new(&[2, 4, 3], "tanh", "tanh").unwrap();
        let (x, y) = make_simple_data();
        original.train(&x, &y, &x, &y, 5, 0.05, "/tmp/test_runs");

        let path = "/tmp/test_mlp_roundtrip.bin";
        original.save(path).expect("save failed");

        let loaded = MLP::load(path).expect("load failed");


        assert_eq!(original.predict(&x), loaded.predict(&x));


        assert_eq!(original.layers.len(), loaded.layers.len());


        for li in 0..original.layers.len() {
            let orig = &original.layers[li];
            let ld = &loaded.layers[li];
            assert_eq!(orig.activation_name, ld.activation_name);
            assert_eq!(orig.weights.rows, ld.weights.rows);
            assert_eq!(orig.weights.cols, ld.weights.cols);
            for i in 0..orig.weights.data.len() {
                assert_eq!(orig.weights.data[i], ld.weights.data[i]);
            }
            for i in 0..orig.biases.data.len() {
                assert_eq!(orig.biases.data[i], ld.biases.data[i]);
            }
        }
    }

    #[test]
    fn test_load_bad_magic_fails() {
        let path = "/tmp/test_mlp_bad_magic.bin";
        std::fs::write(path, b"NOTAMLPFILE").unwrap();

        let result = MLP::load(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_missing_file_fails() {
        let result = MLP::load("/tmp/this_file_does_not_exist_xyz123.bin");
        assert!(result.is_err());
    }
}
