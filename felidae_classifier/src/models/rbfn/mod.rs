// src/models/rbfn/mod.rs

pub mod basis;
pub mod centers;

use crate::tensor::Matrix;
use basis::build_phi;
use centers::kmeans;
use std::fs::File;
use std::io::{Read, Write};

pub struct RBFN {
    centers: Matrix,
    weights: Matrix,
    gamma: f32,
}

impl RBFN {
    pub fn train(x: &Matrix, y: &Matrix, k: usize, gamma: f32, kmeans_iters: usize, seed: u64) -> Self {
        // Step 1: find the K centers
        let centers = kmeans(x, k, kmeans_iters, seed);

        // Step 2: build the Phi matrix (N x K)
        let phi = build_phi(x, &centers, gamma);

        // Step 3: pseudo-inverse solve
        // W = (Phi^T Phi)^-1 Phi^T Y
        // I have encountered an issue when trying to inverse the Phi^T Phi  is singular or nearly
        // singular (which is common in high dimensions so my fix to that is to add a small lambda*I
        // W = (Phi^T Phi + lambda*I)^-1 Phi^T Y
        let phi_t = phi.transpose();
        let mut phi_t_phi = phi_t.dot(&phi);

        // add lambda to the diagonal
        let lambda = 1e-6;
        for i in 0..phi_t_phi.rows {
            let v = phi_t_phi.get(i, i) + lambda;
            phi_t_phi.set(i, i, v);
        }

        let inv = phi_t_phi.inverse();
        let phi_t_y = phi_t.dot(y);
        let weights = inv.dot(&phi_t_y);

        Self { centers, weights, gamma }
    }

    pub fn train_naive(x: &Matrix, y: &Matrix, gamma: f32) -> Self {
        let centers = x.clone();
        let phi = build_phi(x, &centers, gamma);
        let weights = phi.inverse().dot(y);

        Self { centers, weights, gamma }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        let phi = build_phi(x, &self.centers, self.gamma);
        phi.dot(&self.weights)
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        let output = self.forward(x);
        let mut predictions = Vec::new();

        for i in 0..output.rows {
            let mut best_class = 0;
            let mut best_value = output.get(i, 0);
            for j in 1..output.cols {
                if output.get(i, j) > best_value {
                    best_value = output.get(i, j);
                    best_class = j;
                }
            }
            predictions.push(best_class);
        }
        predictions
    }


    pub fn accuracy(&self, x: &Matrix, y: &Matrix) -> f32 {
        let predictions = self.predict(x);
        let mut correct = 0.0_f32;

        for i in 0..y.rows {
            let mut true_class = 0;
            let mut best = y.get(i, 0);
            for j in 1..y.cols {
                if y.get(i, j) > best {
                    best = y.get(i, j);
                    true_class = j;
                }
            }
            if predictions[i] == true_class {
                correct += 1.0;
            }
        }

        correct / y.rows as f32
    }

    /// Saves the trained RBFN to a binary file.
    ///
    /// File format:
    ///   [4 bytes] magic number "RBFv"
    ///   [4 bytes] format version (u32 little-endian, currently 1)
    ///   [4 bytes] gamma (f32 little-endian)
    ///   [4 bytes] centers.rows (u32) = K
    ///   [4 bytes] centers.cols (u32) = n_features
    ///   [K × n_features × 4 bytes] centers data (f32 each)
    ///   [4 bytes] weights.rows (u32) = K
    ///   [4 bytes] weights.cols (u32) = n_outputs
    ///   [K × n_outputs × 4 bytes] weights data (f32 each)
    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut file = match File::create(path) {
            Ok(f) => f,
            Err(e) => return Err(format!("Failed to create {}: {}", path, e)),
        };

        if let Err(e) = file.write_all(b"RBFv") {
            return Err(format!("write failed: {}", e));
        }

        let version: u32 = 1;
        if let Err(e) = file.write_all(&version.to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }

        if let Err(e) = file.write_all(&self.gamma.to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }

        if let Err(e) = file.write_all(&(self.centers.rows as u32).to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }
        if let Err(e) = file.write_all(&(self.centers.cols as u32).to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }
        for &v in &self.centers.data {
            if let Err(e) = file.write_all(&v.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
        }

        // weights matrix
        if let Err(e) = file.write_all(&(self.weights.rows as u32).to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }
        if let Err(e) = file.write_all(&(self.weights.cols as u32).to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }
        for &v in &self.weights.data {
            if let Err(e) = file.write_all(&v.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
        }

        Ok(())
    }

    /// Loads an RBFN previously written by save.
    pub fn load(path: &str) -> Result<RBFN, String> {
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
        if &magic != b"RBFv" {
            return Err(format!("Not an RBFN model file: bad magic number {:?}", magic));
        }

        let version = read_u32(&mut file)?;
        if version != 1 {
            return Err(format!(
                "Unsupported RBFN file version: {} (this build supports version 1)",
                version
            ));
        }

        let gamma = read_f32(&mut file)?;

        let c_rows = read_u32(&mut file)? as usize;
        let c_cols = read_u32(&mut file)? as usize;
        let mut c_data = Vec::with_capacity(c_rows * c_cols);
        for _ in 0..(c_rows * c_cols) {
            c_data.push(read_f32(&mut file)?);
        }
        let centers = Matrix::from_vec(c_data, c_rows, c_cols);

        let w_rows = read_u32(&mut file)? as usize;
        let w_cols = read_u32(&mut file)? as usize;
        // gamma
        let mut w_data = Vec::with_capacity(w_rows * w_cols);
        for _ in 0..(w_rows * w_cols) {
            w_data.push(read_f32(&mut file)?);
        }
        let weights = Matrix::from_vec(w_data, w_rows, w_cols);

        Ok(RBFN { centers, weights, gamma })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a simple 2-class, 2D dataset: one cluster near (0,0) labeled
    /// class 0, one near (5,5) labeled class 1. One-hot targets.
    fn toy_dataset() -> (Matrix, Matrix) {
        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            0.2, 0.1,
            0.1, 0.3,
            5.0, 5.0,
            5.1, 4.9,
            4.8, 5.2,
        ], 6, 2);
        // one-hot: class 0 = [1, 0], class 1 = [0, 1]
        let y = Matrix::from_vec(vec![
            1.0, 0.0,
            1.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            0.0, 1.0,
            0.0, 1.0,
        ], 6, 2);
        (x, y)
    }

    #[test]
    fn test_naive_perfectly_fits_training_data() {
        // The naive RBFN interpolates every training point exactly,
        // so training accuracy must be 100%.
        let (x, y) = toy_dataset();
        let model = RBFN::train_naive(&x, &y, 1.0);
        let acc = model.accuracy(&x, &y);
        assert!((acc - 1.0).abs() < 1e-6, "naive should fit training perfectly, got {}", acc);
    }

    #[test]
    fn test_kcenters_separates_two_clusters() {
        // With 2 centers on two clean clusters, training accuracy should be perfect.
        let (x, y) = toy_dataset();
        let model = RBFN::train(&x, &y, 2, 1.0, 20, 0);
        let acc = model.accuracy(&x, &y);
        assert!(acc > 0.99, "expected clean separation, got {}", acc);
    }

    #[test]
    fn test_predict_length_matches_input() {
        let (x, y) = toy_dataset();
        let model = RBFN::train(&x, &y, 2, 1.0, 20, 0);
        let preds = model.predict(&x);
        assert_eq!(preds.len(), x.rows);
    }

    #[test]
    fn test_predict_classes_are_in_range() {
        let (x, y) = toy_dataset();
        let model = RBFN::train(&x, &y, 2, 1.0, 20, 0);
        let preds = model.predict(&x);
        for p in preds {
            assert!(p < 2, "class index {} out of range", p);
        }
    }

    #[test]
    fn test_generalizes_to_nearby_point() {
        // A point near the class-0 cluster should be predicted as class 0.
        let (x, y) = toy_dataset();
        let model = RBFN::train(&x, &y, 2, 1.0, 20, 0);
        let test_point = Matrix::from_vec(vec![0.15, 0.15], 1, 2);
        let pred = model.predict(&test_point);
        assert_eq!(pred[0], 0, "point near origin should be class 0");
    }

    #[test]
    fn test_forward_output_shape() {
        let (x, y) = toy_dataset();
        let model = RBFN::train(&x, &y, 2, 1.0, 20, 0);
        let out = model.forward(&x);
        assert_eq!(out.rows, 6);
        assert_eq!(out.cols, 2); // one column per class
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let (x, y) = toy_dataset();
        let original = RBFN::train(&x, &y, 2, 1.0, 20, 0);

        let path = "/tmp/test_rbfn_roundtrip.bin";
        original.save(path).expect("save failed");
        let loaded = RBFN::load(path).expect("load failed");

        // Predictions must match exactly
        assert_eq!(original.predict(&x), loaded.predict(&x));

        // gamma and matrix dims preserved
        assert_eq!(original.gamma, loaded.gamma);
        assert_eq!(original.centers.rows, loaded.centers.rows);
        assert_eq!(original.weights.data.len(), loaded.weights.data.len());
        for i in 0..original.weights.data.len() {
            assert_eq!(original.weights.data[i], loaded.weights.data[i]);
        }
    }

    #[test]
    fn test_load_bad_magic_fails() {
        let path = "/tmp/test_rbfn_bad_magic.bin";
        std::fs::write(path, b"NOTANRBFFILE").unwrap();
        assert!(RBFN::load(path).is_err());
    }
}
