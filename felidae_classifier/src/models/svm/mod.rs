// src/models/svm/mod.rs

pub mod kernel;
pub mod smo;
pub mod multiclass;

use crate::tensor::Matrix;
use kernel::Kernel;
use smo::solve_dual;
use std::fs::File;
use std::io::{Read, Write};

pub struct SVM {
    support_vectors: Matrix,
    sv_weights: Vec<f32>,
    bias: f32,
    kernel: Kernel,
}

const ALPHA_THRESHOLD: f32 = 1e-4;

impl SVM {
    pub fn train(x: &Matrix, y: &[f32], kernel: Kernel, c: f32) -> Self {
        let alpha = solve_dual(x, y, &kernel, c);


        let mut sv_indices = Vec::new();
        for (i, &a) in alpha.iter().enumerate() {
            if a > ALPHA_THRESHOLD {
                sv_indices.push(i);
            }
        }

        let n_features = x.cols;
        let mut sv_data = Vec::with_capacity(sv_indices.len() * n_features);
        let mut sv_weights = Vec::with_capacity(sv_indices.len());

        for &i in &sv_indices {
            for j in 0..n_features {
                sv_data.push(x.get(i, j));
            }
            sv_weights.push(alpha[i] * y[i]);
        }

        let support_vectors = Matrix::from_vec(sv_data, sv_indices.len(), n_features);


        let mut bias_sum = 0.0_f32;
        let mut n_free = 0;
        for &i in &sv_indices {
            if alpha[i] < c - ALPHA_THRESHOLD {
                let mut decision = 0.0_f32;
                for &j in &sv_indices {
                    let k_ij = kernel.compute(x, j, i);
                    decision += alpha[j] * y[j] * k_ij;
                }
                bias_sum += y[i] - decision; 
                n_free += 1;
            }
        }
        let bias = if n_free == 0 {
            0.0
        } else {
            bias_sum / n_free as f32
        };

        Self { support_vectors, sv_weights, bias, kernel }
    }

    fn decision_value(&self, x: &Matrix, row: usize) -> f32 {
        let mut sum = 0.0_f32;
        for n in 0..self.support_vectors.rows {
            let k = self.kernel.compute_cross(&self.support_vectors, n, x, row);
            sum += self.sv_weights[n] * k;
        }
        sum + self.bias
    }

    pub fn forward(&self, x: &Matrix) -> Vec<f32> {
        (0..x.rows).map(|row| self.decision_value(x, row)).collect()
    }

    pub fn predict(&self, x: &Matrix) -> Vec<f32> {
        self.forward(x)
            .into_iter()
            .map(|v| if v >= 0.0 { 1.0 } else { -1.0 })
            .collect()
    }

    pub fn accuracy(&self, x: &Matrix, y: &[f32]) -> f32 {
        let preds = self.predict(x);
        let mut correct = 0.0_f32;
        for i in 0..y.len() {
            if preds[i] == y[i] {
                correct += 1.0;
            }
        }
        correct / y.len() as f32
    }

    pub fn n_support_vectors(&self) -> usize {
        self.support_vectors.rows
    }


    pub fn write_to(&self, file: &mut File) -> Result<(), String> {
        let (kernel_type, gamma): (u8, f32) = match self.kernel {
            Kernel::Linear => (0, 0.0),
            Kernel::Rbf { gamma } => (1, gamma),
        };

        file.write_all(&[kernel_type]).map_err(|e| e.to_string())?;
        file.write_all(&gamma.to_le_bytes()).map_err(|e| e.to_string())?;
        file.write_all(&self.bias.to_le_bytes()).map_err(|e| e.to_string())?;

        let n_sv = self.support_vectors.rows as u32;
        let n_features = self.support_vectors.cols as u32;
        file.write_all(&n_sv.to_le_bytes()).map_err(|e| e.to_string())?;
        file.write_all(&n_features.to_le_bytes()).map_err(|e| e.to_string())?;

        for &w in &self.sv_weights {
            file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        for &v in &self.support_vectors.data {
            file.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }

        Ok(())    
    }


    pub fn read_from(file: &mut File) -> Result<SVM, String> {
        let read_u8 = |file: &mut File| -> Result<u8, String> {
            let mut buf = [0u8; 1];
            file.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf[0])
        };
        let read_u32 = |file: &mut File| -> Result<u32, String> {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(u32::from_le_bytes(buf))
        };
        let read_f32 = |file: &mut File| -> Result<f32, String> {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(f32::from_le_bytes(buf))
        };

        let kernel_type = read_u8(file)?;
        let gamma = read_f32(file)?;
        let bias = read_f32(file)?;

        let kernel = match kernel_type {
            0 => Kernel::Linear,
            1 => Kernel::Rbf { gamma },
            other => return Err(format!("Unknown kernel type byte: {}", other)),
        };

        let n_sv = read_u32(file)? as usize;
        let n_features = read_u32(file)? as usize;

        let mut sv_weights = Vec::with_capacity(n_sv);
        for _ in 0..n_sv {
            sv_weights.push(read_f32(file)?);
        }

        let mut sv_data = Vec::with_capacity(n_sv * n_features);
        for _ in 0..(n_sv * n_features) {
            sv_data.push(read_f32(file)?);
        }
        let support_vectors = Matrix::from_vec(sv_data, n_sv, n_features);

        Ok(SVM { support_vectors, sv_weights, bias, kernel })
    }
}


#[cfg(test)]
mod tests {
    use super::*;


    fn toy_linear_dataset() -> (Matrix, Vec<f32>) {
        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            0.2, 0.1,
            0.1, 0.2,
            5.0, 5.0,
            5.2, 4.9,
            4.9, 5.1,
        ], 6, 2);
        let y = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn test_linear_svm_fits_training_data() {

        let (x, y) = toy_linear_dataset();
        let svm = SVM::train(&x, &y, Kernel::Linear);
        let acc = svm.accuracy(&x, &y);
        assert!(acc > 0.99, "expected clean separation, got {}", acc);
    }

    #[test]
    fn test_svm_uses_fewer_support_vectors_than_examples() {

        let (x, y) = toy_linear_dataset();
        let svm = SVM::train(&x, &y, Kernel::Linear);
        assert!(
            svm.n_support_vectors() < x.rows,
            "expected sparsity, got {} support vectors out of {} examples",
            svm.n_support_vectors(), x.rows
        );
    }

    #[test]
    fn test_bias_near_zero_for_symmetric_data() {

        let x = Matrix::from_vec(vec![
            -1.0, 0.0,
             1.0, 0.0,
        ], 2, 2);
        let y = vec![-1.0, 1.0];
        let svm = SVM::train(&x, &y, Kernel::Linear);
        assert!(svm.bias.abs() < 0.5, "expected small bias, got {}", svm.bias);
    }

    #[test]
    fn test_rbf_kernel_solves_nonlinear_case() {

        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            0.1, 0.1,
            -0.1, 0.1,
            5.0, 0.0,
            0.0, 5.0,
            -5.0, 0.0,
            0.0, -5.0,
        ], 7, 2);
        let y = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];

        let svm = SVM::train(&x, &y, Kernel::Rbf { gamma: 0.5 });
        let acc = svm.accuracy(&x, &y);
        assert!(acc > 0.85, "RBF kernel should handle this, got {}", acc);
    }

    #[test]
    fn test_predict_length_matches_input() {
        let (x, y) = toy_linear_dataset();
        let svm = SVM::train(&x, &y, Kernel::Linear);
        let preds = svm.predict(&x);
        assert_eq!(preds.len(), x.rows);
    }

    #[test]
    fn test_predictions_are_valid_labels() {
        let (x, y) = toy_linear_dataset();
        let svm = SVM::train(&x, &y, Kernel::Linear);
        for p in svm.predict(&x) {
            assert!(p == 1.0 || p == -1.0, "prediction should be +-1, got {}", p);
        }
    }

    #[test]
    fn test_write_read_roundtrip() {
        use std::fs::File;

        let (x, y) = toy_linear_dataset();
        let original = SVM::train(&x, &y, Kernel::Linear);

        let path = "/tmp/test_svm_roundtrip.bin";
        {
            let mut file = File::create(path).unwrap();
            original.write_to(&mut file).unwrap();
        }
        let loaded = {
            let mut file = File::open(path).unwrap();
            SVM::read_from(&mut file).unwrap()
        };

        assert_eq!(original.predict(&x), loaded.predict(&x));
        assert_eq!(original.n_support_vectors(), loaded.n_support_vectors());
        assert!((original.bias - loaded.bias).abs() < 1e-6);
    }
}
