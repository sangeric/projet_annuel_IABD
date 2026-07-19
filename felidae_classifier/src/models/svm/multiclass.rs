// src/models/svm/multiclass.rs

use crate::tensor::Matrix;
use crate::models::svm::kernel::Kernel;
use crate::models::svm::SVM;
use std::fs::File;
use std::io::{Write, Read};

pub struct MulticlassSVM {
    classifiers: Vec<SVM>,
}

impl MulticlassSVM {
    pub fn train(x: &Matrix, labels: &[usize], n_classes: usize, kernel: Kernel, c: f32) -> Self {
        let mut classifiers = Vec::with_capacity(n_classes);

        for class in 0..n_classes {
            let binary_y: Vec<f32> = labels.iter().map(|&label| if label == class { 1.0 } else { -1.0 }).collect();

            let svm = SVM::train(x, &binary_y, kernel, c);
            classifiers.push(svm);
        }

        Self { classifiers }
    }

    pub fn forward(&self, x: &Matrix) -> Vec<Vec<f32>> {
        let per_classifier: Vec<Vec<f32>> = self.classifiers.iter().map(|svm| svm.forward(x)).collect();

        let n_examples = x.rows;
        let n_classes = self.classifiers.len();
        let mut result = vec![vec![0.0_f32; n_classes]; n_examples];
        for class in 0..n_classes {
            for example in 0..n_examples {
                result[example][class] = per_classifier[class][example];
            }
        }
        result
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        let scores = self.forward(x);
        scores
            .into_iter()
            .map(|row| {
                let mut best_class = 0;
                let mut best_value = row[0];
                for (class, &value) in row.iter().enumerate().skip(1) {
                    if value > best_value {
                        best_value = value;
                        best_class = class;
                    }
                }
                best_class
            })
            .collect()
    }

    pub fn accuracy(&self, x: &Matrix, labels: &[usize]) -> f32 {
        let preds = self.predict(x);
        let mut correct = 0.0_f32;
        for i in 0..labels.len() {
            if preds[i] == labels[i] {
                correct += 1.0;
            }
        }
        correct / labels.len() as f32
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;

        file.write_all(b"SVM1").map_err(|e| e.to_string())?;
        let version: u32 = 1;
        file.write_all(&version.to_le_bytes()).map_err(|e| e.to_string())?;

        let n_classes = self.classifiers.len() as u32;
        file.write_all(&n_classes.to_le_bytes()).map_err(|e| e.to_string())?;

        for svm in &self.classifiers {
            svm.write_to(&mut file)?;
        }

        Ok(())
    }


    pub fn load(path: &str) -> Result<MulticlassSVM, String> {
        let mut file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"SVM1" {
            return Err(format!("Not an SVM model file: bad magic number {:?}", magic));
        }

        let mut version_buf = [0u8; 4];
        file.read_exact(&mut version_buf).map_err(|e| e.to_string())?;
        let version = u32::from_le_bytes(version_buf);
        if version != 1 {
            return Err(format!("Unsupported SVM file version: {}", version));
        }

        let mut n_classes_buf = [0u8; 4];
        file.read_exact(&mut n_classes_buf).map_err(|e| e.to_string())?;
        let n_classes = u32::from_le_bytes(n_classes_buf) as usize;

        let mut classifiers = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            classifiers.push(SVM::read_from(&mut file)?);
        }

        Ok(MulticlassSVM { classifiers })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn toy_three_class_dataset() -> (Matrix, Vec<usize>) {
        let x = Matrix::from_vec(vec![

            0.1, 0.1,
            0.2, 0.15,
            0.15, 0.2,

            0.5, 0.9,
            0.55, 0.85,
            0.45, 0.88,

            0.9, 0.1,
            0.85, 0.15,
            0.88, 0.2,
        ], 9, 2);
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        (x, labels)
    }

    #[test]
    fn test_trains_one_classifier_per_class() {
        let (x, labels) = toy_three_class_dataset();
        let model = MulticlassSVM::train(&x, &labels, 3, Kernel::Linear);
        assert_eq!(model.classifiers.len(), 3);
    }

    #[test]
    fn test_fits_well_separated_data_perfectly() {

        let (x, labels) = toy_three_class_dataset();
        let model = MulticlassSVM::train(&x, &labels, 3, Kernel::Linear);
        let acc = model.accuracy(&x, &labels);
        assert!(acc > 0.99, "expected near-perfect fit, got {}", acc);
    }

    #[test]
    fn test_predict_returns_valid_class_indices() {
        let (x, labels) = toy_three_class_dataset();
        let model = MulticlassSVM::train(&x, &labels, 3, Kernel::Linear);
        for p in model.predict(&x) {
            assert!(p < 3, "predicted class {} out of range", p);
        }
    }

    #[test]
    fn test_predict_length_matches_input() {
        let (x, labels) = toy_three_class_dataset();
        let model = MulticlassSVM::train(&x, &labels, 3, Kernel::Linear);
        let preds = model.predict(&x);
        assert_eq!(preds.len(), x.rows);
    }

    #[test]
    fn test_forward_shape() {

        let (x, labels) = toy_three_class_dataset();
        let model = MulticlassSVM::train(&x, &labels, 3, Kernel::Linear);
        let scores = model.forward(&x);
        assert_eq!(scores.len(), x.rows);
        for row in &scores {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn test_argmax_picks_least_negative_when_all_reject() {

        let (x, labels) = toy_three_class_dataset();
        let model = MulticlassSVM::train(&x, &labels, 3, Kernel::Linear);


        let query = Matrix::from_vec(vec![0.05, 0.3], 1, 2);
        let scores = model.forward(&query);


        let class0_score = scores[0][0];
        let class1_score = scores[0][1];
        let class2_score = scores[0][2];
        assert!(class0_score > class1_score);
        assert!(class0_score > class2_score);

        let pred = model.predict(&query);
        assert_eq!(pred[0], 0, "expected class 0 to win as the least-rejecting classifier");
    }

    #[test]
    fn test_rbf_kernel_handles_nonlinear_three_class_case() {

        let x = Matrix::from_vec(vec![

            0.1, 0.1,
            0.9, 0.9,

            0.9, 0.1,
            0.1, 0.9,

            0.5, 0.5,
            0.52, 0.48,
        ], 6, 2);
        let labels = vec![0, 0, 1, 1, 2, 2];

        let model = MulticlassSVM::train(&x, &labels, 3, Kernel::Rbf { gamma: 5.0 });
        let acc = model.accuracy(&x, &labels);
        assert!(acc > 0.8, "RBF kernel should handle this reasonably, got {}", acc);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let (x, labels) = toy_three_class_dataset();
        let original = MulticlassSVM::train(&x, &labels, 3, Kernel::Linear);

        let path = "/tmp/test_multiclass_svm_roundtrip.bin";
        original.save(path).expect("save failed");
        let loaded = MulticlassSVM::load(path).expect("load failed");

        assert_eq!(original.predict(&x), loaded.predict(&x));
    }

    #[test]
    fn test_load_bad_magic_fails() {
        std::fs::write("/tmp/test_svm_bad_magic.bin", b"NOTASVM").unwrap();
        assert!(MulticlassSVM::load("/tmp/test_svm_bad_magic.bin").is_err());
    }
}
