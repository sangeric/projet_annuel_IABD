// src/features/scaler.rs

use crate::tensor::Matrix;

// the standard scaler centres each feature to mean 0 and scales to std 1.
pub struct StandardScaler {
    means: Vec<f32>,
    stds: Vec<f32>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            stds: Vec::new(),
        }
    }

    // Computes mean and std for each column of x
    pub fn fit(&mut self, x: &Matrix) {
        let n_rows = x.rows as f32;

        // per columns means
        let mut means = Vec::new();
        for i in 0..x.cols {
            let mut sum = 0.0_f32;
            for j in 0..x.rows {
                sum += x.get(j, i);
            }
            means.push(sum / n_rows);
        }
        
        // per columns stds
        let mut stds = Vec::new();
        for i in 0..x.cols {
            let mut sum_sd = 0.0_f32;
            for j in 0..x.rows {
                let diff = x.get(j, i) - means[i];
                sum_sd += diff * diff;
            }

            let std = (sum_sd / n_rows).sqrt();

            // Without this, transform() would divide by zero.
            let safe_std = if std < 1e-7 { 1.0 } else { std };

            stds.push(safe_std);
        }
        self.means = means;
        self.stds = stds;
    }

    // Applies the stored scaling to a matrix.
    // Each value becomes (x - column_mean) / column_std.
    // Returns a new Matrix; does not modify the input.
    pub fn transform(&self, x: &Matrix) -> Matrix {
        assert_eq!(x.cols, self.means.len(), "transform called with {} columns but scaler was fit with {}", x.cols, self.means.len());
        
        let mut result_data = Vec::new();
        for i in 0..x.rows {
            for j in 0..x.cols {
                let scaled = (x.get(i, j) - self.means[j]) / self.stds[j];
                result_data.push(scaled);
            }
        }

        Matrix::from_vec(result_data, x.rows, x.cols)
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> Matrix {
        self.fit(x);
        self.transform(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_matrix() -> Matrix {
        Matrix::from_vec(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2)
    }

    #[test]
    fn test_fit_means() {
        let mut scaler = StandardScaler::new();
        let x = make_test_matrix();
        scaler.fit(&x);

        assert!((scaler.means[0] - 2.0).abs() < 1e-5);
        assert!((scaler.means[1] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_transform_produces_zero_mean() {
        // After transform, each column of the training data should have mean ~0
        let mut scaler = StandardScaler::new();
        let x = make_test_matrix();
        let scaled = scaler.fit_transform(&x);

        for j in 0..scaled.cols {
            let mut sum = 0.0_f32;
            for i in 0..scaled.rows {
                sum += scaled.get(i, j);
            }
            let mean = sum / scaled.rows as f32;
            assert!(mean.abs() < 1e-5, "column {} mean is {}, expected ~0", j, mean);
        }
    }

    #[test]
    fn test_transform_produces_unit_std() {
        // After transform, each column should have std ~1
        let mut scaler = StandardScaler::new();
        let x = make_test_matrix();
        let scaled = scaler.fit_transform(&x);

        for j in 0..scaled.cols {
            let mut sum_sq = 0.0_f32;
            for i in 0..scaled.rows {
                sum_sq += scaled.get(i, j) * scaled.get(i, j);
            }
            let std = (sum_sq / scaled.rows as f32).sqrt();
            assert!((std - 1.0).abs() < 1e-5, "column {} std is {}, expected ~1", j, std);
        }
    }

    #[test]
    fn test_constant_column_does_not_crash() {
        // A column that's the same everywhere has std 0 — must not divide by zero
        let x = Matrix::from_vec(vec![5.0, 1.0, 5.0, 2.0, 5.0, 3.0], 3, 2);
        let mut scaler = StandardScaler::new();
        let scaled = scaler.fit_transform(&x);

        // The constant column should now be all zeros (everything - mean = 0)
        for i in 0..scaled.rows {
            assert!(scaled.get(i, 0).abs() < 1e-5);
        }
    }
}
