// src/models/svm/kernel.rs

use crate::tensor::Matrix;
use crate::models::rbfn::basis::{gaussian, squared_distance};

#[derive(Clone, Copy)]
pub enum Kernel {
    Linear,
    Rbf { gamma: f32 },
}

impl Kernel {
    pub fn compute(&self, x: &Matrix, a: usize, b: usize) -> f32 {
        match self {
            Kernel::Linear => dot_rows(x, a, x, b),
            Kernel::Rbf { gamma } => {
                let dist_sq = squared_distance(x, a, x, b);
                gaussian(dist_sq, *gamma)
            }
        }
    }

    pub fn compute_cross(&self, x: &Matrix, a: usize, other: &Matrix, b: usize) -> f32 {
        match self {
            Kernel::Linear => dot_rows(x, a, other, b),
            Kernel::Rbf { gamma } => {
                let dist_sq = squared_distance(x, a, other, b);
                gaussian(dist_sq, *gamma)
            }
        }
    }
}

fn dot_rows(x: &Matrix, a: usize, other: &Matrix, b: usize) -> f32 {
    let mut sum = 0.0_f32;
    for j in 0..x.cols {
        sum += x.get(a, j) * other.get(b, j);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_kernel_matches_dot_product() {

        let x = Matrix::from_vec(vec![3.0, 4.0], 1, 2);
        let k = Kernel::Linear;
        let val = k.compute(&x, 0, 0);
        assert!((val - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_rbf_kernel_is_one_at_same_point() {
        let x = Matrix::from_vec(vec![1.0, 2.0, 5.0, 6.0], 2, 2);
        let k = Kernel::Rbf { gamma: 1.0 };
        let val = k.compute(&x, 0, 0);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rbf_kernel_decays_with_distance() {
        let x = Matrix::from_vec(vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0], 3, 2);
        let k = Kernel::Rbf { gamma: 1.0 };
        let near = k.compute(&x, 0, 1);
        let far = k.compute(&x, 0, 2);
        assert!(near > far);
    }

    #[test]
    fn test_compute_cross_matches_compute_on_same_matrix() {
        let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let k = Kernel::Rbf { gamma: 0.5 };
        let a = k.compute(&x, 0, 1);
        let b = k.compute_cross(&x, 0, &x, 1);
        assert!((a - b).abs() < 1e-6);
    }
}
