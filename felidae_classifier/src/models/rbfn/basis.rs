// src/models/rbfn/basis.rs

use crate::tensor::Matrix;

pub fn squared_distance(x: &Matrix, a: usize, centers: &Matrix, b: usize) -> f32 {
    let mut sum = 0.0_f32;
    for j in 0..x.cols {
        let diff = x.get(a, j) - centers.get(b, j);
        sum += diff * diff;
    }
    sum
}


pub fn gaussian(dist_squared: f32, gamma: f32) -> f32 {
    (-gamma * dist_squared).exp()
}


pub fn build_phi(x: &Matrix, centers: &Matrix, gamma: f32) -> Matrix {
    let mut phi = Matrix::zeros(x.rows, centers.rows);

    for i in 0..x.rows {
        for j in 0..centers.rows {
            let dist_sq = squared_distance(x, i, centers, j);
            phi.set(i, j, gaussian(dist_sq, gamma));
        }
    }
    phi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_distance_zero_for_same_point() {

        let x = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
        let d = squared_distance(&x, 0, &x, 0);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_squared_distance_known_value() {

        let a = Matrix::from_vec(vec![0.0, 0.0], 1, 2);
        let b = Matrix::from_vec(vec![3.0, 4.0], 1, 2);
        let d = squared_distance(&a, 0, &b, 0);
        assert!((d - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_is_one_at_center() {

        let g = gaussian(0.0, 5.0);
        assert!((g - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_decays_with_distance() {

        let near = gaussian(1.0, 1.0);
        let far = gaussian(10.0, 1.0);
        assert!(near > far);
        assert!(far > 0.0);
    }

    #[test]
    fn test_gaussian_larger_gamma_decays_faster() {

        let small_gamma = gaussian(1.0, 0.1);
        let large_gamma = gaussian(1.0, 10.0);
        assert!(large_gamma < small_gamma);
    }

    #[test]
    fn test_build_phi_shape() {

        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            1.0, 1.0,
            2.0, 2.0,
            3.0, 3.0,
        ], 4, 2);
        let centers = Matrix::from_vec(vec![
            0.0, 0.0,
            3.0, 3.0,
        ], 2, 2);
        let phi = build_phi(&x, &centers, 1.0);
        assert_eq!(phi.rows, 4);
        assert_eq!(phi.cols, 2);
    }

    #[test]
    fn test_build_phi_diagonal_is_one_when_centers_are_examples() {

        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            5.0, 5.0,
        ], 2, 2);
        let phi = build_phi(&x, &x, 1.0);
        assert!((phi.get(0, 0) - 1.0).abs() < 1e-6);
        assert!((phi.get(1, 1) - 1.0).abs() < 1e-6);

        assert!(phi.get(0, 1) < 1.0);
    }
}
