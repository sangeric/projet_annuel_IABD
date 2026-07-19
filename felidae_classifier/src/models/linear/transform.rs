// src/models/linear/transform.rs

use crate::tensor::Matrix;


pub fn polynomial_transform(x: &Matrix) -> Matrix {
    assert_eq!(
        x.cols, 2,
        "polynomial_transform expects 2 input features, got {}",
        x.cols
    );

    let n_samples = x.rows;
    let n_output_features = 5;

    let mut result_data = Vec::new();

    for i in 0..n_samples {
        let xi = x.get(i, 0);
        let yi = x.get(i, 1);

        result_data.push(xi);
        result_data.push(yi);
        result_data.push(xi * yi);
        result_data.push(xi * xi);
        result_data.push(yi * yi);
    }

    Matrix::from_vec(result_data, n_samples, n_output_features)
}


pub fn transformed_feature_count() -> usize {
    5
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_shape() {

        let x = Matrix::from_vec(vec![
            0.1, 0.9,
            0.9, 0.9,
            0.5, 0.5,
        ], 3, 2);

        let transformed = polynomial_transform(&x);
        assert_eq!(transformed.rows, 3);
        assert_eq!(transformed.cols, 5);
    }

    #[test]
    fn test_transform_values() {

        let x = Matrix::from_vec(vec![0.5, 0.5], 1, 2);
        let t = polynomial_transform(&x);

        assert!((t.get(0, 0) - 0.5).abs() < 1e-6);
        assert!((t.get(0, 1) - 0.5).abs() < 1e-6);
        assert!((t.get(0, 2) - 0.25).abs() < 1e-6);
        assert!((t.get(0, 3) - 0.25).abs() < 1e-6);
        assert!((t.get(0, 4) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_diagonal_separation() {

        let x = Matrix::from_vec(vec![
            0.9, 0.9,
            0.1, 0.1,
        ], 2, 2);

        let t = polynomial_transform(&x);
        let top_right_interaction = t.get(0, 2);
        let bot_left_interaction  = t.get(1, 2);

        assert!(top_right_interaction > bot_left_interaction);
        assert!((top_right_interaction - 0.81).abs() < 1e-6);
        assert!((bot_left_interaction  - 0.01).abs() < 1e-6);
    }
}
