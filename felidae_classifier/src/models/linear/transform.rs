// src/models/linear/transform.rs

use crate::tensor::Matrix;

/// Applies a non-linear transformation to the input matrix.
///
/// Maps each 2D point [x, y] to a 5D point [x, y, x*y, x², y²]
/// This makes the XOR-like dataset linearly separable, which the
/// raw [x, y] inputs are not.
///
/// Input shape:  (N × 2)
/// Output shape: (N × 5)
pub fn polynomial_transform(x: &Matrix) -> Matrix {
    assert_eq!(
        x.cols, 2,
        "polynomial_transform expects 2 input features, got {}",
        x.cols
    );

    let n_samples = x.rows;
    let n_output_features = 5; // [x, y, x*y, x², y²]

    let mut result_data = Vec::new();

    for i in 0..n_samples {
        let xi = x.get(i, 0); // first coordinate
        let yi = x.get(i, 1); // second coordinate

        result_data.push(xi);        // x        — original feature 1
        result_data.push(yi);        // y        — original feature 2
        result_data.push(xi * yi);   // x*y      — interaction term
        result_data.push(xi * xi);   // x²       — squared feature 1
        result_data.push(yi * yi);   // y²       — squared feature 2
    }

    Matrix::from_vec(result_data, n_samples, n_output_features)
}

/// Returns the number of features after transformation.
/// Used by LinearClassifier::new() to set the right input size.
/// Input features: 2 → Output features: 5
pub fn transformed_feature_count() -> usize {
    5
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_shape() {
        // 3 samples, 2 features → should become 3 samples, 5 features
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
        // Single point [0.5, 0.5]
        // Expected: [0.5, 0.5, 0.25, 0.25, 0.25]
        let x = Matrix::from_vec(vec![0.5, 0.5], 1, 2);
        let t = polynomial_transform(&x);

        assert!((t.get(0, 0) - 0.5).abs() < 1e-6);   // x
        assert!((t.get(0, 1) - 0.5).abs() < 1e-6);   // y
        assert!((t.get(0, 2) - 0.25).abs() < 1e-6);  // x*y
        assert!((t.get(0, 3) - 0.25).abs() < 1e-6);  // x²
        assert!((t.get(0, 4) - 0.25).abs() < 1e-6);  // y²
    }

    #[test]
    fn test_diagonal_separation() {
        // [0.9, 0.9] and [0.1, 0.1] should have very different x*y values
        // [0.9, 0.9] → x*y = 0.81
        // [0.1, 0.1] → x*y = 0.01
        // This is what makes the diagonal classes separable
        let x = Matrix::from_vec(vec![
            0.9, 0.9,
            0.1, 0.1,
        ], 2, 2);

        let t = polynomial_transform(&x);
        let top_right_interaction = t.get(0, 2); // x*y for [0.9, 0.9]
        let bot_left_interaction  = t.get(1, 2); // x*y for [0.1, 0.1]

        assert!(top_right_interaction > bot_left_interaction);
        assert!((top_right_interaction - 0.81).abs() < 1e-6);
        assert!((bot_left_interaction  - 0.01).abs() < 1e-6);
    }
}
