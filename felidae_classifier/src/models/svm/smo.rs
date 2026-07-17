// src/models/svm/smo.rs


use crate::tensor::Matrix;
use crate::models::svm::kernel::Kernel;
use osqp::{CscMatrix, Problem, Settings};

/// Solves the SVM dual QP from the slides:
///
///   minimize (1/2) alpha^T Q alpha + (-1...-1)^T alpha
///   subject to Y^T alpha = 0
///              alpha >= 0
///
/// where Q[n][m] = y_n * y_m * K(X_n, X_m).
///
/// x is the training examples (N x n_features), y is +1/-1 labels (length N).
/// Returns the solved alpha vector (length N).
///
/// c is the soft-margin penalty. A large c approaches hard-margin
/// (little tolerance for violations); a small c allows more margin
/// violations in exchange for a smoother boundary

pub fn solve_dual(x: &Matrix, y: &[f32], kernel: &Kernel, c: f32) -> Vec<f32> {
    let n = x.rows;

    // Q[i][j] = y_i * y_j * K(x_i, x_j)
    let mut q_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let k_ij = kernel.compute(x, i, j) as f64;
            q_matrix[i][j] = (y[i] * y[j]) as f64 * k_ij;
        }
    }

    for i in 0..n {
        q_matrix[i][i] += 1e-8;
    }

    let linear_term: Vec<f64> = vec![-1.0; n];

    // Row 0: the equality constraint  Y^T alpha = 0   -> l = u = 0
    // Rows 1..=n: the box constraints alpha_i >= 0     -> l = 0, u = +inf
    let mut a_matrix: Vec<Vec<f64>> = Vec::with_capacity(n + 1);

    // equality constraint row: coefficients are just the labels
    let equality_row: Vec<f64> = y.iter().map(|&yi| yi as f64).collect();
    a_matrix.push(equality_row);

    // one row per alpha_i >= 0 (identity matrix)
    for i in 0..n {
        let mut row = vec![0.0; n];
        row[i] = 1.0;
        a_matrix.push(row);
    }

    let mut lower_bounds: Vec<f64> = Vec::with_capacity(n + 1);
    let mut upper_bounds: Vec<f64> = Vec::with_capacity(n + 1);

    // equality: 0 <= (Y^T alpha) <= 0
    lower_bounds.push(0.0);
    upper_bounds.push(0.0);

    // inequalities: 0 <= alpha_i <= +infinity
    for _ in 0..n {
        lower_bounds.push(0.0);
        upper_bounds.push(c as f64);
    }

    // P must be square, structurally upper triangular, in CSC format.
    // ---- Hand everything to OSQP ----
    let p_csc = CscMatrix::from(&q_matrix[..]).into_upper_tri();

    let settings = Settings::default().verbose(false);

    let mut problem = Problem::new(
        p_csc,
        &linear_term,
        &a_matrix[..],
        &lower_bounds,
        &upper_bounds,
        &settings,
    )
    .expect("failed to set up OSQP problem");

    let result = problem.solve();
    let solution = result.solution().expect("OSQP failed to find a solution");
    let alpha_f64 = solution.x();

    alpha_f64.iter().map(|&a| a.max(0.0) as f32).collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_point_symmetric_case() {
        let x = Matrix::from_vec(vec![1.0, -1.0], 2, 1);
        let y = vec![1.0, -1.0];
        let kernel = Kernel::Linear;

        let alpha = solve_dual(&x, &y, &kernel);

        assert_eq!(alpha.len(), 2);
        // Both points are support vectors in this symmetric case.
        assert!(alpha[0] > 1e-3, "alpha[0] should be nonzero, got {}", alpha[0]);
        assert!(alpha[1] > 1e-3, "alpha[1] should be nonzero, got {}", alpha[1]);
        // By symmetry the two alphas should be equal.
        assert!((alpha[0] - alpha[1]).abs() < 1e-2);
    }

    #[test]
    fn test_alpha_is_never_negative() {
        // A slightly less trivial dataset — alpha must always respect alpha >= 0
        // regardless of geometry.
        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            5.0, 5.0,
            6.0, 5.0,
            5.0, 6.0,
        ], 6, 2);
        let y = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let kernel = Kernel::Linear;

        let alpha = solve_dual(&x, &y, &kernel);

        for &a in &alpha {
            assert!(a >= -1e-6, "found a negative alpha: {}", a);
        }
    }

    #[test]
    fn test_equality_constraint_is_satisfied() {
        // sum(alpha_n * y_n) must equal 0 — this is Y^T alpha = 0 from the slides.
        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            5.0, 5.0,
            6.0, 5.0,
            5.0, 6.0,
        ], 6, 2);
        let y = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let kernel = Kernel::Linear;

        let alpha = solve_dual(&x, &y, &kernel);

        let sum: f32 = alpha.iter().zip(y.iter()).map(|(&a, &yi)| a * yi).sum();
        assert!(sum.abs() < 1e-2, "Y^T alpha should be ~0, got {}", sum);
    }

    #[test]
    fn test_far_points_have_near_zero_alpha() {
        // Two clusters far apart, plus one point buried deep inside its own
        // cluster (not near the boundary). That deep point should NOT be a
        // support vector -> its alpha should be near zero.
        let x = Matrix::from_vec(vec![
            0.0, 0.0,     // near the boundary
            -10.0, -10.0, // deep inside class -1, far from the boundary
            1.0, 1.0,     // near the boundary
            11.0, 11.0,   // deep inside class +1, far from the boundary
        ], 4, 2);
        let y = vec![-1.0, -1.0, 1.0, 1.0];
        let kernel = Kernel::Linear;

        let alpha = solve_dual(&x, &y, &kernel);

        // the deep points (index 1 and 3) should contribute far less than
        // the boundary points (index 0 and 2)
        assert!(alpha[1] < alpha[0], "deep point should have smaller alpha than boundary point");
        assert!(alpha[3] < alpha[2], "deep point should have smaller alpha than boundary point");
    }

    #[test]
    fn test_rbf_kernel_solves_without_error() {
        // Just a smoke test that the RBF kernel path works end-to-end
        // through OSQP without panicking, on a small non-trivial dataset.
        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            0.2, 0.1,
            5.0, 5.0,
            5.2, 4.9,
        ], 4, 2);
        let y = vec![-1.0, -1.0, 1.0, 1.0];
        let kernel = Kernel::Rbf { gamma: 1.0 };

        let alpha = solve_dual(&x, &y, &kernel);
        assert_eq!(alpha.len(), 4);
        for &a in &alpha {
            assert!(a.is_finite());
        }
    }
}
