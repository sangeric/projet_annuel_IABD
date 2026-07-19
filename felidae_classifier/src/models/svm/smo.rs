// src/models/svm/smo.rs


use crate::tensor::Matrix;
use crate::models::svm::kernel::Kernel;
use osqp::{CscMatrix, Problem, Settings};



pub fn solve_dual(x: &Matrix, y: &[f32], kernel: &Kernel, c: f32) -> Vec<f32> {
    let n = x.rows;


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


    let mut a_matrix: Vec<Vec<f64>> = Vec::with_capacity(n + 1);


    let equality_row: Vec<f64> = y.iter().map(|&yi| yi as f64).collect();
    a_matrix.push(equality_row);


    for i in 0..n {
        let mut row = vec![0.0; n];
        row[i] = 1.0;
        a_matrix.push(row);
    }

    let mut lower_bounds: Vec<f64> = Vec::with_capacity(n + 1);
    let mut upper_bounds: Vec<f64> = Vec::with_capacity(n + 1);


    lower_bounds.push(0.0);
    upper_bounds.push(0.0);


    for _ in 0..n {
        lower_bounds.push(0.0);
        upper_bounds.push(c as f64);
    }


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

        assert!(alpha[0] > 1e-3, "alpha[0] should be nonzero, got {}", alpha[0]);
        assert!(alpha[1] > 1e-3, "alpha[1] should be nonzero, got {}", alpha[1]);

        assert!((alpha[0] - alpha[1]).abs() < 1e-2);
    }

    #[test]
    fn test_alpha_is_never_negative() {

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

        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            -10.0, -10.0,
            1.0, 1.0,
            11.0, 11.0,
        ], 4, 2);
        let y = vec![-1.0, -1.0, 1.0, 1.0];
        let kernel = Kernel::Linear;

        let alpha = solve_dual(&x, &y, &kernel);


        assert!(alpha[1] < alpha[0], "deep point should have smaller alpha than boundary point");
        assert!(alpha[3] < alpha[2], "deep point should have smaller alpha than boundary point");
    }

    #[test]
    fn test_rbf_kernel_solves_without_error() {

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
