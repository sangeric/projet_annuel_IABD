// src/tensor/mod.rs

/// A row-major matrix of f32 values.
///
/// Memory layout: element at row i, col j is at index i * cols + j
/// This is the same as C's 2D arrays — unlike numpy which can be either.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    /// Creates a matrix filled with zeros — like numpy.zeros((rows, cols))
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let mut data = Vec::new();
        for _ in 0..rows * cols {
            data.push(0.0_f32);
        }
        Self { data, rows, cols }
    }

    /// Creates a matrix from an existing flat Vec — you must provide the right shape.
    /// Like numpy.array([...]).reshape(rows, cols)
    pub fn from_vec(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} doesn't match shape {}x{}",
            data.len(),
            rows,
            cols
        );
        Self { data, rows, cols }
    }

    /// Read element at (row, col) — immutable borrow, like matrix[i][j] in C
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Write element at (row, col) — requires mutable borrow
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * self.cols + col] = val;
    }

    /// Matrix multiplication (dot product) — self is (M x K), rhs is (K x N), result is (M x N)
    /// Like numpy.dot(a, b) or a @ b
    pub fn dot(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(
            self.cols,
            rhs.rows,
            "Shape mismatch: ({}x{}) dot ({}x{})",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );

        let mut result = Matrix::zeros(self.rows, rhs.cols);

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = 0.0_f32;
                for k in 0..self.cols {
                    sum += self.get(i, k) * rhs.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        result
    }

    /// Transpose — like numpy.T
    /// Turns (M x N) into (N x M)
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }

        result
    }

    /// Elementwise addition — both matrices must have the same shape
    /// Like numpy: a + b
    pub fn add(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut result_data = Vec::new();

        for i in 0..self.data.len() {
            result_data.push(self.data[i] + rhs.data[i]);
        }

        Matrix::from_vec(result_data, self.rows, self.cols)
    }

    /// Elementwise subtraction — like numpy: a - b
    pub fn sub(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut result_data = Vec::new();

        for i in 0..self.data.len() {
            result_data.push(self.data[i] - rhs.data[i]);
        }

        Matrix::from_vec(result_data, self.rows, self.cols)
    }

    /// Elementwise multiplication (Hadamard product) — like numpy: a * b
    pub fn mul_elementwise(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut result_data = Vec::new();

        for i in 0..self.data.len() {
            result_data.push(self.data[i] * rhs.data[i]);
        }

        Matrix::from_vec(result_data, self.rows, self.cols)
    }

    /// Multiply every element by a scalar — like numpy: a * 0.01
    pub fn scale(&self, factor: f32) -> Matrix {
        let mut result_data = Vec::new();

        for i in 0..self.data.len() {
            result_data.push(self.data[i] * factor);
        }

        Matrix::from_vec(result_data, self.rows, self.cols)
    }

    /// Apply any function to every element — like numpy: np.vectorize(f)(matrix)
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Matrix {
        let mut result_data = Vec::new();

        for i in 0..self.data.len() {
            result_data.push(f(self.data[i]));
        }

        Matrix::from_vec(result_data, self.rows, self.cols)
    }

    /// Add a bias vector (1 x cols) to every row of self (rows x cols)
    /// Used constantly in neural networks: output = input.dot(weights) + bias
    pub fn add_bias_row(&self, bias: &Matrix) -> Matrix {
        assert_eq!(bias.rows, 1);
        assert_eq!(bias.cols, self.cols);

        let mut result = self.clone();

        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = result.get(i, j) + bias.get(0, j);
                result.set(i, j, val);
            }
        }

        result
    }
}

// ---- Tests ----------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m = Matrix::zeros(2, 3);
        for i in 0..m.data.len() {
            assert_eq!(m.data[i], 0.0);
        }
    }

    #[test]
    fn test_dot() {
        // [1, 2]   [5, 6]   [19, 22]
        // [3, 4] × [7, 8] = [43, 50]
        let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.dot(&b);
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_transpose() {
        // [1, 2, 3]     [1, 4]
        // [4, 5, 6]  →  [2, 5]
        //               [3, 6]
        let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 1), 4.0);
    }

    #[test]
    fn test_add() {
        let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.add(&b);
        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(1, 1), 12.0);
    }

    #[test]
    fn test_sub() {
        let a = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let b = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let c = a.sub(&b);
        assert_eq!(c.get(0, 0), 4.0);
        assert_eq!(c.get(1, 1), 4.0);
    }

    #[test]
    fn test_mul_elementwise() {
        // [1, 2]   [5, 6]   [5,  12]
        // [3, 4] * [7, 8] = [21, 32]
        let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.mul_elementwise(&b);
        assert_eq!(c.get(0, 0), 5.0);
        assert_eq!(c.get(0, 1), 12.0);
        assert_eq!(c.get(1, 0), 21.0);
        assert_eq!(c.get(1, 1), 32.0);
    }

    #[test]
    fn test_scale() {
        let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = a.scale(2.0);
        assert_eq!(b.get(0, 0), 2.0);
        assert_eq!(b.get(1, 1), 8.0);
    }

    #[test]
    fn test_add_bias_row() {
        let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let bias = Matrix::from_vec(vec![10.0, 20.0], 1, 2);
        let result = m.add_bias_row(&bias);
        assert_eq!(result.get(0, 0), 11.0);
        assert_eq!(result.get(0, 1), 22.0);
        assert_eq!(result.get(1, 0), 13.0);
        assert_eq!(result.get(1, 1), 24.0);
    }
}
