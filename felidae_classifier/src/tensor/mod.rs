// src/tensor/mod.rs


#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let mut data = Vec::new();
        for _ in 0..rows * cols {
            data.push(0.0_f32);
        }
        Self { data, rows, cols }
    }


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


    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }


    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * self.cols + col] = val;
    }



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


    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }

        result
    }


    pub fn add(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut result_data = Vec::new();

        for i in 0..self.data.len() {
            result_data.push(self.data[i] + rhs.data[i]);
        }

        Matrix::from_vec(result_data, self.rows, self.cols)
    }


    pub fn sub(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut result_data = Vec::new();

        for i in 0..self.data.len() {
            result_data.push(self.data[i] - rhs.data[i]);
        }

        Matrix::from_vec(result_data, self.rows, self.cols)
    }

    pub fn mul_elementwise(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut result_data = Vec::new();

        for i in 0..self.data.len() {
            result_data.push(self.data[i] * rhs.data[i]);
        }

        Matrix::from_vec(result_data, self.rows, self.cols)
    }


    pub fn scale(&self, factor: f32) -> Matrix {
        let mut result_data = Vec::new();

        for i in 0..self.data.len() {
            result_data.push(self.data[i] * factor);
        }

        Matrix::from_vec(result_data, self.rows, self.cols)
    }


    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Matrix {
        let mut result_data = Vec::new();

        for i in 0..self.data.len() {
            result_data.push(f(self.data[i]));
        }

        Matrix::from_vec(result_data, self.rows, self.cols)
    }


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

    pub fn inverse(&self) -> Matrix {
        assert_eq!(self.rows, self.cols, "Matrix must be square to invert");
        let n = self.rows;

        let mut aug = Matrix::zeros(n, n * 2);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, i + n, 1.0);
        }

        for col in 0..n {

            let pivot = aug.get(col, col);
            assert!(pivot.abs() > 1e-10, "Matrix is singular, cannot invert");

            for j in 0..n * 2 {
                let val = aug.get(col, j) / pivot;
                aug.set(col, j, val);
            }

            for row in 0..n {
                if row != col {
                    let factor = aug.get(row, col);
                    for j in 0..n * 2 {
                        let val = aug.get(row, j) - factor * aug.get(col, j);
                        aug.set(row, j, val);
                    }
                }
            }
        }

        let mut result = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                result.set(i, j, aug.get(i, j + n));
            }
        }
        result
    }
}


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

        let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.dot(&b);
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_transpose() {

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
